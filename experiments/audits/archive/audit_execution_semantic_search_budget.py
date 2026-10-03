import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from GDesigner.gdt.gtd_framework import GTDFramework
from GDesigner.gdt.proxy_reward_model import ProxyRewardModel
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Helpers
# ============================================================

def topo_key(A):
    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().int().tolist()

    return tuple(
        tuple(int(x) for x in row)
        for row in A
    )


def count_edges(A):
    if isinstance(A, torch.Tensor):
        return int(A.sum().item())

    return int(
        sum(
            sum(int(x) for x in row)
            for row in A
        )
    )


# ============================================================
# Exact runtime semantics for THIS released implementation
#
# Confirmed empirically by 16-position probe:
#
# flat 0  -> 0->1
# flat 1  -> 0->2
# flat 2  -> 0->3
# flat 3  -> 1->0
# flat 4  -> 1->2
# flat 5  -> 1->3
# flat 6  -> 2->0
# flat 7  -> 2->1
# flat 8  -> 2->3
# flat 9  -> 3->0
# flat 10 -> 3->1
# flat 11 -> 3->2
# flat 12-15 ignored
# ============================================================

class RuntimeSemantics:

    def __init__(self, n):
        self.n = n

        # Same effective potential-edge order observed from Graph:
        self.potential_edges = []

        for i in range(n):
            for j in range(n):
                if i != j:
                    self.potential_edges.append(
                        (i, j)
                    )

        self.num_runtime_edges = len(
            self.potential_edges
        )

    # --------------------------------------------------------
    # Stage 1:
    # exact mask-index reinterpretation
    # --------------------------------------------------------

    def map_raw_to_runtime_candidate(self, raw_A):

        raw_key = topo_key(raw_A)

        flat = [
            int(x)
            for row in raw_key
            for x in row
        ]

        runtime_A = [
            [0 for _ in range(self.n)]
            for _ in range(self.n)
        ]

        # Python zip truncates to shortest input.
        #
        # Therefore only the first N(N-1) flattened mask
        # positions are consumed.
        for bit, (src, dst) in zip(
            flat,
            self.potential_edges,
        ):
            if bit:
                runtime_A[src][dst] = 1

        return torch.tensor(
            runtime_A,
            dtype=torch.int64,
        )

    # --------------------------------------------------------
    # Stage 2:
    # cycle filtering in construct_spatial_connection()
    #
    # Edges are considered in potential_spatial_edges order.
    # --------------------------------------------------------

    @staticmethod
    def _path_exists(
        adjacency,
        start,
        target,
    ):
        if start == target:
            return True

        stack = [start]
        visited = set()

        while stack:

            u = stack.pop()

            if u == target:
                return True

            if u in visited:
                continue

            visited.add(u)

            for v, exists in enumerate(
                adjacency[u]
            ):
                if exists:
                    stack.append(v)

        return False

    def map_runtime_to_executed(
        self,
        runtime_A,
    ):

        runtime_A = (
            runtime_A
            .detach()
            .cpu()
            .int()
            .tolist()
        )

        executed = [
            [0 for _ in range(self.n)]
            for _ in range(self.n)
        ]

        for src, dst in self.potential_edges:

            if runtime_A[src][dst] == 0:
                continue

            # Adding src -> dst creates a cycle if
            # dst can already reach src.
            if self._path_exists(
                executed,
                dst,
                src,
            ):
                continue

            executed[src][dst] = 1

        return torch.tensor(
            executed,
            dtype=torch.int64,
        )

    def full_project(self, raw_A):

        mapped = (
            self.map_raw_to_runtime_candidate(
                raw_A
            )
        )

        executed = (
            self.map_runtime_to_executed(
                mapped
            )
        )

        return mapped, executed


# ============================================================
# Optional validation against saved downstream result
# ============================================================

def validate_against_saved_results(
    semantics,
    result_path,
):

    if result_path is None:
        return None

    data = json.load(
        open(
            result_path,
            encoding="utf-8",
        )
    )

    total = 0
    exact = 0
    hamming = []

    for item in data:

        if (
            "Generated_Topology"
            not in item
            or
            "Executed_Topology"
            not in item
        ):
            continue

        raw = item[
            "Generated_Topology"
        ]

        expected = torch.tensor(
            item[
                "Executed_Topology"
            ],
            dtype=torch.int64,
        )

        _, predicted = (
            semantics.full_project(
                raw
            )
        )

        total += 1

        exact += int(
            torch.equal(
                predicted,
                expected,
            )
        )

        hamming.append(
            int(
                (
                    predicted
                    != expected
                )
                .sum()
                .item()
            )
        )

    if total == 0:
        return None

    result = {
        "records":
            total,

        "exact_matches":
            exact,

        "match_rate":
            exact / total,

        "mean_hamming":
            float(
                np.mean(
                    hamming
                )
            ),
    }

    return result


# ============================================================
# Instrumented guider
# ============================================================

class ExecutionSemanticAuditGuider:

    def __init__(
        self,
        proxy_model,
        runtime_semantics,
        K,
        device,
        utility_weight=1.0,
        cost_weight=-0.1,
    ):
        self.proxy = proxy_model
        self.proxy.eval()

        self.runtime = (
            runtime_semantics
        )

        self.K = K
        self.device = device

        self.utility_weight = (
            utility_weight
        )

        self.cost_weight = (
            cost_weight
        )

        self.records = []

    @torch.no_grad()
    def guide(
        self,
        current_At_prob,
        timestep,
        unguided_A0_prediction,
        node_features,
        task_condition,
    ):

        batch_size = (
            unguided_A0_prediction
            .shape[0]
        )

        output = torch.zeros_like(
            unguided_A0_prediction
        )

        for b in range(batch_size):

            probs = (
                unguided_A0_prediction[b]
            )

            # ------------------------------------------------
            # 1. Same Bernoulli candidates as official guider
            # ------------------------------------------------

            candidates = [
                torch.bernoulli(
                    probs
                ).float()
                for _ in range(
                    self.K
                )
            ]

            # ------------------------------------------------
            # 2. Raw-space uniqueness
            # ------------------------------------------------

            raw_keys = [
                topo_key(A)
                for A in candidates
            ]

            K_raw_unique = len(
                set(raw_keys)
            )

            # ------------------------------------------------
            # 3. Runtime mask-interpretation uniqueness
            # ------------------------------------------------

            mapped_candidates = []

            executed_candidates = []

            for A in candidates:

                mapped, executed = (
                    self.runtime
                    .full_project(A)
                )

                mapped_candidates.append(
                    mapped
                )

                executed_candidates.append(
                    executed
                )

            mapped_keys = [
                topo_key(A)
                for A
                in mapped_candidates
            ]

            exec_keys = [
                topo_key(A)
                for A
                in executed_candidates
            ]

            K_mapped_unique = len(
                set(
                    mapped_keys
                )
            )

            K_exec_unique = len(
                set(
                    exec_keys
                )
            )

            # ------------------------------------------------
            # 4. Official Proxy scoring on RAW candidate
            # ------------------------------------------------

            pyg_list = []

            x = node_features[b]

            condition = (
                task_condition[b]
                .unsqueeze(0)
            )

            for A in candidates:

                edge_index, _ = (
                    dense_to_sparse(
                        A.to(
                            self.device
                        )
                    )
                )

                pyg_list.append(
                    Data(
                        x=x.clone().to(
                            self.device
                        ),

                        edge_index=(
                            edge_index
                        ),

                        condition=(
                            condition
                            .clone()
                            .to(
                                self.device
                            )
                        ),
                    )
                )

            proxy_batch = (
                Batch
                .from_data_list(
                    pyg_list
                )
                .to(
                    self.device
                )
            )

            pred = self.proxy(
                proxy_batch
            )

            utility = pred[:, 0]
            cost = pred[:, 1]

            reward = (
                self.utility_weight
                * utility

                +

                self.cost_weight
                * cost
            )

            best_idx = int(
                torch.argmax(
                    reward
                ).item()
            )

            output[b] = candidates[
                best_idx
            ]

            # ------------------------------------------------
            # 5. Search-budget decomposition
            # ------------------------------------------------

            index_alias_loss = (
                1.0
                -
                K_mapped_unique
                / K_raw_unique
                if K_raw_unique > 0
                else 0.0
            )

            cycle_alias_loss = (
                1.0
                -
                K_exec_unique
                / K_mapped_unique
                if K_mapped_unique > 0
                else 0.0
            )

            total_alias_loss = (
                1.0
                -
                K_exec_unique
                / K_raw_unique
                if K_raw_unique > 0
                else 0.0
            )

            selected_exec_key = (
                exec_keys[
                    best_idx
                ]
            )

            first_exec_key = (
                exec_keys[0]
            )

            self.records.append({

                "timestep":
                    int(
                        timestep[b]
                        .item()
                    ),

                "K_nominal":
                    self.K,

                "K_raw_unique":
                    K_raw_unique,

                "K_mapped_unique":
                    K_mapped_unique,

                "K_exec_unique":
                    K_exec_unique,

                "raw_ratio":
                    K_raw_unique
                    / self.K,

                "mapped_ratio":
                    K_mapped_unique
                    / self.K,

                "effective_ratio":
                    K_exec_unique
                    / self.K,

                "index_alias_loss":
                    index_alias_loss,

                "cycle_alias_loss":
                    cycle_alias_loss,

                "total_alias_loss":
                    total_alias_loss,

                "reward_spread":
                    float(
                        (
                            reward.max()
                            - reward.min()
                        )
                        .item()
                    ),

                "selected_idx":
                    best_idx,

                "selected_exec_equals_first":
                    bool(
                        selected_exec_key
                        ==
                        first_exec_key
                    ),
            })

        return output


# ============================================================
# Dataset helper
# ============================================================

def load_questions(
    path,
    limit,
):

    questions = []

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            obj = json.loads(
                line
            )

            q = (
                obj.get(
                    "question"
                )
                or obj.get(
                    "Question"
                )
                or obj.get(
                    "task"
                )
            )

            if q is None:
                continue

            questions.append(
                q
            )

            if (
                len(
                    questions
                )
                >= limit
            ):
                break

    return questions


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--proxy-model",
        default=(
            "proxy_model_gsm8k.pth"
        ),
    )

    parser.add_argument(
        "--diffusion-model",
        default=(
            "diffusion_model_gsm8k.pth"
        ),
    )

    parser.add_argument(
        "--dataset",
        default=(
            "datasets/gsm8k/"
            "gsm8k_test.jsonl"
        ),
    )

    parser.add_argument(
        "--questions",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20264000,
    )

    parser.add_argument(
        "--validate-result",
        default=None,
    )

    parser.add_argument(
        "--output",
        default=(
            "audit_results/"
            "execution_semantic_search_budget/"
            "summary.json"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    n = 4

    runtime = RuntimeSemantics(
        n=n
    )

    # --------------------------------------------------------
    # Optional validation
    # --------------------------------------------------------

    validation = (
        validate_against_saved_results(
            runtime,
            args.validate_result,
        )
    )

    print("=" * 78)
    print("RUNTIME SEMANTICS VALIDATION")
    print("=" * 78)

    if validation is None:

        print(
            "No validation file supplied."
        )

    else:

        print(
            json.dumps(
                validation,
                indent=2,
            )
        )

    # --------------------------------------------------------
    # Proxy
    # --------------------------------------------------------

    proxy = ProxyRewardModel(
        task_cond_input_dim=384,
        node_feature_dim=384,
        condition_dim=128,
        gnn_hidden_dim=32,
        gnn_layers=2,
        mlp_hidden_dim=64,
        num_reward_components=2,
    ).to(device)

    proxy.load_state_dict(
        torch.load(
            args.proxy_model,
            map_location=device,
        )
    )

    proxy.eval()

    # --------------------------------------------------------
    # Diffusion
    # --------------------------------------------------------

    framework = GTDFramework(
        task_cond_input_dim=384,
        node_feature_dim=384,
        condition_dim=128,
        time_embed_dim=128,
        gt_num_layers=2,
        gt_num_heads=2,
        diffusion_num_timesteps=50,
        device=device,
    )

    framework.diffusion_model.load_state_dict(
        torch.load(
            args.diffusion_model,
            map_location=device,
        )
    )

    framework.diffusion_model.to(
        device
    )

    framework.diffusion_model.eval()

    # --------------------------------------------------------
    # Homogeneous MathSolver node features
    # --------------------------------------------------------

    prompt_set = (
        PromptSetRegistry.get(
            "gsm8k"
        )
    )

    profile = (
        prompt_set.get_description(
            "MathSolver"
        )
    )

    emb = np.asarray(
        get_sentence_embedding(
            profile
        ),
        dtype=np.float32,
    )

    node_features = (
        torch
        .from_numpy(
            np.stack(
                [
                    emb.copy()
                    for _ in range(n)
                ],
                axis=0,
            )
        )
        .to(device)
    )

    # --------------------------------------------------------
    # Questions
    # --------------------------------------------------------

    questions = load_questions(
        args.dataset,
        args.questions,
    )

    all_records = []

    print()
    print("=" * 78)
    print("EXECUTION-SEMANTIC SEARCH BUDGET AUDIT")
    print("=" * 78)

    print(
        "questions:",
        len(questions)
    )

    print(
        "K:",
        args.candidates
    )

    for q_idx, question in enumerate(
        questions
    ):

        seed = (
            args.base_seed
            + q_idx
        )

        set_seed(
            seed
        )

        cond_np = np.asarray(
            get_sentence_embedding(
                question
            ),
            dtype=np.float32,
        )

        condition = (
            torch
            .from_numpy(
                cond_np
            )
            .unsqueeze(0)
            .to(device)
        )

        guider = (
            ExecutionSemanticAuditGuider(
                proxy_model=proxy,
                runtime_semantics=runtime,
                K=args.candidates,
                device=device,
            )
        )

        with torch.no_grad():

            framework.diffusion_model.sample(
                num_nodes=n,
                batch_size=1,

                node_features=(
                    node_features
                    .unsqueeze(0)
                ),

                task_condition=(
                    condition
                ),

                guider=guider,
            )

        for item in guider.records:

            item[
                "question_index"
            ] = q_idx

            item[
                "seed"
            ] = seed

        all_records.extend(
            guider.records
        )

        mean_raw = np.mean(
            [
                r[
                    "K_raw_unique"
                ]
                for r in guider.records
            ]
        )

        mean_mapped = np.mean(
            [
                r[
                    "K_mapped_unique"
                ]
                for r in guider.records
            ]
        )

        mean_exec = np.mean(
            [
                r[
                    "K_exec_unique"
                ]
                for r in guider.records
            ]
        )

        print(
            f"[{q_idx+1:02d}/"
            f"{len(questions):02d}] "
            f"raw="
            f"{mean_raw:.3f} | "
            f"mapped="
            f"{mean_mapped:.3f} | "
            f"exec="
            f"{mean_exec:.3f}"
        )

    # ========================================================
    # Aggregate
    # ========================================================

    def arr(name):

        return np.asarray(
            [
                r[name]
                for r
                in all_records
            ],
            dtype=float,
        )

    raw = arr(
        "K_raw_unique"
    )

    mapped = arr(
        "K_mapped_unique"
    )

    executed = arr(
        "K_exec_unique"
    )

    index_alias = arr(
        "index_alias_loss"
    )

    cycle_alias = arr(
        "cycle_alias_loss"
    )

    total_alias = arr(
        "total_alias_loss"
    )

    reward_spread = arr(
        "reward_spread"
    )

    same_as_first = np.asarray(
        [
            r[
                "selected_exec_equals_first"
            ]
            for r
            in all_records
        ],
        dtype=bool,
    )

    K = args.candidates

    summary = {

        "questions":
            len(
                questions
            ),

        "total_diffusion_steps":
            len(
                all_records
            ),

        "K_nominal":
            K,

        "mean_K_raw_unique":
            float(
                raw.mean()
            ),

        "mean_K_mapped_unique":
            float(
                mapped.mean()
            ),

        "mean_K_exec_unique":
            float(
                executed.mean()
            ),

        "raw_search_ratio":
            float(
                raw.mean()
                / K
            ),

        "mapped_search_ratio":
            float(
                mapped.mean()
                / K
            ),

        "effective_search_ratio":
            float(
                executed.mean()
                / K
            ),

        "mean_index_alias_loss":
            float(
                index_alias.mean()
            ),

        "mean_cycle_alias_loss":
            float(
                cycle_alias.mean()
            ),

        "mean_total_alias_loss":
            float(
                total_alias.mean()
            ),

        "fraction_steps_K_exec_eq_1":
            float(
                np.mean(
                    executed == 1
                )
            ),

        "fraction_steps_K_exec_lt_K":
            float(
                np.mean(
                    executed < K
                )
            ),

        "fraction_steps_exec_lt_mapped":
            float(
                np.mean(
                    executed
                    < mapped
                )
            ),

        "fraction_steps_mapped_lt_raw":
            float(
                np.mean(
                    mapped
                    < raw
                )
            ),

        "proxy_selected_same_executed_as_first_rate":
            float(
                same_as_first.mean()
            ),

        "mean_reward_spread":
            float(
                reward_spread.mean()
            ),

        "max_reward_spread":
            float(
                reward_spread.max()
            ),

        "runtime_validation":
            validation,
    }

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "summary":
                    summary,

                "steps":
                    all_records,
            },
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
