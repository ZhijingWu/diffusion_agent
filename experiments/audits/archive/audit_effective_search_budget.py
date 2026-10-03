import argparse
import contextlib
import io
import json
import random
from pathlib import Path

import numpy as np
import torch

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from GDesigner.graph.graph import Graph
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
# Execution projector
#
# Important:
# We do NOT assume raw adjacency row/column ordering equals
# runtime node ordering.
#
# We instantiate one real GDesigner Graph and read its actual
# potential_spatial_edges ordering. Candidate masks are then
# interpreted exactly in that ordering.
# ============================================================

class ExecutionProjector:

    def __init__(
        self,
        domain,
        llm_name,
        agent_names,
        decision_method,
    ):
        self.domain = domain
        self.llm_name = llm_name
        self.agent_names = agent_names
        self.decision_method = decision_method

        n = len(agent_names)

        zeros = [
            [0 for _ in range(n)]
            for _ in range(n)
        ]

        # Suppress the many debug prints in Graph.__init__.
        with contextlib.redirect_stdout(io.StringIO()):

            template = Graph(
                domain=domain,
                llm_name=llm_name,
                agent_names=agent_names,
                decision_method=decision_method,
                fixed_spatial_masks=zeros,
            )

        self.node_ids = list(
            template.nodes.keys()
        )

        self.node_to_idx = {
            node_id: i
            for i, node_id
            in enumerate(self.node_ids)
        }

        self.potential_edges = list(
            template.potential_spatial_edges
        )

        self.n = len(
            self.node_ids
        )

        expected = self.n * self.n

        if len(self.potential_edges) != expected:

            raise RuntimeError(
                "Unexpected number of potential spatial edges: "
                f"{len(self.potential_edges)} vs expected {expected}. "
                "Inspect Graph.init_potential_edges() before continuing."
            )

        # Cache because many candidates repeat.
        self.cache = {}

        print(
            "ExecutionProjector initialized:"
        )

        print(
            "  nodes:",
            self.n,
        )

        print(
            "  potential edges:",
            len(self.potential_edges),
        )

    # --------------------------------------------------------
    # Same graph-theoretic meaning as Graph.check_cycle:
    #
    # adding src -> dst creates a cycle iff dst can already
    # reach src.
    # --------------------------------------------------------

    @staticmethod
    def _has_path(
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

    def project(self, raw_A):

        key = topo_key(raw_A)

        if key in self.cache:
            return self.cache[key].clone()

        flat = [
            int(x)
            for row in key
            for x in row
        ]

        executed = [
            [0 for _ in range(self.n)]
            for _ in range(self.n)
        ]

        for mask_value, potential_edge in zip(
            flat,
            self.potential_edges,
        ):

            if mask_value == 0:
                continue

            # Source code semantics:
            #
            # out_node = potential_connection[0]
            # in_node  = potential_connection[1]
            # out_node.add_successor(in_node)
            #
            out_id = potential_edge[0]
            in_id = potential_edge[1]

            src = self.node_to_idx[out_id]
            dst = self.node_to_idx[in_id]

            # Self loops are cycles.
            if self._has_path(
                executed,
                dst,
                src,
            ):
                continue

            executed[src][dst] = 1

        result = torch.tensor(
            executed,
            dtype=torch.int64,
        )

        self.cache[key] = result

        return result.clone()


# ============================================================
# Validate projector against ACTUAL saved MAS execution.
# ============================================================

def validate_projector(
    projector,
    result_path,
):

    if result_path is None:
        return None

    print()
    print("=" * 78)
    print("PROJECTOR VALIDATION")
    print("=" * 78)

    data = json.load(
        open(
            result_path,
            encoding="utf-8",
        )
    )

    exact = 0

    hamming = []

    for item in data:

        raw = item[
            "Generated_Topology"
        ]

        expected = torch.tensor(
            item["Executed_Topology"],
            dtype=torch.int64,
        )

        predicted = projector.project(
            raw
        )

        same = torch.equal(
            predicted,
            expected,
        )

        exact += int(same)

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

    match_rate = (
        exact
        / len(data)
    )

    mean_hamming = float(
        np.mean(hamming)
    )

    print(
        "records:",
        len(data),
    )

    print(
        "exact projector matches:",
        f"{exact}/{len(data)} "
        f"({match_rate:.1%})",
    )

    print(
        "mean Hamming vs actual:",
        mean_hamming,
    )

    return {
        "records":
            len(data),

        "exact_matches":
            exact,

        "match_rate":
            match_rate,

        "mean_hamming":
            mean_hamming,
    }


# ============================================================
# Instrumented official-style Proxy guider
# ============================================================

class SearchBudgetGuider:

    def __init__(
        self,
        proxy_reward_model,
        projector,
        num_candidates_per_step,
        device,
        utility_weight=1.0,
        cost_weight=-0.1,
    ):

        self.proxy = (
            proxy_reward_model
        )

        self.proxy.eval()

        self.projector = (
            projector
        )

        self.K = (
            num_candidates_per_step
        )

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

        batch_size, _, _ = (
            unguided_A0_prediction
            .shape
        )

        output = torch.zeros_like(
            unguided_A0_prediction
        )

        for b in range(batch_size):

            probs = (
                unguided_A0_prediction[b]
            )

            # ---------------------------------------------
            # Same K Bernoulli candidate generation
            # ---------------------------------------------

            candidates = [

                torch.bernoulli(
                    probs
                ).float()

                for _ in range(
                    self.K
                )
            ]

            # ---------------------------------------------
            # Raw candidate diversity
            # ---------------------------------------------

            raw_keys = [
                topo_key(A)
                for A in candidates
            ]

            K_raw_unique = len(
                set(raw_keys)
            )

            # ---------------------------------------------
            # Runtime-executed candidate diversity
            # ---------------------------------------------

            executed = [

                self.projector.project(
                    A
                )

                for A in candidates
            ]

            exec_keys = [
                topo_key(A)
                for A in executed
            ]

            K_exec_unique = len(
                set(exec_keys)
            )

            # ---------------------------------------------
            # Proxy scoring: same semantics as official
            # GuidedGeneration
            # ---------------------------------------------

            pyg_list = []

            current_x = (
                node_features[b]
            )

            current_condition = (
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
                        x=current_x
                        .clone()
                        .to(self.device),

                        edge_index=(
                            edge_index
                        ),

                        condition=(
                            current_condition
                            .clone()
                            .to(self.device)
                        ),
                    )
                )

            proxy_batch = (
                Batch
                .from_data_list(
                    pyg_list
                )
                .to(self.device)
            )

            pred = self.proxy(
                proxy_batch
            )

            utility = pred[:, 0]
            cost = pred[:, 1]

            composite = (
                self.utility_weight
                * utility

                +

                self.cost_weight
                * cost
            )

            best_idx = int(
                torch.argmax(
                    composite
                ).item()
            )

            output[b] = candidates[
                best_idx
            ]

            # ---------------------------------------------
            # Does Proxy choose behaviorally the same graph
            # as arbitrary first candidate?
            # ---------------------------------------------

            selected_exec_key = (
                exec_keys[
                    best_idx
                ]
            )

            first_exec_key = (
                exec_keys[0]
            )

            selected_exec_equals_first = (
                selected_exec_key
                == first_exec_key
            )

            # ---------------------------------------------
            # Aliasing metrics
            # ---------------------------------------------

            raw_search_ratio = (
                K_raw_unique
                / self.K
            )

            effective_search_ratio = (
                K_exec_unique
                / self.K
            )

            if K_raw_unique > 0:

                aliasing_rate = (
                    1.0
                    -
                    K_exec_unique
                    / K_raw_unique
                )

            else:

                aliasing_rate = 0.0

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

                "K_exec_unique":
                    K_exec_unique,

                "raw_search_ratio":
                    raw_search_ratio,

                "effective_search_ratio":
                    effective_search_ratio,

                "aliasing_rate":
                    aliasing_rate,

                "reward_spread":
                    float(
                        (
                            composite.max()
                            - composite.min()
                        )
                        .item()
                    ),

                "selected_idx":
                    best_idx,

                "selected_exec_equals_first":
                    bool(
                        selected_exec_equals_first
                    ),

                "raw_candidates": [
                    [
                        list(row)
                        for row in key
                    ]
                    for key
                    in raw_keys
                ],

                "executed_candidates": [
                    A.int().tolist()
                    for A
                    in executed
                ],
            })

        return output


# ============================================================
# Load questions
# ============================================================

def load_questions(
    path,
    limit,
):

    result = []

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            item = json.loads(
                line
            )

            q = (
                item.get(
                    "question"
                )
                or item.get(
                    "Question"
                )
                or item.get(
                    "task"
                )
            )

            if q is None:
                continue

            result.append(q)

            if len(result) >= limit:
                break

    return result


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
        help=(
            "Optional real downstream result JSON "
            "containing Generated_Topology and "
            "Executed_Topology."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "audit_results/"
            "effective_search_budget/"
            "summary.json"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # Agent team
    # --------------------------------------------------------

    agent_names = [
        "MathSolver"
        for _ in range(4)
    ]

    # --------------------------------------------------------
    # Exact runtime-edge-order projector
    # --------------------------------------------------------

    projector = ExecutionProjector(
        domain="gsm8k",
        llm_name="gpt-4o-mini",
        agent_names=agent_names,
        decision_method="FinalRefer",
    )

    validation = validate_projector(
        projector,
        args.validate_result,
    )

    # If validation is provided and poor, stop.
    if (
        validation is not None
        and validation["match_rate"] < 0.95
    ):

        print()
        print(
            "STOP: projector does not match "
            "actual execution closely enough."
        )

        print(
            "Do NOT interpret execution-aliasing "
            "metrics yet."
        )

        return

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
    # Node features
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

    profile_embedding = np.asarray(
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
                    profile_embedding.copy()
                    for _ in range(4)
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

    all_steps = []

    print()
    print("=" * 78)
    print("EFFECTIVE SEARCH BUDGET AUDIT")
    print("=" * 78)

    print(
        "Questions:",
        len(questions),
    )

    print(
        "K:",
        args.candidates,
    )

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    for q_idx, question in enumerate(
        questions
    ):

        seed = (
            args.base_seed
            + q_idx
        )

        set_seed(seed)

        condition_np = np.asarray(
            get_sentence_embedding(
                question
            ),
            dtype=np.float32,
        )

        condition = (
            torch
            .from_numpy(
                condition_np
            )
            .unsqueeze(0)
            .to(device)
        )

        guider = SearchBudgetGuider(
            proxy_reward_model=proxy,
            projector=projector,
            num_candidates_per_step=(
                args.candidates
            ),
            device=device,
        )

        with torch.no_grad():

            framework.diffusion_model.sample(
                num_nodes=4,
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

        for step in guider.records:

            step[
                "question_index"
            ] = q_idx

            step[
                "seed"
            ] = seed

        all_steps.extend(
            guider.records
        )

        mean_exec = np.mean(
            [
                x["K_exec_unique"]
                for x in guider.records
            ]
        )

        mean_alias = np.mean(
            [
                x["aliasing_rate"]
                for x in guider.records
            ]
        )

        print(
            f"[{q_idx + 1:02d}/"
            f"{len(questions):02d}] "
            f"mean K_exec="
            f"{mean_exec:.3f} | "
            f"aliasing="
            f"{mean_alias:.3f}"
        )

    # ========================================================
    # Aggregate
    # ========================================================

    raw_unique = np.asarray(
        [
            x["K_raw_unique"]
            for x in all_steps
        ],
        dtype=float,
    )

    exec_unique = np.asarray(
        [
            x["K_exec_unique"]
            for x in all_steps
        ],
        dtype=float,
    )

    aliasing = np.asarray(
        [
            x["aliasing_rate"]
            for x in all_steps
        ],
        dtype=float,
    )

    reward_spread = np.asarray(
        [
            x["reward_spread"]
            for x in all_steps
        ],
        dtype=float,
    )

    same_as_first = np.asarray(
        [
            x[
                "selected_exec_equals_first"
            ]
            for x in all_steps
        ],
        dtype=bool,
    )

    K = args.candidates

    summary = {
        "questions":
            len(questions),

        "diffusion_steps_total":
            len(all_steps),

        "K_nominal":
            K,

        "mean_K_raw_unique":
            float(
                raw_unique.mean()
            ),

        "mean_K_exec_unique":
            float(
                exec_unique.mean()
            ),

        "effective_search_ratio":
            float(
                exec_unique.mean()
                / K
            ),

        "mean_aliasing_rate":
            float(
                aliasing.mean()
            ),

        "fraction_steps_exec_K_equals_1":
            float(
                np.mean(
                    exec_unique == 1
                )
            ),

        "fraction_steps_exec_K_less_than_K":
            float(
                np.mean(
                    exec_unique < K
                )
            ),

        "fraction_steps_exec_less_than_raw":
            float(
                np.mean(
                    exec_unique
                    < raw_unique
                )
            ),

        "proxy_selected_same_executed_graph_as_first_rate":
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

        "projector_validation":
            validation,
    }

    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    for key, value in summary.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key}: "
                f"{value}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "summary":
            summary,

        "steps":
            all_steps,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
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
