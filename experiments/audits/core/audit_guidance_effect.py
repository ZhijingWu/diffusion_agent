import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from GDesigner.gdt.proxy_reward_model import ProxyRewardModel
from GDesigner.gdt.gtd_framework import GTDFramework
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
# Graph utilities
# ============================================================

def count_edges(A):
    return int(A.sum().item())


def hamming_distance(A, B):
    return int((A != B).sum().item())


def normalized_hamming(A, B):
    return float((A != B).float().mean().item())


def topology_key(A):
    return tuple(
        tuple(int(x) for x in row)
        for row in A.tolist()
    )


# ============================================================
# Execution-time DAG projection
# Same semantics as Graph.check_cycle()
# ============================================================

def would_create_cycle(adj, src, dst):

    n = len(adj)

    if src == dst:
        return True

    stack = [dst]
    visited = set()

    while stack:

        u = stack.pop()

        if u == src:
            return True

        if u in visited:
            continue

        visited.add(u)

        for v in range(n):

            if adj[u][v] == 1:
                stack.append(v)

    return False


def project_to_executed(A):

    raw = A.tolist()

    n = len(raw)

    executed = [
        [0 for _ in range(n)]
        for _ in range(n)
    ]

    for src in range(n):

        for dst in range(n):

            if raw[src][dst] == 0:
                continue

            if would_create_cycle(
                executed,
                src,
                dst
            ):
                continue

            executed[src][dst] = 1

    return torch.tensor(
        executed,
        dtype=torch.int64
    )


# ============================================================
# Proxy-guided guider with instrumentation
# ============================================================

class InstrumentedProxyGuider:

    def __init__(
        self,
        proxy_model,
        macp_weights,
        num_candidates_per_step,
        device,
    ):

        self.proxy_model = proxy_model
        self.proxy_model.eval()

        self.utility_weight = (
            macp_weights["utility"]
        )

        self.cost_weight = (
            macp_weights["cost"]
        )

        self.K = (
            num_candidates_per_step
        )

        self.device = device

        self.history = []

    def guide(
        self,
        current_At_prob,
        timestep,
        unguided_A0_prediction,
        node_features,
        task_condition,
    ):

        batch_size, num_nodes, _ = (
            unguided_A0_prediction.shape
        )

        output = torch.zeros_like(
            unguided_A0_prediction
        )

        for b in range(batch_size):

            probs = (
                unguided_A0_prediction[b]
            )

            candidates = []

            # IMPORTANT:
            # exactly K Bernoulli draws,
            # matching original GuidedGeneration.
            for _ in range(self.K):

                candidate = torch.bernoulli(
                    probs
                ).float()

                candidates.append(
                    candidate
                )

            data_list = []

            condition = (
                task_condition[b]
                .unsqueeze(0)
            )

            for A in candidates:

                edge_index, _ = (
                    dense_to_sparse(A)
                )

                data_list.append(
                    Data(
                        x=node_features[b].clone(),
                        edge_index=edge_index,
                        condition=condition.clone(),
                    )
                )

            proxy_batch = (
                Batch
                .from_data_list(data_list)
                .to(self.device)
            )

            with torch.no_grad():

                rewards = self.proxy_model(
                    proxy_batch
                )

            utility = rewards[:, 0]
            cost = rewards[:, 1]

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

            record = {

                "timestep": int(
                    timestep[b].item()
                ),

                "selected_idx": (
                    best_idx
                ),

                "utility_min": float(
                    utility.min().item()
                ),

                "utility_max": float(
                    utility.max().item()
                ),

                "utility_spread": float(
                    (
                        utility.max()
                        - utility.min()
                    ).item()
                ),

                "cost_min": float(
                    cost.min().item()
                ),

                "cost_max": float(
                    cost.max().item()
                ),

                "cost_spread": float(
                    (
                        cost.max()
                        - cost.min()
                    ).item()
                ),

                "reward_min": float(
                    composite.min().item()
                ),

                "reward_max": float(
                    composite.max().item()
                ),

                "reward_spread": float(
                    (
                        composite.max()
                        - composite.min()
                    ).item()
                ),
            }

            self.history.append(
                record
            )

        return output


# ============================================================
# Control guider
#
# Generates the SAME NUMBER of Bernoulli candidates,
# but ignores Proxy and always selects candidate 0.
#
# This isolates:
#
#   "Proxy selection"
#
# from:
#
#   "best-of-K stochastic sampling itself"
# ============================================================

class FirstCandidateGuider:

    def __init__(
        self,
        num_candidates_per_step,
    ):

        self.K = (
            num_candidates_per_step
        )

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

            candidates = []

            # IMPORTANT:
            # same K Bernoulli calls
            # as Proxy guider.
            for _ in range(self.K):

                candidates.append(
                    torch.bernoulli(
                        probs
                    ).float()
                )

            # Ignore rewards.
            output[b] = candidates[0]

        return output


# ============================================================
# Load questions
# ============================================================

def load_questions(
    path,
    limit,
):

    questions = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            obj = json.loads(line)

            q = (
                obj.get("question")
                or obj.get("task")
                or obj.get("Question")
            )

            if q is None:
                continue

            questions.append(q)

            if len(questions) >= limit:
                break

    return questions


# ============================================================
# Generate one topology
# ============================================================

@torch.no_grad()
def generate_one(
    diffusion_model,
    num_nodes,
    node_features,
    task_condition,
    guider,
):

    probs = diffusion_model.sample(
        num_nodes=num_nodes,
        batch_size=1,
        node_features=(
            node_features.unsqueeze(0)
        ),
        task_condition=(
            task_condition
        ),
        guider=guider,
    )

    A = (
        probs
        .squeeze(0)
        .gt(0.5)
        .int()
        .cpu()
    )

    return A


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--proxy-model",
        default="proxy_model_gsm8k.pth",
    )

    parser.add_argument(
        "--diffusion-model",
        default="diffusion_model_gsm8k.pth",
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
        "--seeds",
        type=int,
        default=3,
        help=(
            "Number of random seeds "
            "per question."
        ),
    )

    parser.add_argument(
        "--candidates",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        default=(
            "audit_results/"
            "guidance_effect.json"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 76)
    print("GTD GUIDANCE EFFECT AUDIT")
    print("=" * 76)

    print("Device:", device)
    print(
        "Questions:",
        args.questions
    )
    print(
        "Seeds/question:",
        args.seeds
    )
    print(
        "Candidates:",
        args.candidates
    )

    # ========================================================
    # Proxy
    # ========================================================

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

    # ========================================================
    # GTD framework / diffusion model
    # ========================================================

    framework = GTDFramework(
        task_cond_input_dim=384,
        node_feature_dim=384,
        condition_dim=128,
        time_embed_dim=128,
        gt_num_layers=2,
        gt_num_heads=2,
        diffusion_num_timesteps=50,
        proxy_reward_model=proxy,
        macp_weights={
            "utility": 1.0,
            "cost": -0.1,
        },
        num_candidates_per_step=(
            args.candidates
        ),
        device=device,
    )

    framework.diffusion_model.load_state_dict(
        torch.load(
            args.diffusion_model,
            map_location=device,
        )
    )

    framework.diffusion_model.to(device)
    framework.diffusion_model.eval()

    diffusion_model = (
        framework.diffusion_model
    )

    # ========================================================
    # MathSolver × 4 features
    # ========================================================

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

    node_features = torch.from_numpy(
        np.stack(
            [
                profile_embedding.copy()
                for _ in range(4)
            ]
        )
    ).to(device)

    # ========================================================
    # Questions
    # ========================================================

    questions = load_questions(
        args.dataset,
        args.questions,
    )

    results = []

    reward_spreads = []
    utility_spreads = []
    cost_spreads = []

    proxy_vs_first_exact = 0
    proxy_vs_unguided_exact = 0
    first_vs_unguided_exact = 0

    proxy_vs_first_hamming = []
    proxy_vs_unguided_hamming = []
    first_vs_unguided_hamming = []

    proxy_raw_edges = []
    first_raw_edges = []
    unguided_raw_edges = []

    proxy_exec_edges = []
    first_exec_edges = []
    unguided_exec_edges = []

    proxy_topologies = set()
    first_topologies = set()
    unguided_topologies = set()

    total_runs = 0

    # ========================================================
    # Main experiment
    # ========================================================

    for q_idx, question in enumerate(
        questions
    ):

        condition_np = np.asarray(
            get_sentence_embedding(
                question
            ),
            dtype=np.float32,
        )

        condition = (
            torch
            .from_numpy(condition_np)
            .unsqueeze(0)
            .to(device)
        )

        print()
        print(
            f"[Question "
            f"{q_idx + 1}/"
            f"{len(questions)}]"
        )

        for seed_idx in range(
            args.seeds
        ):

            seed = (
                q_idx * 1000
                + seed_idx
                + 12345
            )

            # ------------------------------------------------
            # A. Proxy-guided
            # ------------------------------------------------

            set_seed(seed)

            proxy_guider = (
                InstrumentedProxyGuider(
                    proxy_model=proxy,
                    macp_weights={
                        "utility": 1.0,
                        "cost": -0.1,
                    },
                    num_candidates_per_step=(
                        args.candidates
                    ),
                    device=device,
                )
            )

            A_proxy = generate_one(
                diffusion_model,
                4,
                node_features,
                condition,
                proxy_guider,
            )

            # ------------------------------------------------
            # B. First-candidate control
            # ------------------------------------------------

            set_seed(seed)

            first_guider = (
                FirstCandidateGuider(
                    args.candidates
                )
            )

            A_first = generate_one(
                diffusion_model,
                4,
                node_features,
                condition,
                first_guider,
            )

            # ------------------------------------------------
            # C. Unguided
            # ------------------------------------------------

            set_seed(seed)

            A_unguided = generate_one(
                diffusion_model,
                4,
                node_features,
                condition,
                guider=None,
            )

            # ------------------------------------------------
            # Executed topology projection
            # ------------------------------------------------

            E_proxy = (
                project_to_executed(
                    A_proxy
                )
            )

            E_first = (
                project_to_executed(
                    A_first
                )
            )

            E_unguided = (
                project_to_executed(
                    A_unguided
                )
            )

            # ------------------------------------------------
            # Comparisons
            # ------------------------------------------------

            p_f_exact = torch.equal(
                A_proxy,
                A_first,
            )

            p_u_exact = torch.equal(
                A_proxy,
                A_unguided,
            )

            f_u_exact = torch.equal(
                A_first,
                A_unguided,
            )

            proxy_vs_first_exact += int(
                p_f_exact
            )

            proxy_vs_unguided_exact += int(
                p_u_exact
            )

            first_vs_unguided_exact += int(
                f_u_exact
            )

            p_f_ham = normalized_hamming(
                A_proxy,
                A_first,
            )

            p_u_ham = normalized_hamming(
                A_proxy,
                A_unguided,
            )

            f_u_ham = normalized_hamming(
                A_first,
                A_unguided,
            )

            proxy_vs_first_hamming.append(
                p_f_ham
            )

            proxy_vs_unguided_hamming.append(
                p_u_ham
            )

            first_vs_unguided_hamming.append(
                f_u_ham
            )

            # ------------------------------------------------
            # Edge counts
            # ------------------------------------------------

            proxy_raw_edges.append(
                count_edges(A_proxy)
            )

            first_raw_edges.append(
                count_edges(A_first)
            )

            unguided_raw_edges.append(
                count_edges(A_unguided)
            )

            proxy_exec_edges.append(
                count_edges(E_proxy)
            )

            first_exec_edges.append(
                count_edges(E_first)
            )

            unguided_exec_edges.append(
                count_edges(E_unguided)
            )

            proxy_topologies.add(
                topology_key(
                    A_proxy
                )
            )

            first_topologies.add(
                topology_key(
                    A_first
                )
            )

            unguided_topologies.add(
                topology_key(
                    A_unguided
                )
            )

            # ------------------------------------------------
            # Proxy candidate score spread
            # ------------------------------------------------

            local_reward_spreads = [
                x["reward_spread"]
                for x in proxy_guider.history
            ]

            local_utility_spreads = [
                x["utility_spread"]
                for x in proxy_guider.history
            ]

            local_cost_spreads = [
                x["cost_spread"]
                for x in proxy_guider.history
            ]

            reward_spreads.extend(
                local_reward_spreads
            )

            utility_spreads.extend(
                local_utility_spreads
            )

            cost_spreads.extend(
                local_cost_spreads
            )

            result = {

                "question_index": (
                    q_idx
                ),

                "seed": seed,

                "proxy_raw_edges": (
                    count_edges(
                        A_proxy
                    )
                ),

                "first_raw_edges": (
                    count_edges(
                        A_first
                    )
                ),

                "unguided_raw_edges": (
                    count_edges(
                        A_unguided
                    )
                ),

                "proxy_executed_edges": (
                    count_edges(
                        E_proxy
                    )
                ),

                "first_executed_edges": (
                    count_edges(
                        E_first
                    )
                ),

                "unguided_executed_edges": (
                    count_edges(
                        E_unguided
                    )
                ),

                "proxy_vs_first_exact": (
                    p_f_exact
                ),

                "proxy_vs_unguided_exact": (
                    p_u_exact
                ),

                "proxy_vs_first_hamming": (
                    p_f_ham
                ),

                "proxy_vs_unguided_hamming": (
                    p_u_ham
                ),

                "first_vs_unguided_hamming": (
                    f_u_ham
                ),

                "mean_candidate_reward_spread": (
                    float(
                        np.mean(
                            local_reward_spreads
                        )
                    )
                ),

                "max_candidate_reward_spread": (
                    float(
                        np.max(
                            local_reward_spreads
                        )
                    )
                ),
            }

            results.append(
                result
            )

            total_runs += 1

            print(
                f" seed={seed} | "
                f"P/F Hamming="
                f"{p_f_ham:.3f} | "
                f"P/U Hamming="
                f"{p_u_ham:.3f} | "
                f"reward spread="
                f"{np.mean(local_reward_spreads):.3e}"
            )

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "questions": (
            len(questions)
        ),

        "seeds_per_question": (
            args.seeds
        ),

        "total_runs": (
            total_runs
        ),

        "proxy_vs_first": {

            "exact_match_rate": (
                proxy_vs_first_exact
                / total_runs
            ),

            "mean_normalized_hamming": float(
                np.mean(
                    proxy_vs_first_hamming
                )
            ),
        },

        "proxy_vs_unguided": {

            "exact_match_rate": (
                proxy_vs_unguided_exact
                / total_runs
            ),

            "mean_normalized_hamming": float(
                np.mean(
                    proxy_vs_unguided_hamming
                )
            ),
        },

        "first_vs_unguided": {

            "exact_match_rate": (
                first_vs_unguided_exact
                / total_runs
            ),

            "mean_normalized_hamming": float(
                np.mean(
                    first_vs_unguided_hamming
                )
            ),
        },

        "candidate_proxy_discrimination": {

            "mean_reward_spread": float(
                np.mean(
                    reward_spreads
                )
            ),

            "max_reward_spread": float(
                np.max(
                    reward_spreads
                )
            ),

            "mean_utility_spread": float(
                np.mean(
                    utility_spreads
                )
            ),

            "mean_cost_spread": float(
                np.mean(
                    cost_spreads
                )
            ),
        },

        "average_raw_edges": {

            "proxy_guided": float(
                np.mean(
                    proxy_raw_edges
                )
            ),

            "first_candidate": float(
                np.mean(
                    first_raw_edges
                )
            ),

            "unguided": float(
                np.mean(
                    unguided_raw_edges
                )
            ),
        },

        "average_executed_edges": {

            "proxy_guided": float(
                np.mean(
                    proxy_exec_edges
                )
            ),

            "first_candidate": float(
                np.mean(
                    first_exec_edges
                )
            ),

            "unguided": float(
                np.mean(
                    unguided_exec_edges
                )
            ),
        },

        "unique_raw_topologies": {

            "proxy_guided": (
                len(
                    proxy_topologies
                )
            ),

            "first_candidate": (
                len(
                    first_topologies
                )
            ),

            "unguided": (
                len(
                    unguided_topologies
                )
            ),
        },
    }

    print()
    print("=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # ========================================================
    # Save
    # ========================================================

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
                "summary": summary,
                "runs": results,
            },
            f,
            indent=2,
        )

    print()
    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()