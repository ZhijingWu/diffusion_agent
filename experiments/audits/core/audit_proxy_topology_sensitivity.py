import argparse
import json
import random

import numpy as np
import torch

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import global_mean_pool

from GDesigner.gdt.proxy_reward_model import ProxyRewardModel
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


# ============================================================
# Topology constructors
# ============================================================

def empty_graph(n):
    return [[0 for _ in range(n)] for _ in range(n)]


def full_graph(n):
    return [
        [0 if i == j else 1 for j in range(n)]
        for i in range(n)
    ]


def chain_graph(n):
    A = empty_graph(n)
    for i in range(n - 1):
        A[i][i + 1] = 1
    return A


def star_graph(n):
    A = empty_graph(n)
    for j in range(1, n):
        A[0][j] = 1
    return A


def reverse_star_graph(n):
    A = empty_graph(n)
    for i in range(1, n):
        A[i][0] = 1
    return A


def directed_cycle_graph(n):
    A = empty_graph(n)
    for i in range(n):
        A[i][(i + 1) % n] = 1
    return A


def random_graph(n, p, seed):
    rng = random.Random(seed)

    A = empty_graph(n)

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            if rng.random() < p:
                A[i][j] = 1

    return A


def count_edges(A):
    return sum(
        int(x)
        for row in A
        for x in row
    )


# ============================================================
# Dataset helper
# ============================================================

def load_questions(path, limit):
    questions = []

    with open(path, "r", encoding="utf-8") as f:
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

    if not questions:
        raise RuntimeError(
            f"No questions could be read from {path}"
        )

    return questions


# ============================================================
# PyG helper
# ============================================================

def make_data(
    adjacency,
    node_features,
    task_condition,
):
    A = torch.tensor(
        adjacency,
        dtype=torch.float32,
    )

    edge_index, _ = dense_to_sparse(A)

    return Data(
        x=node_features.clone(),
        edge_index=edge_index,
        condition=task_condition.clone(),
    )


# ============================================================
# Extract graph embedding BEFORE condition concatenation
# ============================================================

@torch.no_grad()
def get_graph_embeddings(
    proxy_model,
    pyg_batch,
):
    x = pyg_batch.x
    edge_index = pyg_batch.edge_index
    batch_vector = pyg_batch.batch

    h = x

    for layer in proxy_model.gnn_layers:
        h = layer(
            h,
            edge_index,
        )

        h = torch.relu(h)

    pooled = global_mean_pool(
        h,
        batch_vector,
    )

    return pooled


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
        "--dataset",
        default="datasets/gsm8k/gsm8k_test.jsonl",
    )

    parser.add_argument(
        "--questions",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--agent-name",
        default="MathSolver",
    )

    parser.add_argument(
        "--utility-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--cost-weight",
        type=float,
        default=-0.1,
    )

    parser.add_argument(
        "--output",
        default=(
            "audit_results/"
            "proxy_topology_sensitivity.json"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    n = args.num_agents

    # ========================================================
    # Load Proxy
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

    state = torch.load(
        args.proxy_model,
        map_location=device,
    )

    proxy.load_state_dict(state)
    proxy.eval()

    # ========================================================
    # Reconstruct homogeneous MathSolver node features
    # ========================================================

    prompt_set = PromptSetRegistry.get("gsm8k")

    profile = prompt_set.get_description(
        args.agent_name
    )

    embedding = np.asarray(
        get_sentence_embedding(profile),
        dtype=np.float32,
    )

    node_features_np = np.stack(
        [embedding.copy() for _ in range(n)],
        axis=0,
    )

    node_features = torch.from_numpy(
        node_features_np
    ).to(device)

    # Verify that all agent node features are identical.
    max_node_feature_diff = float(
        torch.max(
            torch.abs(
                node_features
                - node_features[0:1]
            )
        ).item()
    )

    # ========================================================
    # Define very different topology structures
    # ========================================================

    topologies = {
        "empty": empty_graph(n),

        "chain": chain_graph(n),

        "star_out": star_graph(n),

        "star_in": reverse_star_graph(n),

        "cycle": directed_cycle_graph(n),

        "random_sparse": random_graph(
            n,
            p=0.25,
            seed=1234,
        ),

        "random_dense": random_graph(
            n,
            p=0.75,
            seed=5678,
        ),

        "fully_connected": full_graph(n),
    }

    print("=" * 76)
    print("PROXY TOPOLOGY SENSITIVITY AUDIT")
    print("=" * 76)

    print("Device:", device)
    print(
        "Max difference between MathSolver node features:",
        f"{max_node_feature_diff:.12f}",
    )

    print()

    print("Topology edge counts:")

    for name, A in topologies.items():
        print(
            f"  {name:18s}: "
            f"{count_edges(A)} edges"
        )

    # ========================================================
    # Questions
    # ========================================================

    questions = load_questions(
        args.dataset,
        args.questions,
    )

    all_results = []

    utility_ranges = []
    cost_ranges = []
    reward_ranges = []
    embedding_ranges = []

    # ========================================================
    # Evaluate each question
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

        condition = torch.from_numpy(
            condition_np
        ).unsqueeze(0).to(device)

        data_list = []

        topology_names = list(
            topologies.keys()
        )

        for name in topology_names:

            data_list.append(
                make_data(
                    topologies[name],
                    node_features,
                    condition,
                )
            )

        batch = Batch.from_data_list(
            data_list
        ).to(device)

        with torch.no_grad():

            preds = (
                proxy(batch)
                .detach()
                .cpu()
                .numpy()
            )

            graph_embeddings = (
                get_graph_embeddings(
                    proxy,
                    batch,
                )
                .detach()
                .cpu()
                .numpy()
            )

        per_topology = []

        utilities = []
        costs = []
        rewards = []

        for i, name in enumerate(
            topology_names
        ):

            u = float(
                preds[i][0]
            )

            c = float(
                preds[i][1]
            )

            reward = (
                args.utility_weight * u
                +
                args.cost_weight * c
            )

            utilities.append(u)
            costs.append(c)
            rewards.append(reward)

            per_topology.append({
                "name": name,
                "edges": count_edges(
                    topologies[name]
                ),
                "utility": u,
                "cost": c,
                "reward": reward,
            })

        utility_range = (
            max(utilities)
            - min(utilities)
        )

        cost_range = (
            max(costs)
            - min(costs)
        )

        reward_range = (
            max(rewards)
            - min(rewards)
        )

        # Compare graph embeddings against
        # topology 0 as reference.
        reference = (
            graph_embeddings[0]
        )

        max_embedding_difference = 0.0

        for emb in graph_embeddings[1:]:

            diff = np.max(
                np.abs(
                    emb - reference
                )
            )

            max_embedding_difference = max(
                max_embedding_difference,
                float(diff),
            )

        utility_ranges.append(
            utility_range
        )

        cost_ranges.append(
            cost_range
        )

        reward_ranges.append(
            reward_range
        )

        embedding_ranges.append(
            max_embedding_difference
        )

        all_results.append({
            "question_index": q_idx,
            "question": question,

            "utility_range": (
                utility_range
            ),

            "cost_range": (
                cost_range
            ),

            "reward_range": (
                reward_range
            ),

            "max_graph_embedding_difference": (
                max_embedding_difference
            ),

            "topologies": (
                per_topology
            ),
        })

        # Print first 3 questions in detail
        if q_idx < 3:

            print()
            print(
                "=" * 76
            )

            print(
                f"QUESTION {q_idx + 1}"
            )

            print(
                question[:100]
            )

            print(
                "-" * 76
            )

            print(
                f"{'topology':18s} "
                f"{'edges':>5s} "
                f"{'utility':>12s} "
                f"{'cost':>12s} "
                f"{'reward':>12s}"
            )

            for row in per_topology:

                print(
                    f"{row['name']:18s} "
                    f"{row['edges']:5d} "
                    f"{row['utility']:12.8f} "
                    f"{row['cost']:12.8f} "
                    f"{row['reward']:12.8f}"
                )

            print()

            print(
                "utility range:",
                f"{utility_range:.12f}"
            )

            print(
                "cost range:",
                f"{cost_range:.12f}"
            )

            print(
                "reward range:",
                f"{reward_range:.12f}"
            )

            print(
                "max graph embedding diff:",
                f"{max_embedding_difference:.12f}"
            )

    # ========================================================
    # Aggregate summary
    # ========================================================

    summary = {

        "num_questions": (
            len(questions)
        ),

        "num_topologies": (
            len(topologies)
        ),

        "max_node_feature_difference": (
            max_node_feature_diff
        ),

        "mean_utility_range_across_topologies": float(
            np.mean(
                utility_ranges
            )
        ),

        "max_utility_range_across_topologies": float(
            np.max(
                utility_ranges
            )
        ),

        "mean_cost_range_across_topologies": float(
            np.mean(
                cost_ranges
            )
        ),

        "max_cost_range_across_topologies": float(
            np.max(
                cost_ranges
            )
        ),

        "mean_reward_range_across_topologies": float(
            np.mean(
                reward_ranges
            )
        ),

        "max_reward_range_across_topologies": float(
            np.max(
                reward_ranges
            )
        ),

        "mean_max_graph_embedding_difference": float(
            np.mean(
                embedding_ranges
            )
        ),

        "max_graph_embedding_difference_overall": float(
            np.max(
                embedding_ranges
            )
        ),
    }

    print()
    print("=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)

    for k, v in summary.items():

        if isinstance(v, float):

            print(
                f"{k}: {v:.12f}"
            )

        else:

            print(
                f"{k}: {v}"
            )

    output = {
        "summary": summary,
        "results": all_results,
    }

    import os

    output_dir = os.path.dirname(
        args.output
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True,
        )

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()