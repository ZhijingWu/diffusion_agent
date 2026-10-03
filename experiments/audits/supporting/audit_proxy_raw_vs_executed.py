import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse

from GDesigner.gdt.proxy_reward_model import ProxyRewardModel
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry


# ============================================================
# 1. Reproduce Graph's execution-time DAG projection
# ============================================================

def would_create_cycle(adj, src, dst):
    """
    Return True if adding src -> dst would create a cycle.

    This reproduces the semantics of Graph.check_cycle():

        if new_node in target_nodes:
            return True

    So self-loops are immediately rejected.

    For a normal edge src -> dst, adding it creates a cycle
    iff there is already a path:

        dst -> ... -> src
    """
    n = len(adj)

    if src == dst:
        return True

    stack = [dst]
    visited = set()

    while stack:
        node = stack.pop()

        if node == src:
            return True

        if node in visited:
            continue

        visited.add(node)

        for nxt in range(n):
            if adj[node][nxt] == 1:
                stack.append(nxt)

    return False


def project_to_executed_topology(raw_adj):
    """
    Reproduce the fixed-spatial-mask execution path used by Graph.

    Potential edges are processed in row-major order:
        src outer loop
        dst inner loop

    Returns:
        executed adjacency matrix
    """
    n = len(raw_adj)

    executed = [
        [0 for _ in range(n)]
        for _ in range(n)
    ]

    for src in range(n):
        for dst in range(n):

            if raw_adj[src][dst] == 0:
                continue

            if would_create_cycle(executed, src, dst):
                continue

            executed[src][dst] = 1

    return executed


def count_edges(adj):
    return sum(
        int(x)
        for row in adj
        for x in row
    )


# ============================================================
# 2. Utilities
# ============================================================

def find_latest_result():
    files = sorted(
        glob.glob("result/gtd_gsm8k/*.json"),
        key=os.path.getmtime
    )

    if not files:
        raise FileNotFoundError(
            "No result/gtd_gsm8k/*.json found."
        )

    return files[-1]


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return None

    if np.std(x) == 0 or np.std(y) == 0:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def mean_abs(values):
    if not values:
        return None

    return float(np.mean(np.abs(values)))


# ============================================================
# 3. Build Proxy input
# ============================================================

def adjacency_to_data(
    adjacency,
    node_features,
    task_condition,
):
    adj_tensor = torch.tensor(
        adjacency,
        dtype=torch.float32
    )

    edge_index, _ = dense_to_sparse(adj_tensor)

    return Data(
        x=node_features.clone(),
        edge_index=edge_index,
        condition=task_condition.clone(),
    )


# ============================================================
# 4. Main audit
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare ProxyRewardModel predictions on raw GTD "
            "topologies versus cycle-filtered executable topologies."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Phase-3 result JSON. "
            "Defaults to latest result/gtd_gsm8k/*.json"
        ),
    )

    parser.add_argument(
        "--proxy-model",
        type=str,
        default="proxy_model_gsm8k.pth",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="audit_results/proxy_raw_vs_executed",
    )

    parser.add_argument(
        "--agent-name",
        type=str,
        default="MathSolver",
    )

    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
    )

    # Match run_gsm8k.py defaults
    parser.add_argument(
        "--task-cond-input-dim",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--node-feature-dim",
        type=int,
        default=384,
    )

    parser.add_argument(
        "--condition-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--gnn-layers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=64,
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

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    input_path = (
        args.input
        if args.input is not None
        else find_latest_result()
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("PROXY RAW-vs-EXECUTED SEMANTIC AUDIT")
    print("=" * 72)

    print("Device:", device)
    print("Result:", input_path)
    print("Proxy:", args.proxy_model)

    # --------------------------------------------------------
    # Load experiment results
    # --------------------------------------------------------

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as f:
        results = json.load(f)

    if not isinstance(results, list):
        raise ValueError(
            "Expected Phase-3 result file to contain a list."
        )

    # --------------------------------------------------------
    # Rebuild the exact Proxy model architecture
    # --------------------------------------------------------

    proxy_model = ProxyRewardModel(
        task_cond_input_dim=args.task_cond_input_dim,
        node_feature_dim=args.node_feature_dim,
        condition_dim=args.condition_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        num_reward_components=2,
    ).to(device)

    state_dict = torch.load(
        args.proxy_model,
        map_location=device,
    )

    proxy_model.load_state_dict(state_dict)
    proxy_model.eval()

    # --------------------------------------------------------
    # Reconstruct node features exactly as run_gsm8k.py does
    # --------------------------------------------------------

    prompt_set = PromptSetRegistry.get("gsm8k")

    agent_names = [
        args.agent_name
        for _ in range(args.num_agents)
    ]

    agent_profiles = [
        prompt_set.get_description(name)
        for name in agent_names
    ]

    node_features = torch.tensor(
        [
            get_sentence_embedding(profile)
            for profile in agent_profiles
        ],
        dtype=torch.float32,
    ).to(device)

    # --------------------------------------------------------
    # Build raw/executed graph pairs
    # --------------------------------------------------------

    pyg_objects = []

    metadata = []

    for idx, record in enumerate(results):

        raw_adj = record.get(
            "Generated_Topology"
        )

        if raw_adj is None:
            print(
                f"[WARN] record {idx}: "
                "no Generated_Topology"
            )
            continue

        raw_adj = [
            [int(x) for x in row]
            for row in raw_adj
        ]

        executed_adj = (
            project_to_executed_topology(
                raw_adj
            )
        )

        question = record["Question"]

        condition_embedding = torch.tensor(
            get_sentence_embedding(question),
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        raw_data = adjacency_to_data(
            raw_adj,
            node_features,
            condition_embedding,
        )

        executed_data = adjacency_to_data(
            executed_adj,
            node_features,
            condition_embedding,
        )

        # Important ordering:
        # raw0, executed0, raw1, executed1, ...
        pyg_objects.append(raw_data)
        pyg_objects.append(executed_data)

        metadata.append(
            {
                "index": idx,
                "question": question,
                "solved": bool(
                    record.get(
                        "Solved",
                        False
                    )
                ),
                "raw_adj": raw_adj,
                "executed_adj": executed_adj,
            }
        )

    # --------------------------------------------------------
    # Batch Proxy inference
    # --------------------------------------------------------

    batch = Batch.from_data_list(
        pyg_objects
    ).to(device)

    with torch.no_grad():
        predictions = (
            proxy_model(batch)
            .detach()
            .cpu()
            .numpy()
        )

    # Prediction order:
    #
    # 0 = raw graph 0
    # 1 = executed graph 0
    # 2 = raw graph 1
    # 3 = executed graph 1
    # ...

    per_record = []

    delta_utility = []
    delta_cost = []
    delta_composite = []

    raw_predicted_utilities = []
    executed_predicted_utilities = []

    raw_predicted_costs = []
    executed_predicted_costs = []

    raw_edge_counts = []
    executed_edge_counts = []

    solved_labels = []

    raw_utility_errors = []
    executed_utility_errors = []

    raw_cost_errors = []
    executed_cost_errors = []

    proxy_prefers_raw = 0
    proxy_prefers_executed = 0
    proxy_ties = 0

    changed_graphs = 0

    for i, meta in enumerate(metadata):

        raw_pred = predictions[2 * i]
        executed_pred = predictions[
            2 * i + 1
        ]

        raw_u = float(raw_pred[0])
        raw_c = float(raw_pred[1])

        exe_u = float(executed_pred[0])
        exe_c = float(executed_pred[1])

        raw_score = (
            args.utility_weight * raw_u
            +
            args.cost_weight * raw_c
        )

        exe_score = (
            args.utility_weight * exe_u
            +
            args.cost_weight * exe_c
        )

        du = exe_u - raw_u
        dc = exe_c - raw_c
        ds = exe_score - raw_score

        raw_edges = count_edges(
            meta["raw_adj"]
        )

        executed_edges = count_edges(
            meta["executed_adj"]
        )

        solved = float(
            meta["solved"]
        )

        graph_changed = (
            meta["raw_adj"]
            != meta["executed_adj"]
        )

        if graph_changed:
            changed_graphs += 1

        if raw_score > exe_score + 1e-8:
            proxy_prefers_raw += 1

        elif exe_score > raw_score + 1e-8:
            proxy_prefers_executed += 1

        else:
            proxy_ties += 1

        delta_utility.append(du)
        delta_cost.append(dc)
        delta_composite.append(ds)

        raw_predicted_utilities.append(
            raw_u
        )

        executed_predicted_utilities.append(
            exe_u
        )

        raw_predicted_costs.append(
            raw_c
        )

        executed_predicted_costs.append(
            exe_c
        )

        raw_edge_counts.append(
            raw_edges
        )

        executed_edge_counts.append(
            executed_edges
        )

        solved_labels.append(
            solved
        )

        # Utility label:
        # actual MAS result corresponds to executed graph.
        raw_utility_errors.append(
            (raw_u - solved) ** 2
        )

        executed_utility_errors.append(
            (exe_u - solved) ** 2
        )

        # Phase-1 Proxy cost label was edge count.
        raw_cost_errors.append(
            abs(raw_c - raw_edges)
        )

        executed_cost_errors.append(
            abs(exe_c - executed_edges)
        )

        per_record.append(
            {
                "index": meta["index"],
                "question": meta["question"],
                "solved": bool(
                    meta["solved"]
                ),

                "topology_changed": (
                    graph_changed
                ),

                "raw_edge_count": (
                    raw_edges
                ),

                "executed_edge_count": (
                    executed_edges
                ),

                "proxy_raw": {
                    "utility": raw_u,
                    "cost": raw_c,
                    "composite": (
                        raw_score
                    ),
                },

                "proxy_executed": {
                    "utility": exe_u,
                    "cost": exe_c,
                    "composite": (
                        exe_score
                    ),
                },

                "delta_executed_minus_raw": {
                    "utility": du,
                    "cost": dc,
                    "composite": ds,
                },
            }
        )

    n = len(per_record)

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    raw_utility_mse = float(
        np.mean(
            raw_utility_errors
        )
    )

    executed_utility_mse = float(
        np.mean(
            executed_utility_errors
        )
    )

    raw_cost_mae = float(
        np.mean(
            raw_cost_errors
        )
    )

    executed_cost_mae = float(
        np.mean(
            executed_cost_errors
        )
    )

    summary = {
        "input_result": (
            input_path
        ),

        "proxy_model": (
            args.proxy_model
        ),

        "num_records": n,

        "graphs_changed": (
            changed_graphs
        ),

        "graph_change_rate": (
            changed_graphs / n
        ),

        "weights": {
            "utility": (
                args.utility_weight
            ),
            "cost": (
                args.cost_weight
            ),
        },

        "proxy_prediction_shift": {

            "mean_abs_utility_delta": (
                mean_abs(
                    delta_utility
                )
            ),

            "mean_abs_cost_delta": (
                mean_abs(
                    delta_cost
                )
            ),

            "mean_abs_composite_delta": (
                mean_abs(
                    delta_composite
                )
            ),

            "mean_utility_delta": float(
                np.mean(
                    delta_utility
                )
            ),

            "mean_cost_delta": float(
                np.mean(
                    delta_cost
                )
            ),

            "mean_composite_delta": float(
                np.mean(
                    delta_composite
                )
            ),
        },

        "proxy_pairwise_preference": {
            "raw_higher_score": (
                proxy_prefers_raw
            ),
            "executed_higher_score": (
                proxy_prefers_executed
            ),
            "tie": (
                proxy_ties
            ),
        },

        "cost_alignment": {

            "raw_predicted_cost_mae_vs_raw_edges": (
                raw_cost_mae
            ),

            "executed_predicted_cost_mae_vs_executed_edges": (
                executed_cost_mae
            ),

            "raw_cost_edge_pearson": (
                safe_pearson(
                    raw_predicted_costs,
                    raw_edge_counts,
                )
            ),

            "executed_cost_edge_pearson": (
                safe_pearson(
                    executed_predicted_costs,
                    executed_edge_counts,
                )
            ),
        },

        "utility_alignment_with_actual_MAS_outcome": {

            "raw_proxy_utility_mse": (
                raw_utility_mse
            ),

            "executed_proxy_utility_mse": (
                executed_utility_mse
            ),

            "raw_proxy_utility_pearson": (
                safe_pearson(
                    raw_predicted_utilities,
                    solved_labels,
                )
            ),

            "executed_proxy_utility_pearson": (
                safe_pearson(
                    executed_predicted_utilities,
                    solved_labels,
                )
            ),
        },
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary_path = (
        output_dir
        / "summary.json"
    )

    records_path = (
        output_dir
        / "per_record.jsonl"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        records_path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in per_record:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Records: {n}"
    )

    print(
        f"Topology changed: "
        f"{changed_graphs}/{n} "
        f"({changed_graphs / n:.1%})"
    )

    print()

    print(
        "=== PROXY PREDICTION SHIFT ==="
    )

    print(
        "Mean |Δ utility|:",
        f"{mean_abs(delta_utility):.4f}"
    )

    print(
        "Mean |Δ cost|:",
        f"{mean_abs(delta_cost):.4f}"
    )

    print(
        "Mean |Δ composite reward|:",
        f"{mean_abs(delta_composite):.4f}"
    )

    print()

    print(
        "=== PAIRWISE COMPOSITE SCORE ==="
    )

    print(
        "Proxy prefers RAW:",
        proxy_prefers_raw
    )

    print(
        "Proxy prefers EXECUTED:",
        proxy_prefers_executed
    )

    print(
        "Tie:",
        proxy_ties
    )

    print()

    print(
        "=== COST ALIGNMENT ==="
    )

    print(
        "Raw prediction MAE vs raw edge count:",
        f"{raw_cost_mae:.4f}"
    )

    print(
        "Executed prediction MAE vs executed edge count:",
        f"{executed_cost_mae:.4f}"
    )

    print(
        "Raw predicted-cost / raw-edge Pearson:",
        summary[
            "cost_alignment"
        ][
            "raw_cost_edge_pearson"
        ]
    )

    print(
        "Executed predicted-cost / executed-edge Pearson:",
        summary[
            "cost_alignment"
        ][
            "executed_cost_edge_pearson"
        ]
    )

    print()

    print(
        "=== UTILITY ALIGNMENT WITH ACTUAL MAS RESULT ==="
    )

    print(
        "Raw proxy utility MSE:",
        f"{raw_utility_mse:.4f}"
    )

    print(
        "Executed proxy utility MSE:",
        f"{executed_utility_mse:.4f}"
    )

    print(
        "Raw utility / solved Pearson:",
        summary[
            "utility_alignment_with_actual_MAS_outcome"
        ][
            "raw_proxy_utility_pearson"
        ]
    )

    print(
        "Executed utility / solved Pearson:",
        summary[
            "utility_alignment_with_actual_MAS_outcome"
        ][
            "executed_proxy_utility_pearson"
        ]
    )

    print()

    print("Saved:")
    print(" ", summary_path)
    print(" ", records_path)


if __name__ == "__main__":
    main()