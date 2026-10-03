import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path


def would_create_cycle(adj, src, dst):
    """
    Mimic Graph.check_cycle(in_node, {out_node}) for a proposed edge src -> dst.

    The original code rejects an edge if there is already a path:
        dst -> ... -> src

    Self-loop src == dst is therefore immediately rejected.
    """
    n = len(adj)

    # Self-loop: original check_cycle(new_node, {out_node})
    # immediately returns True when new_node == out_node.
    if src == dst:
        return True

    # DFS from dst; if src is reachable, adding src -> dst creates a cycle.
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


def simulate_executed_topology(raw_adj):
    """
    Reproduce the fixed-mask path of Graph.construct_spatial_connection().

    Important:
    - potential_spatial_edges are enumerated in row-major order:
          for node1:
              for node2:
    - for each raw edge == 1:
        reject it if check_cycle says it would create a cycle;
        otherwise add it.

    Returns:
        executed_adj
        removed_self_loops
        removed_cycle_edges
        removed_edges list
    """
    n = len(raw_adj)

    executed_adj = [[0 for _ in range(n)] for _ in range(n)]

    removed_self_loops = 0
    removed_cycle_edges = 0
    removed_edges = []

    # Exact same ordering as init_potential_edges():
    # src outer loop, dst inner loop.
    for src in range(n):
        for dst in range(n):
            if raw_adj[src][dst] == 0:
                continue

            if src == dst:
                removed_self_loops += 1
                removed_edges.append(
                    {
                        "src": src,
                        "dst": dst,
                        "reason": "self_loop",
                    }
                )
                continue

            if would_create_cycle(executed_adj, src, dst):
                removed_cycle_edges += 1
                removed_edges.append(
                    {
                        "src": src,
                        "dst": dst,
                        "reason": "cycle",
                    }
                )
                continue

            executed_adj[src][dst] = 1

    return (
        executed_adj,
        removed_self_loops,
        removed_cycle_edges,
        removed_edges,
    )


def count_edges(adj):
    return sum(sum(int(x) for x in row) for row in adj)


def topology_key(adj):
    return tuple(tuple(int(x) for x in row) for row in adj)


def find_latest_result():
    files = sorted(
        glob.glob("result/gtd_gsm8k/*.json"),
        key=os.path.getmtime,
    )

    if not files:
        raise FileNotFoundError(
            "No result/gtd_gsm8k/*.json files found."
        )

    return files[-1]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit mismatch between GTD Generated_Topology "
            "and the topology actually executable after Graph cycle filtering."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Phase-3 result JSON. "
            "If omitted, use latest result/gtd_gsm8k/*.json."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="audit_results/raw_vs_executed",
        help="Directory for audit outputs.",
    )

    args = parser.parse_args()

    input_path = args.input or find_latest_result()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Expected result JSON to be a list, got {type(data)}"
        )

    per_record = []

    total_raw_edges = 0
    total_executed_edges = 0
    total_removed_self = 0
    total_removed_cycle = 0

    changed_graphs = 0
    self_loop_graphs = 0
    cycle_filtered_graphs = 0

    solved_total = 0
    solved_changed = 0
    changed_total = 0

    raw_topologies = Counter()
    executed_topologies = Counter()

    removed_count_distribution = Counter()
    raw_edge_distribution = Counter()
    executed_edge_distribution = Counter()

    for idx, item in enumerate(data):
        raw_adj = item.get("Generated_Topology")

        if raw_adj is None:
            print(f"[WARN] Record {idx} has no Generated_Topology; skipping.")
            continue

        raw_adj = [
            [int(x) for x in row]
            for row in raw_adj
        ]

        (
            executed_adj,
            removed_self,
            removed_cycle,
            removed_edges,
        ) = simulate_executed_topology(raw_adj)

        raw_edges = count_edges(raw_adj)
        executed_edges = count_edges(executed_adj)
        removed_total = raw_edges - executed_edges

        graph_changed = raw_adj != executed_adj
        solved = bool(item.get("Solved", False))

        total_raw_edges += raw_edges
        total_executed_edges += executed_edges
        total_removed_self += removed_self
        total_removed_cycle += removed_cycle

        solved_total += int(solved)

        if graph_changed:
            changed_graphs += 1
            changed_total += 1
            solved_changed += int(solved)

        if removed_self > 0:
            self_loop_graphs += 1

        if removed_cycle > 0:
            cycle_filtered_graphs += 1

        raw_topologies[topology_key(raw_adj)] += 1
        executed_topologies[topology_key(executed_adj)] += 1

        removed_count_distribution[removed_total] += 1
        raw_edge_distribution[raw_edges] += 1
        executed_edge_distribution[executed_edges] += 1

        per_record.append(
            {
                "index": idx,
                "question": item.get("Question"),
                "solved": solved,

                "raw_topology": raw_adj,
                "executed_topology_simulated": executed_adj,

                "raw_edge_count": raw_edges,
                "executed_edge_count": executed_edges,

                "removed_edge_count": removed_total,
                "removed_self_loop_count": removed_self,
                "removed_cycle_edge_count": removed_cycle,

                "removed_edges": removed_edges,

                "topology_changed": graph_changed,
            }
        )

    n = len(per_record)

    if n == 0:
        raise RuntimeError("No valid records were audited.")

    summary = {
        "input_file": input_path,
        "num_records": n,

        "accuracy_from_original_run": solved_total / n,

        "raw": {
            "total_edges": total_raw_edges,
            "avg_edges_per_graph": total_raw_edges / n,
            "unique_topologies": len(raw_topologies),
            "edge_distribution": dict(
                sorted(raw_edge_distribution.items())
            ),
        },

        "executed_simulated": {
            "total_edges": total_executed_edges,
            "avg_edges_per_graph": total_executed_edges / n,
            "unique_topologies": len(executed_topologies),
            "edge_distribution": dict(
                sorted(executed_edge_distribution.items())
            ),
        },

        "mismatch": {
            "graphs_changed": changed_graphs,
            "graph_change_rate": changed_graphs / n,

            "graphs_with_self_loops_removed": self_loop_graphs,
            "self_loop_graph_rate": self_loop_graphs / n,

            "graphs_with_cycle_edges_removed": cycle_filtered_graphs,
            "cycle_filtered_graph_rate": cycle_filtered_graphs / n,

            "total_edges_removed": (
                total_raw_edges - total_executed_edges
            ),

            "avg_edges_removed_per_graph": (
                total_raw_edges - total_executed_edges
            ) / n,

            "total_self_loops_removed": total_removed_self,
            "avg_self_loops_removed_per_graph": total_removed_self / n,

            "total_cycle_edges_removed": total_removed_cycle,
            "avg_cycle_edges_removed_per_graph": total_removed_cycle / n,

            "removed_edge_count_distribution": dict(
                sorted(removed_count_distribution.items())
            ),
        },

        "accuracy_on_changed_graph_subset": (
            solved_changed / changed_total
            if changed_total > 0
            else None
        ),
    }

    summary_path = output_dir / "summary.json"
    records_path = output_dir / "per_record.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(records_path, "w", encoding="utf-8") as f:
        for record in per_record:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    print("=" * 72)
    print("RAW vs EXECUTED TOPOLOGY AUDIT")
    print("=" * 72)

    print(f"Input: {input_path}")
    print(f"Records: {n}")
    print()

    print("=== RAW GENERATED TOPOLOGY ===")
    print(
        f"Average raw edges: "
        f"{summary['raw']['avg_edges_per_graph']:.3f}"
    )
    print(
        f"Unique raw topologies: "
        f"{summary['raw']['unique_topologies']}"
    )
    print(
        f"Raw edge distribution: "
        f"{summary['raw']['edge_distribution']}"
    )

    print()
    print("=== SIMULATED EXECUTED TOPOLOGY ===")
    print(
        f"Average executed edges: "
        f"{summary['executed_simulated']['avg_edges_per_graph']:.3f}"
    )
    print(
        f"Unique executed topologies: "
        f"{summary['executed_simulated']['unique_topologies']}"
    )
    print(
        f"Executed edge distribution: "
        f"{summary['executed_simulated']['edge_distribution']}"
    )

    print()
    print("=== MISMATCH ===")
    print(
        f"Graphs changed: "
        f"{changed_graphs}/{n} "
        f"({changed_graphs / n:.1%})"
    )
    print(
        f"Graphs with self-loop removal: "
        f"{self_loop_graphs}/{n} "
        f"({self_loop_graphs / n:.1%})"
    )
    print(
        f"Graphs with additional cycle-edge removal: "
        f"{cycle_filtered_graphs}/{n} "
        f"({cycle_filtered_graphs / n:.1%})"
    )

    print(
        f"Total edges removed: "
        f"{total_raw_edges - total_executed_edges}"
    )
    print(
        f"Average removed edges / graph: "
        f"{(total_raw_edges - total_executed_edges) / n:.3f}"
    )

    print(
        f"Total self-loops removed: "
        f"{total_removed_self}"
    )
    print(
        f"Average self-loops removed / graph: "
        f"{total_removed_self / n:.3f}"
    )

    print(
        f"Total non-self cycle edges removed: "
        f"{total_removed_cycle}"
    )
    print(
        f"Average non-self cycle edges removed / graph: "
        f"{total_removed_cycle / n:.3f}"
    )

    print(
        "Removed-edge-count distribution:",
        summary["mismatch"]["removed_edge_count_distribution"],
    )

    print()
    print("=== ORIGINAL MAS OUTCOME ===")
    print(
        f"Overall accuracy: "
        f"{summary['accuracy_from_original_run']:.3f}"
    )

    if changed_total > 0:
        print(
            f"Accuracy among graphs whose topology changed: "
            f"{summary['accuracy_on_changed_graph_subset']:.3f}"
        )

    print()
    print("Saved:")
    print(f"  {summary_path}")
    print(f"  {records_path}")


if __name__ == "__main__":
    main()