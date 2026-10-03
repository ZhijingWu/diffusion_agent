#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np


# ============================================================
# Helpers
# ============================================================

def as_matrix(A):
    return np.asarray(A, dtype=np.int64)


def edge_set(A, include_self=False):
    A = as_matrix(A)
    n = A.shape[0]

    return {
        (i, j)
        for i in range(n)
        for j in range(n)
        if A[i, j] != 0
        and (include_self or i != j)
    }


def topo_key(A):
    A = as_matrix(A)

    return tuple(
        tuple(int(x) for x in row)
        for row in A
    )


def safe_mean(xs):
    if not xs:
        return None

    return float(
        np.mean(xs)
    )


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return None

    if np.std(x) == 0 or np.std(y) == 0:
        return None

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def mae(x, y):
    if not x:
        return None

    return float(
        np.mean(
            np.abs(
                np.asarray(x, dtype=float)
                -
                np.asarray(y, dtype=float)
            )
        )
    )


def jaccard(S1, S2):
    union = S1 | S2

    if not union:
        return 1.0

    return len(S1 & S2) / len(union)


# ============================================================
# Released runtime semantics
# ============================================================

class RuntimeSemantics:

    def __init__(self, n):
        self.n = n

        self.potential_edges = [
            (i, j)
            for i in range(n)
            for j in range(n)
            if i != j
        ]

    @staticmethod
    def _path_exists(A, start, target):
        if start == target:
            return True

        n = A.shape[0]
        stack = [start]
        visited = set()

        while stack:
            u = stack.pop()

            if u == target:
                return True

            if u in visited:
                continue

            visited.add(u)

            for v in range(n):
                if A[u, v] != 0:
                    stack.append(v)

        return False

    # --------------------------------------------------------
    # Current released runtime interpretation
    # --------------------------------------------------------

    def raw_to_old_mapped(self, raw_A):
        raw = as_matrix(raw_A)

        flat = raw.reshape(-1).tolist()

        mapped = np.zeros(
            (self.n, self.n),
            dtype=np.int64,
        )

        for bit, (src, dst) in zip(
            flat,
            self.potential_edges,
        ):
            if bit:
                mapped[src, dst] = 1

        return mapped

    # --------------------------------------------------------
    # Cycle filtering
    # --------------------------------------------------------

    def cycle_filter(self, A):
        A = as_matrix(A)

        out = np.zeros(
            (self.n, self.n),
            dtype=np.int64,
        )

        for src, dst in self.potential_edges:

            if A[src, dst] == 0:
                continue

            if self._path_exists(
                out,
                dst,
                src,
            ):
                continue

            out[src, dst] = 1

        return out

    def old_execute(self, raw_A):
        mapped = self.raw_to_old_mapped(
            raw_A
        )

        executed = self.cycle_filter(
            mapped
        )

        return mapped, executed

    # --------------------------------------------------------
    # Canonical execution:
    #
    # A[i,j] means i -> j
    # diagonal removed
    # then same cycle rule
    # --------------------------------------------------------

    def canonical_execute(self, raw_A):
        raw = as_matrix(raw_A).copy()

        np.fill_diagonal(
            raw,
            0,
        )

        return self.cycle_filter(
            raw
        )


# ============================================================
# Data loading
# ============================================================

def load_jsonl(path):
    rows = []

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)

            except Exception as e:
                raise RuntimeError(
                    f"Invalid JSON on line {line_no}: {e}"
                )

            rows.append(obj)

    return rows


def get_performance(row):
    perf = row.get(
        "performance",
        {}
    )

    utility = perf.get(
        "utility"
    )

    cost = perf.get(
        "cost"
    )

    return utility, cost


# ============================================================
# Main audit
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="gtd_gsm8k_dataset_training.jsonl",
    )

    parser.add_argument(
        "--output",
        default=(
            "audit_results/"
            "training_execution_semantics/"
            "summary.json"
        ),
    )

    args = parser.parse_args()

    rows = load_jsonl(
        args.dataset
    )

    if not rows:
        raise RuntimeError(
            "Dataset is empty."
        )

    if "graph" not in rows[0]:
        raise RuntimeError(
            "Dataset rows do not contain 'graph'."
        )

    n = len(
        rows[0]["graph"]
    )

    runtime = RuntimeSemantics(
        n
    )

    records = []

    # --------------------------------------------------------
    # Per sample
    # --------------------------------------------------------

    for idx, row in enumerate(rows):

        raw = as_matrix(
            row["graph"]
        )

        utility, cost = (
            get_performance(
                row
            )
        )

        old_mapped, old_exec = (
            runtime.old_execute(
                raw
            )
        )

        canonical_exec = (
            runtime.canonical_execute(
                raw
            )
        )

        raw_nonself = raw.copy()

        np.fill_diagonal(
            raw_nonself,
            0,
        )

        raw_edges = len(
            edge_set(
                raw_nonself
            )
        )

        raw_edges_including_self = int(
            raw.sum()
        )

        old_mapped_edges = len(
            edge_set(
                old_mapped
            )
        )

        old_exec_edges = len(
            edge_set(
                old_exec
            )
        )

        canonical_exec_edges = len(
            edge_set(
                canonical_exec
            )
        )

        rec = {
            "index":
                idx,

            "utility":
                utility,

            "cost":
                cost,

            "raw_edges_nonself":
                raw_edges,

            "raw_edges_including_self":
                raw_edges_including_self,

            "self_loops":
                int(
                    np.trace(raw)
                ),

            "old_mapped_edges":
                old_mapped_edges,

            "old_executed_edges":
                old_exec_edges,

            "canonical_executed_edges":
                canonical_exec_edges,

            "raw_old_mapped_jaccard":
                jaccard(
                    edge_set(raw_nonself),
                    edge_set(old_mapped),
                ),

            "raw_old_executed_jaccard":
                jaccard(
                    edge_set(raw_nonself),
                    edge_set(old_exec),
                ),

            "raw_canonical_executed_jaccard":
                jaccard(
                    edge_set(raw_nonself),
                    edge_set(canonical_exec),
                ),

            "raw_key":
                topo_key(raw),

            "old_exec_key":
                topo_key(old_exec),

            "canonical_exec_key":
                topo_key(
                    canonical_exec
                ),
        }

        records.append(
            rec
        )

    # ========================================================
    # Basic dataset statistics
    # ========================================================

    utility_vals = [
        r["utility"]
        for r in records
        if r["utility"] is not None
    ]

    cost_vals = [
        r["cost"]
        for r in records
        if r["cost"] is not None
    ]

    raw_counts = [
        r[
            "raw_edges_nonself"
        ]
        for r in records
    ]

    raw_counts_all = [
        r[
            "raw_edges_including_self"
        ]
        for r in records
    ]

    old_mapped_counts = [
        r[
            "old_mapped_edges"
        ]
        for r in records
    ]

    old_exec_counts = [
        r[
            "old_executed_edges"
        ]
        for r in records
    ]

    canonical_exec_counts = [
        r[
            "canonical_executed_edges"
        ]
        for r in records
    ]

    # ========================================================
    # Cost alignment
    #
    # This is important:
    #
    # Which graph semantics does the stored training "cost"
    # appear to describe?
    # ========================================================

    cost_alignment = {}

    if len(cost_vals) == len(records):

        cost_alignment = {

            "cost_vs_raw_nonself": {
                "mae":
                    mae(
                        cost_vals,
                        raw_counts,
                    ),

                "pearson":
                    pearson(
                        cost_vals,
                        raw_counts,
                    ),
            },

            "cost_vs_raw_including_self": {
                "mae":
                    mae(
                        cost_vals,
                        raw_counts_all,
                    ),

                "pearson":
                    pearson(
                        cost_vals,
                        raw_counts_all,
                    ),
            },

            "cost_vs_old_mapped": {
                "mae":
                    mae(
                        cost_vals,
                        old_mapped_counts,
                    ),

                "pearson":
                    pearson(
                        cost_vals,
                        old_mapped_counts,
                    ),
            },

            "cost_vs_old_executed": {
                "mae":
                    mae(
                        cost_vals,
                        old_exec_counts,
                    ),

                "pearson":
                    pearson(
                        cost_vals,
                        old_exec_counts,
                    ),
            },

            "cost_vs_canonical_executed": {
                "mae":
                    mae(
                        cost_vals,
                        canonical_exec_counts,
                    ),

                "pearson":
                    pearson(
                        cost_vals,
                        canonical_exec_counts,
                    ),
            },
        }

    # ========================================================
    # Group by RAW topology
    # ========================================================

    raw_groups = defaultdict(
        list
    )

    for r in records:
        raw_groups[
            r["raw_key"]
        ].append(
            r
        )

    raw_group_summary = []

    for key, group in raw_groups.items():

        utils = [
            x["utility"]
            for x in group
            if x["utility"] is not None
        ]

        costs = [
            x["cost"]
            for x in group
            if x["cost"] is not None
        ]

        raw_group_summary.append({
            "count":
                len(group),

            "raw_edges":
                group[0][
                    "raw_edges_nonself"
                ],

            "old_executed_edges":
                group[0][
                    "old_executed_edges"
                ],

            "canonical_executed_edges":
                group[0][
                    "canonical_executed_edges"
                ],

            "mean_utility":
                safe_mean(utils),

            "mean_cost":
                safe_mean(costs),

            "raw_old_exec_jaccard":
                group[0][
                    "raw_old_executed_jaccard"
                ],

            "raw_canonical_exec_jaccard":
                group[0][
                    "raw_canonical_executed_jaccard"
                ],
        })

    raw_group_summary.sort(
        key=lambda x:
        (
            -x["count"],
            x["raw_edges"],
        )
    )

    # ========================================================
    # Execution alias groups
    #
    # Different raw graphs that become the same OLD executed
    # graph are especially interesting.
    # ========================================================

    exec_groups = defaultdict(
        list
    )

    for r in records:
        exec_groups[
            r["old_exec_key"]
        ].append(
            r
        )

    collision_groups = []

    for exec_key, group in exec_groups.items():

        distinct_raw = {
            r["raw_key"]
            for r in group
        }

        if len(
            distinct_raw
        ) <= 1:
            continue

        utils = [
            x["utility"]
            for x in group
            if x["utility"] is not None
        ]

        costs = [
            x["cost"]
            for x in group
            if x["cost"] is not None
        ]

        collision_groups.append({

            "num_records":
                len(group),

            "num_distinct_raw_graphs":
                len(
                    distinct_raw
                ),

            "mean_utility":
                safe_mean(utils),

            "utility_values":
                sorted(
                    set(utils)
                ),

            "mean_cost":
                safe_mean(costs),

            "cost_values":
                sorted(
                    set(costs)
                ),
        })

    collision_groups.sort(
        key=lambda x:
        (
            -x[
                "num_distinct_raw_graphs"
            ],
            -x[
                "num_records"
            ],
        )
    )

    # ========================================================
    # Semantic consistency metrics
    # ========================================================

    mean_raw_old_map_j = safe_mean([
        r[
            "raw_old_mapped_jaccard"
        ]
        for r in records
    ])

    mean_raw_old_exec_j = safe_mean([
        r[
            "raw_old_executed_jaccard"
        ]
        for r in records
    ])

    mean_raw_canonical_exec_j = safe_mean([
        r[
            "raw_canonical_executed_jaccard"
        ]
        for r in records
    ])

    # ========================================================
    # Summary
    # ========================================================

    summary = {

        "dataset":
            args.dataset,

        "records":
            len(records),

        "num_nodes":
            n,

        "unique_raw_topologies":
            len(
                raw_groups
            ),

        "unique_old_executed_topologies":
            len(
                exec_groups
            ),

        "mean_utility":
            safe_mean(
                utility_vals
            ),

        "mean_cost":
            safe_mean(
                cost_vals
            ),

        "mean_raw_edges_nonself":
            safe_mean(
                raw_counts
            ),

        "mean_old_mapped_edges":
            safe_mean(
                old_mapped_counts
            ),

        "mean_old_executed_edges":
            safe_mean(
                old_exec_counts
            ),

        "mean_canonical_executed_edges":
            safe_mean(
                canonical_exec_counts
            ),

        "mean_raw_to_old_mapped_jaccard":
            mean_raw_old_map_j,

        "mean_raw_to_old_executed_jaccard":
            mean_raw_old_exec_j,

        "mean_raw_to_canonical_executed_jaccard":
            mean_raw_canonical_exec_j,

        "cost_alignment":
            cost_alignment,

        "execution_collision_groups":
            len(
                collision_groups
            ),

        "raw_topology_groups":
            raw_group_summary,

        "top_execution_collisions":
            collision_groups[:20],
    }

    # ========================================================
    # Print
    # ========================================================

    print("=" * 84)
    print("TRAINING -> EXECUTION SEMANTICS AUDIT")
    print("=" * 84)

    print(
        "Dataset:",
        args.dataset
    )

    print(
        "Records:",
        len(records)
    )

    print(
        "Nodes:",
        n
    )

    print(
        "Unique raw topologies:",
        len(raw_groups)
    )

    print(
        "Unique OLD executed topologies:",
        len(exec_groups)
    )

    print()

    print("=" * 84)
    print("SEMANTIC FIDELITY")
    print("=" * 84)

    print(
        "mean raw -> OLD mapped Jaccard:",
        f"{mean_raw_old_map_j:.6f}"
    )

    print(
        "mean raw -> OLD executed Jaccard:",
        f"{mean_raw_old_exec_j:.6f}"
    )

    print(
        "mean raw -> CANONICAL executed Jaccard:",
        f"{mean_raw_canonical_exec_j:.6f}"
    )

    print()

    print("=" * 84)
    print("COST LABEL ALIGNMENT")
    print("=" * 84)

    if not cost_alignment:

        print(
            "No usable performance.cost labels found."
        )

    else:

        for name, values in (
            cost_alignment.items()
        ):

            print(
                f"{name:32s} "
                f"MAE={values['mae']:.6f} "
                f"Pearson={values['pearson']}"
            )

    print()

    print("=" * 84)
    print("RAW TOPOLOGY GROUPS")
    print("=" * 84)

    for i, x in enumerate(
        raw_group_summary,
        1,
    ):

        print(
            f"#{i:02d} "
            f"n={x['count']:3d} | "
            f"raw_edges={x['raw_edges']:2d} | "
            f"old_exec={x['old_executed_edges']:2d} | "
            f"canonical_exec={x['canonical_executed_edges']:2d} | "
            f"utility={x['mean_utility']:.4f} | "
            f"cost={x['mean_cost']:.4f} | "
            f"J_old={x['raw_old_exec_jaccard']:.4f} | "
            f"J_canonical={x['raw_canonical_exec_jaccard']:.4f}"
        )

    print()

    print("=" * 84)
    print("EXECUTION COLLISIONS")
    print("=" * 84)

    print(
        "OLD executed topologies receiving >1 distinct raw graph:",
        len(
            collision_groups
        )
    )

    for i, x in enumerate(
        collision_groups[:10],
        1,
    ):

        print(
            f"#{i:02d} "
            f"records={x['num_records']} | "
            f"distinct_raw={x['num_distinct_raw_graphs']} | "
            f"utility_values={x['utility_values']} | "
            f"cost_values={x['cost_values']}"
        )

    # ========================================================
    # Automatic diagnosis
    # ========================================================

    print()
    print("=" * 84)
    print("AUTOMATIC DIAGNOSIS")
    print("=" * 84)

    if cost_alignment:

        best_cost_semantics = min(
            cost_alignment.items(),
            key=lambda kv:
            kv[1]["mae"]
        )

        print(
            "Stored cost aligns most closely with:",
            best_cost_semantics[0]
        )

        print(
            "Best MAE:",
            best_cost_semantics[1]["mae"]
        )

    print(
        "Raw graph vs OLD execution fidelity:",
        f"{mean_raw_old_exec_j:.4f}"
    )

    print(
        "Raw graph vs CANONICAL execution fidelity:",
        f"{mean_raw_canonical_exec_j:.4f}"
    )

    if (
        mean_raw_canonical_exec_j
        >
        mean_raw_old_exec_j
        + 0.25
    ):

        print(
            "\nRESULT: training graph labels are structurally "
            "much closer to canonical edge semantics than to "
            "the released old runtime execution semantics."
        )

    else:

        print(
            "\nRESULT: canonical semantics do not show a large "
            "fidelity advantage on this training dataset."
        )

    if collision_groups:

        print(
            "\nNOTE: multiple distinct raw training graphs collapse "
            "to identical old-runtime executed graphs. This means "
            "the supervised graph representation can distinguish "
            "examples that the runtime behavior cannot."
        )

    # ========================================================
    # Save
    # ========================================================

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print()
    print(
        "Saved:",
        out
    )


if __name__ == "__main__":
    main()
