#!/usr/bin/env python3

import argparse
import glob
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


# ============================================================
# Basic graph helpers
# ============================================================

def as_binary_matrix(A):
    return np.asarray(A, dtype=np.int64)


def edge_set(A, include_self=False):
    A = as_binary_matrix(A)
    n = A.shape[0]

    return {
        (i, j)
        for i in range(n)
        for j in range(n)
        if A[i, j] != 0
        and (include_self or i != j)
    }


def safe_ratio(num, den):
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return num / den


def jaccard(S1, S2):
    union = S1 | S2

    if not union:
        return 1.0

    return len(S1 & S2) / len(union)


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (precision + recall)
    )


# ============================================================
# Reachability
# ============================================================

def reachability_pairs(A):
    """
    Returns ordered reachable node pairs (i, j), i != j.
    """
    A = as_binary_matrix(A)
    n = A.shape[0]

    reachable = set()

    for start in range(n):
        stack = [start]
        visited = set()

        while stack:
            u = stack.pop()

            for v in range(n):
                if (
                    u != v
                    and A[u, v] != 0
                    and v not in visited
                ):
                    visited.add(v)
                    stack.append(v)

        for v in visited:
            if v != start:
                reachable.add((start, v))

    return reachable


# ============================================================
# Degree / source / sink helpers
# ============================================================

def degree_vectors(A):
    A = as_binary_matrix(A).copy()

    np.fill_diagonal(A, 0)

    out_degree = A.sum(axis=1)
    in_degree = A.sum(axis=0)

    return in_degree, out_degree


def source_nodes(A):
    in_degree, _ = degree_vectors(A)

    return {
        i
        for i, d in enumerate(in_degree)
        if d == 0
    }


def sink_nodes(A):
    _, out_degree = degree_vectors(A)

    return {
        i
        for i, d in enumerate(out_degree)
        if d == 0
    }


def pearson_or_nan(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if (
        len(x) == 0
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


# ============================================================
# Exact released-runtime semantics
#
# Empirically confirmed for the released Graph implementation:
#
# For N nodes:
#   fixed_spatial_masks is flattened to N^2 bits
#   potential_spatial_edges has N(N-1) edges
#   zip(...) consumes only the first N(N-1) mask entries
#
# Those bits are interpreted in potential-edge order:
#   0->1, 0->2, ..., 1->0, 1->2, ...
#
# Then Graph rejects edges that would create cycles.
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

    # --------------------------------------------------------
    # Stage A:
    # raw N x N adjacency bit positions
    # -> runtime-interpreted candidate graph
    # --------------------------------------------------------

    def raw_to_mapped(self, raw_A):
        raw = as_binary_matrix(raw_A)

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
    # Stage B:
    # runtime candidate
    # -> cycle-filtered executed DAG
    # --------------------------------------------------------

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

    def mapped_to_executed(self, mapped_A):
        mapped = as_binary_matrix(mapped_A)

        executed = np.zeros(
            (self.n, self.n),
            dtype=np.int64,
        )

        for src, dst in self.potential_edges:

            if mapped[src, dst] == 0:
                continue

            # Adding src -> dst would form a cycle iff
            # dst can already reach src.
            if self._path_exists(
                executed,
                dst,
                src,
            ):
                continue

            executed[src, dst] = 1

        return executed

    def project(self, raw_A):
        mapped = self.raw_to_mapped(
            raw_A
        )

        executed = self.mapped_to_executed(
            mapped
        )

        return mapped, executed


# ============================================================
# Per-record metrics
# ============================================================

def compute_metrics(
    raw,
    mapped,
    executed,
):

    raw = as_binary_matrix(raw)
    mapped = as_binary_matrix(mapped)
    executed = as_binary_matrix(executed)

    n = raw.shape[0]

    raw_edges = edge_set(
        raw,
        include_self=False,
    )

    mapped_edges = edge_set(
        mapped,
        include_self=False,
    )

    exec_edges = edge_set(
        executed,
        include_self=False,
    )

    # --------------------------------------------------------
    # Edge-level semantic fidelity
    # --------------------------------------------------------

    overlap_raw_exec = (
        raw_edges
        & exec_edges
    )

    precision_raw_exec = safe_ratio(
        len(overlap_raw_exec),
        len(exec_edges),
    )

    recall_raw_exec = safe_ratio(
        len(overlap_raw_exec),
        len(raw_edges),
    )

    f1_raw_exec = f1_score(
        precision_raw_exec,
        recall_raw_exec,
    )

    # --------------------------------------------------------
    # Decomposition:
    #
    # raw -> mapped = index/representation effect
    # mapped -> executed = cycle-filtering effect
    # --------------------------------------------------------

    raw_mapped_jaccard = jaccard(
        raw_edges,
        mapped_edges,
    )

    mapped_exec_jaccard = jaccard(
        mapped_edges,
        exec_edges,
    )

    raw_exec_jaccard = jaccard(
        raw_edges,
        exec_edges,
    )

    # --------------------------------------------------------
    # Coordinate Hamming distance
    # Ignore diagonal because runtime has no self-loop edge.
    # --------------------------------------------------------

    mask = ~np.eye(
        n,
        dtype=bool,
    )

    raw_mapped_hamming = int(
        np.sum(
            raw[mask]
            != mapped[mask]
        )
    )

    raw_exec_hamming = int(
        np.sum(
            raw[mask]
            != executed[mask]
        )
    )

    mapped_exec_hamming = int(
        np.sum(
            mapped[mask]
            != executed[mask]
        )
    )

    denom = n * (n - 1)

    # --------------------------------------------------------
    # Reachability fidelity
    # --------------------------------------------------------

    raw_reach = reachability_pairs(
        raw
    )

    mapped_reach = reachability_pairs(
        mapped
    )

    exec_reach = reachability_pairs(
        executed
    )

    # --------------------------------------------------------
    # Sources / sinks
    # --------------------------------------------------------

    raw_sources = source_nodes(
        raw
    )

    exec_sources = source_nodes(
        executed
    )

    raw_sinks = sink_nodes(
        raw
    )

    exec_sinks = sink_nodes(
        executed
    )

    # --------------------------------------------------------
    # Degree correlation
    # --------------------------------------------------------

    raw_in, raw_out = degree_vectors(
        raw
    )

    exec_in, exec_out = degree_vectors(
        executed
    )

    # --------------------------------------------------------
    # Information discarded by current runtime mask semantics
    #
    # flattened positions >= N(N-1) are never consumed.
    # --------------------------------------------------------

    flat = raw.reshape(-1)

    cutoff = n * (n - 1)

    ignored_active_bits = int(
        flat[cutoff:].sum()
    )

    return {
        "raw_edges":
            len(raw_edges),

        "mapped_edges":
            len(mapped_edges),

        "executed_edges":
            len(exec_edges),

        "raw_self_loops":
            int(np.trace(raw)),

        "ignored_active_bits":
            ignored_active_bits,

        "raw_to_mapped_jaccard":
            raw_mapped_jaccard,

        "mapped_to_executed_jaccard":
            mapped_exec_jaccard,

        "raw_to_executed_jaccard":
            raw_exec_jaccard,

        "raw_to_executed_precision":
            precision_raw_exec,

        "raw_to_executed_recall":
            recall_raw_exec,

        "raw_to_executed_f1":
            f1_raw_exec,

        "raw_to_mapped_hamming":
            raw_mapped_hamming,

        "mapped_to_executed_hamming":
            mapped_exec_hamming,

        "raw_to_executed_hamming":
            raw_exec_hamming,

        "raw_to_mapped_hamming_norm":
            raw_mapped_hamming / denom,

        "mapped_to_executed_hamming_norm":
            mapped_exec_hamming / denom,

        "raw_to_executed_hamming_norm":
            raw_exec_hamming / denom,

        "raw_mapped_exact":
            bool(
                np.array_equal(
                    raw,
                    mapped,
                )
            ),

        "mapped_executed_exact":
            bool(
                np.array_equal(
                    mapped,
                    executed,
                )
            ),

        "raw_executed_exact":
            bool(
                np.array_equal(
                    raw,
                    executed,
                )
            ),

        "raw_to_mapped_reach_jaccard":
            jaccard(
                raw_reach,
                mapped_reach,
            ),

        "mapped_to_executed_reach_jaccard":
            jaccard(
                mapped_reach,
                exec_reach,
            ),

        "raw_to_executed_reach_jaccard":
            jaccard(
                raw_reach,
                exec_reach,
            ),

        "source_set_jaccard":
            jaccard(
                raw_sources,
                exec_sources,
            ),

        "sink_set_jaccard":
            jaccard(
                raw_sinks,
                exec_sinks,
            ),

        "in_degree_pearson":
            pearson_or_nan(
                raw_in,
                exec_in,
            ),

        "out_degree_pearson":
            pearson_or_nan(
                raw_out,
                exec_out,
            ),
    }


# ============================================================
# Aggregate helpers
# ============================================================

def mean_metric(records, key):
    values = [
        r[key]
        for r in records
        if isinstance(
            r.get(key),
            (int, float, np.integer, np.floating)
        )
        and not math.isnan(
            float(r[key])
        )
    ]

    if not values:
        return None

    return float(
        np.mean(values)
    )


def fraction_true(records, key):
    if not records:
        return None

    return float(
        np.mean(
            [
                bool(r[key])
                for r in records
            ]
        )
    )


# ============================================================
# File analysis
# ============================================================

def analyze_file(path):

    data = json.load(
        open(
            path,
            encoding="utf-8",
        )
    )

    if not data:
        return None

    first_raw = data[0].get(
        "Generated_Topology"
    )

    if first_raw is None:
        raise ValueError(
            f"{path}: missing Generated_Topology"
        )

    n = len(first_raw)

    runtime = RuntimeSemantics(
        n
    )

    per_record = []

    runtime_exact = 0
    runtime_checked = 0

    for idx, item in enumerate(data):

        raw = item.get(
            "Generated_Topology"
        )

        if raw is None:
            continue

        mapped, predicted_exec = (
            runtime.project(raw)
        )

        saved_exec = item.get(
            "Executed_Topology"
        )

        if saved_exec is not None:

            runtime_checked += 1

            if np.array_equal(
                predicted_exec,
                as_binary_matrix(
                    saved_exec
                ),
            ):
                runtime_exact += 1

            executed = as_binary_matrix(
                saved_exec
            )

        else:

            executed = predicted_exec

        metrics = compute_metrics(
            raw,
            mapped,
            executed,
        )

        metrics.update({
            "record_index":
                idx,

            "seed":
                item.get("Seed"),

            "arm":
                item.get("Arm"),

            "solved":
                item.get("Solved"),

            "question":
                item.get(
                    "Question"
                ),
        })

        per_record.append(
            metrics
        )

    summary = {
        "file":
            path,

        "records":
            len(per_record),

        "runtime_validation_checked":
            runtime_checked,

        "runtime_validation_exact":
            runtime_exact,

        "runtime_validation_rate":
            safe_ratio(
                runtime_exact,
                runtime_checked,
            )
            if runtime_checked > 0
            else None,

        "mean_raw_edges":
            mean_metric(
                per_record,
                "raw_edges",
            ),

        "mean_mapped_edges":
            mean_metric(
                per_record,
                "mapped_edges",
            ),

        "mean_executed_edges":
            mean_metric(
                per_record,
                "executed_edges",
            ),

        "mean_raw_self_loops":
            mean_metric(
                per_record,
                "raw_self_loops",
            ),

        "mean_ignored_active_bits":
            mean_metric(
                per_record,
                "ignored_active_bits",
            ),

        # ----------------------------------------------------
        # Most important fidelity metrics
        # ----------------------------------------------------

        "mean_raw_to_mapped_jaccard":
            mean_metric(
                per_record,
                "raw_to_mapped_jaccard",
            ),

        "mean_mapped_to_executed_jaccard":
            mean_metric(
                per_record,
                "mapped_to_executed_jaccard",
            ),

        "mean_raw_to_executed_jaccard":
            mean_metric(
                per_record,
                "raw_to_executed_jaccard",
            ),

        "mean_raw_to_executed_precision":
            mean_metric(
                per_record,
                "raw_to_executed_precision",
            ),

        "mean_raw_to_executed_recall":
            mean_metric(
                per_record,
                "raw_to_executed_recall",
            ),

        "mean_raw_to_executed_f1":
            mean_metric(
                per_record,
                "raw_to_executed_f1",
            ),

        # ----------------------------------------------------
        # Hamming
        # ----------------------------------------------------

        "mean_raw_to_mapped_hamming_norm":
            mean_metric(
                per_record,
                "raw_to_mapped_hamming_norm",
            ),

        "mean_mapped_to_executed_hamming_norm":
            mean_metric(
                per_record,
                "mapped_to_executed_hamming_norm",
            ),

        "mean_raw_to_executed_hamming_norm":
            mean_metric(
                per_record,
                "raw_to_executed_hamming_norm",
            ),

        # ----------------------------------------------------
        # Exact identity rates
        # ----------------------------------------------------

        "fraction_raw_mapped_exact":
            fraction_true(
                per_record,
                "raw_mapped_exact",
            ),

        "fraction_mapped_executed_exact":
            fraction_true(
                per_record,
                "mapped_executed_exact",
            ),

        "fraction_raw_executed_exact":
            fraction_true(
                per_record,
                "raw_executed_exact",
            ),

        # ----------------------------------------------------
        # Higher-order structural semantics
        # ----------------------------------------------------

        "mean_raw_to_mapped_reach_jaccard":
            mean_metric(
                per_record,
                "raw_to_mapped_reach_jaccard",
            ),

        "mean_mapped_to_executed_reach_jaccard":
            mean_metric(
                per_record,
                "mapped_to_executed_reach_jaccard",
            ),

        "mean_raw_to_executed_reach_jaccard":
            mean_metric(
                per_record,
                "raw_to_executed_reach_jaccard",
            ),

        "mean_source_set_jaccard":
            mean_metric(
                per_record,
                "source_set_jaccard",
            ),

        "mean_sink_set_jaccard":
            mean_metric(
                per_record,
                "sink_set_jaccard",
            ),

        "mean_in_degree_pearson":
            mean_metric(
                per_record,
                "in_degree_pearson",
            ),

        "mean_out_degree_pearson":
            mean_metric(
                per_record,
                "out_degree_pearson",
            ),
    }

    return summary, per_record


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Result JSON files or glob patterns. "
            "Example: "
            "audit_results/downstream_matched_control/*.json"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "audit_results/"
            "topology_semantic_fidelity"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Expand globs
    # --------------------------------------------------------

    files = []

    for pattern in args.inputs:

        matched = glob.glob(
            pattern
        )

        if matched:
            files.extend(
                matched
            )

        elif Path(pattern).exists():
            files.append(
                pattern
            )

    files = sorted(
        set(files)
    )

    if not files:
        raise RuntimeError(
            "No input files found."
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_summaries = []
    all_records = []

    print("=" * 84)
    print("TOPOLOGY SEMANTIC FIDELITY AUDIT")
    print("=" * 84)

    print(
        "Files:",
        len(files)
    )

    # --------------------------------------------------------
    # Analyze each file
    # --------------------------------------------------------

    for path in files:

        result = analyze_file(
            path
        )

        if result is None:
            continue

        summary, per_record = result

        all_summaries.append(
            summary
        )

        for row in per_record:

            row = dict(row)

            row[
                "source_file"
            ] = path

            all_records.append(
                row
            )

        print()
        print("-" * 84)

        print(
            Path(path).name
        )

        print(
            "records:",
            summary["records"]
        )

        print(
            "runtime validation:",
            summary[
                "runtime_validation_rate"
            ],
        )

        print()
        print(
            "raw -> mapped Jaccard:",
            round(
                summary[
                    "mean_raw_to_mapped_jaccard"
                ],
                4,
            ),
        )

        print(
            "mapped -> executed Jaccard:",
            round(
                summary[
                    "mean_mapped_to_executed_jaccard"
                ],
                4,
            ),
        )

        print(
            "raw -> executed Jaccard:",
            round(
                summary[
                    "mean_raw_to_executed_jaccard"
                ],
                4,
            ),
        )

        print(
            "raw -> executed precision:",
            round(
                summary[
                    "mean_raw_to_executed_precision"
                ],
                4,
            ),
        )

        print(
            "raw -> executed recall:",
            round(
                summary[
                    "mean_raw_to_executed_recall"
                ],
                4,
            ),
        )

        print(
            "raw -> executed reachability Jaccard:",
            round(
                summary[
                    "mean_raw_to_executed_reach_jaccard"
                ],
                4,
            ),
        )

        print(
            "fraction raw == executed:",
            round(
                summary[
                    "fraction_raw_executed_exact"
                ],
                4,
            ),
        )

    # ========================================================
    # Global aggregate
    # ========================================================

    print()
    print("=" * 84)
    print("GLOBAL SUMMARY")
    print("=" * 84)

    global_summary = {
        "files":
            len(all_summaries),

        "records":
            len(all_records),
    }

    aggregate_keys = [
        "raw_edges",
        "mapped_edges",
        "executed_edges",
        "raw_self_loops",
        "ignored_active_bits",

        "raw_to_mapped_jaccard",
        "mapped_to_executed_jaccard",
        "raw_to_executed_jaccard",

        "raw_to_executed_precision",
        "raw_to_executed_recall",
        "raw_to_executed_f1",

        "raw_to_mapped_hamming_norm",
        "mapped_to_executed_hamming_norm",
        "raw_to_executed_hamming_norm",

        "raw_to_mapped_reach_jaccard",
        "mapped_to_executed_reach_jaccard",
        "raw_to_executed_reach_jaccard",

        "source_set_jaccard",
        "sink_set_jaccard",

        "in_degree_pearson",
        "out_degree_pearson",
    ]

    for key in aggregate_keys:

        global_summary[
            f"mean_{key}"
        ] = mean_metric(
            all_records,
            key,
        )

    global_summary[
        "fraction_raw_mapped_exact"
    ] = fraction_true(
        all_records,
        "raw_mapped_exact",
    )

    global_summary[
        "fraction_mapped_executed_exact"
    ] = fraction_true(
        all_records,
        "mapped_executed_exact",
    )

    global_summary[
        "fraction_raw_executed_exact"
    ] = fraction_true(
        all_records,
        "raw_executed_exact",
    )

    # --------------------------------------------------------
    # Distribution of semantic fidelity
    # --------------------------------------------------------

    jaccards = [
        r[
            "raw_to_executed_jaccard"
        ]
        for r in all_records
    ]

    global_summary[
        "fraction_raw_exec_jaccard_lt_0_25"
    ] = float(
        np.mean(
            np.asarray(
                jaccards
            )
            < 0.25
        )
    )

    global_summary[
        "fraction_raw_exec_jaccard_lt_0_50"
    ] = float(
        np.mean(
            np.asarray(
                jaccards
            )
            < 0.50
        )
    )

    global_summary[
        "fraction_raw_exec_jaccard_lt_0_75"
    ] = float(
        np.mean(
            np.asarray(
                jaccards
            )
            < 0.75
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    for key, value in global_summary.items():

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

    with open(
        output_dir
        / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "global":
                    global_summary,

                "per_file":
                    all_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(
        output_dir
        / "per_record.jsonl",
        "w",
        encoding="utf-8",
    ) as f:

        for row in all_records:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print(
        "Saved:",
        output_dir
        / "summary.json"
    )

    print(
        "Saved:",
        output_dir
        / "per_record.jsonl"
    )


if __name__ == "__main__":
    main()
