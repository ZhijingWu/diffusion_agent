#!/usr/bin/env python3

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


# ============================================================
# Basic helpers
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


def reachability_pairs(A):
    """
    Returns ordered reachable pairs (i, j), i != j.
    """
    A = as_binary_matrix(A)
    n = A.shape[0]

    result = set()

    for start in range(n):
        stack = [start]
        visited = set()

        while stack:
            u = stack.pop()

            for v in range(n):
                if (
                    A[u, v] != 0
                    and v not in visited
                ):
                    visited.add(v)
                    stack.append(v)

        for v in visited:
            if v != start:
                result.add((start, v))

    return result


def mean_metric(records, key):
    vals = []

    for r in records:
        v = r.get(key)

        if isinstance(
            v,
            (int, float, np.integer, np.floating)
        ):
            v = float(v)

            if not math.isnan(v):
                vals.append(v)

    if not vals:
        return None

    return float(np.mean(vals))


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
# Current released runtime semantics
#
# Confirmed experimentally:
#
# flattened raw adjacency:
# [A00,A01,A02,A03,A10,...]
#
# is paired against:
# [(0,1),(0,2),(0,3),
#  (1,0),(1,2),(1,3),
#  (2,0),(2,1),(2,3),
#  (3,0),(3,1),(3,2)]
#
# only first N(N-1) flattened positions are consumed.
# ============================================================

class CurrentRuntime:

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

    def cycle_filter(self, candidate_A):
        candidate_A = as_binary_matrix(
            candidate_A
        )

        executed = np.zeros(
            (self.n, self.n),
            dtype=np.int64,
        )

        for src, dst in self.potential_edges:

            if candidate_A[src, dst] == 0:
                continue

            # Adding src -> dst causes cycle iff
            # dst can already reach src.
            if self._path_exists(
                executed,
                dst,
                src,
            ):
                continue

            executed[src, dst] = 1

        return executed

    def execute_old(self, raw_A):
        mapped = self.raw_to_mapped(
            raw_A
        )

        executed = self.cycle_filter(
            mapped
        )

        return mapped, executed


# ============================================================
# Canonical alignment
#
# Goal:
#
# Generator semantics:
#   A[i,j] means edge i -> j, i != j
#
# Runtime currently consumes first N(N-1) flattened positions.
#
# We therefore build a runtime-input matrix whose first
# N(N-1) flattened bits are exactly the canonical non-self
# edge mask:
#
#   (0->1),(0->2),...,(1->0),...
#
# Then the current runtime's misaligned flatten+zip behavior
# reconstructs the intended adjacency.
# ============================================================

class CanonicalRuntimeRepair:

    def __init__(self, n):
        self.n = n

        self.potential_edges = [
            (i, j)
            for i in range(n)
            for j in range(n)
            if i != j
        ]

        self.current_runtime = (
            CurrentRuntime(n)
        )

    def adjacency_to_canonical_mask(
        self,
        raw_A,
    ):
        raw = as_binary_matrix(raw_A)

        return [
            int(raw[i, j])
            for i, j
            in self.potential_edges
        ]

    def canonical_mask_to_runtime_input(
        self,
        mask,
    ):
        """
        Put canonical N(N-1) edge bits into the first
        N(N-1) flattened positions expected by the current
        buggy runtime.

        Remaining N positions are zero.
        """
        total = self.n * self.n

        flat = [0] * total

        for k, bit in enumerate(mask):
            flat[k] = int(bit)

        return np.asarray(
            flat,
            dtype=np.int64,
        ).reshape(
            self.n,
            self.n,
        )

    def build_aligned_runtime_input(
        self,
        raw_A,
    ):
        mask = (
            self.adjacency_to_canonical_mask(
                raw_A
            )
        )

        runtime_input = (
            self.canonical_mask_to_runtime_input(
                mask
            )
        )

        return runtime_input

    def execute_repaired(
        self,
        raw_A,
    ):
        runtime_input = (
            self.build_aligned_runtime_input(
                raw_A
            )
        )

        # Feed repaired input through the exact same
        # released runtime semantics.
        mapped = (
            self.current_runtime
            .raw_to_mapped(
                runtime_input
            )
        )

        executed = (
            self.current_runtime
            .cycle_filter(
                mapped
            )
        )

        return (
            runtime_input,
            mapped,
            executed,
        )


# ============================================================
# Per-record comparison
# ============================================================

def compare_one(
    raw,
    saved_exec=None,
):
    raw = as_binary_matrix(raw)

    n = raw.shape[0]

    old_runtime = CurrentRuntime(n)
    repair = CanonicalRuntimeRepair(n)

    # --------------------------------------------------------
    # Old pipeline
    # --------------------------------------------------------

    old_mapped, old_exec_pred = (
        old_runtime.execute_old(raw)
    )

    if saved_exec is not None:
        old_exec = as_binary_matrix(
            saved_exec
        )
    else:
        old_exec = old_exec_pred

    # --------------------------------------------------------
    # Repaired pipeline
    # --------------------------------------------------------

    (
        repaired_runtime_input,
        repaired_mapped,
        repaired_exec,
    ) = repair.execute_repaired(
        raw
    )

    # --------------------------------------------------------
    # Edge sets
    # --------------------------------------------------------

    raw_edges = edge_set(
        raw,
        include_self=False,
    )

    old_edges = edge_set(
        old_exec,
        include_self=False,
    )

    repaired_edges = edge_set(
        repaired_exec,
        include_self=False,
    )

    repaired_mapped_edges = edge_set(
        repaired_mapped,
        include_self=False,
    )

    # --------------------------------------------------------
    # Old fidelity
    # --------------------------------------------------------

    old_intersection = (
        raw_edges
        & old_edges
    )

    old_precision = safe_ratio(
        len(old_intersection),
        len(old_edges),
    )

    old_recall = safe_ratio(
        len(old_intersection),
        len(raw_edges),
    )

    old_f1 = f1_score(
        old_precision,
        old_recall,
    )

    old_jaccard = jaccard(
        raw_edges,
        old_edges,
    )

    # --------------------------------------------------------
    # Repaired fidelity
    # --------------------------------------------------------

    new_intersection = (
        raw_edges
        & repaired_edges
    )

    new_precision = safe_ratio(
        len(new_intersection),
        len(repaired_edges),
    )

    new_recall = safe_ratio(
        len(new_intersection),
        len(raw_edges),
    )

    new_f1 = f1_score(
        new_precision,
        new_recall,
    )

    new_jaccard = jaccard(
        raw_edges,
        repaired_edges,
    )

    # --------------------------------------------------------
    # Pure cycle-only residual
    #
    # repaired_mapped should equal intended raw graph
    # after removing self-loops.
    # --------------------------------------------------------

    intended_no_self = raw.copy()
    np.fill_diagonal(
        intended_no_self,
        0,
    )

    intended_edges = edge_set(
        intended_no_self,
        include_self=False,
    )

    mapped_edges = edge_set(
        repaired_mapped,
        include_self=False,
    )

    cycle_only_jaccard = jaccard(
        mapped_edges,
        repaired_edges,
    )

    # --------------------------------------------------------
    # Reachability
    # --------------------------------------------------------

    raw_reach = reachability_pairs(
        raw
    )

    old_reach = reachability_pairs(
        old_exec
    )

    repaired_reach = (
        reachability_pairs(
            repaired_exec
        )
    )

    old_reach_j = jaccard(
        raw_reach,
        old_reach,
    )

    new_reach_j = jaccard(
        raw_reach,
        repaired_reach,
    )

    # --------------------------------------------------------
    # Hamming ignoring diagonal
    # --------------------------------------------------------

    mask = ~np.eye(
        n,
        dtype=bool,
    )

    old_hamming = int(
        np.sum(
            raw[mask]
            != old_exec[mask]
        )
    )

    new_hamming = int(
        np.sum(
            raw[mask]
            != repaired_exec[mask]
        )
    )

    denom = n * (n - 1)

    # --------------------------------------------------------
    # Exactness
    # --------------------------------------------------------

    raw_no_self = raw.copy()
    np.fill_diagonal(
        raw_no_self,
        0,
    )

    old_exact = bool(
        np.array_equal(
            raw_no_self,
            old_exec,
        )
    )

    repaired_exact = bool(
        np.array_equal(
            raw_no_self,
            repaired_exec,
        )
    )

    # --------------------------------------------------------
    # How much coordinate mismatch is removed before cycles
    # --------------------------------------------------------

    repaired_mapping_exact = bool(
        np.array_equal(
            raw_no_self,
            repaired_mapped,
        )
    )

    return {
        "raw_edges":
            len(raw_edges),

        "old_exec_edges":
            len(old_edges),

        "repaired_exec_edges":
            len(repaired_edges),

        "raw_self_loops":
            int(np.trace(raw)),

        # OLD
        "old_jaccard":
            old_jaccard,

        "old_precision":
            old_precision,

        "old_recall":
            old_recall,

        "old_f1":
            old_f1,

        "old_reachability_jaccard":
            old_reach_j,

        "old_hamming_norm":
            old_hamming / denom,

        "old_exact":
            old_exact,

        # NEW
        "repaired_jaccard":
            new_jaccard,

        "repaired_precision":
            new_precision,

        "repaired_recall":
            new_recall,

        "repaired_f1":
            new_f1,

        "repaired_reachability_jaccard":
            new_reach_j,

        "repaired_hamming_norm":
            new_hamming / denom,

        "repaired_exact":
            repaired_exact,

        # GAINS
        "jaccard_gain":
            new_jaccard
            - old_jaccard,

        "precision_gain":
            new_precision
            - old_precision,

        "recall_gain":
            new_recall
            - old_recall,

        "f1_gain":
            new_f1
            - old_f1,

        "reachability_gain":
            new_reach_j
            - old_reach_j,

        "hamming_reduction":
            (
                old_hamming
                - new_hamming
            )
            / denom,

        # residual structural constraint effect
        "cycle_only_jaccard":
            cycle_only_jaccard,

        "repaired_mapping_exact":
            repaired_mapping_exact,
    }


# ============================================================
# Analyze one JSON
# ============================================================

def analyze_file(path):

    data = json.load(
        open(
            path,
            encoding="utf-8",
        )
    )

    rows = []

    old_validation_total = 0
    old_validation_exact = 0

    for idx, item in enumerate(data):

        raw = item.get(
            "Generated_Topology"
        )

        if raw is None:
            continue

        saved_exec = item.get(
            "Executed_Topology"
        )

        row = compare_one(
            raw,
            saved_exec=saved_exec,
        )

        # Validate old projector against real saved execution
        if saved_exec is not None:
            n = len(raw)

            runtime = CurrentRuntime(n)

            _, predicted = (
                runtime.execute_old(
                    raw
                )
            )

            old_validation_total += 1

            old_validation_exact += int(
                np.array_equal(
                    predicted,
                    as_binary_matrix(
                        saved_exec
                    ),
                )
            )

        row.update({
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

        rows.append(
            row
        )

    summary = {
        "file":
            path,

        "records":
            len(rows),

        "old_runtime_validation_rate":
            (
                old_validation_exact
                / old_validation_total
                if old_validation_total
                else None
            ),

        # Old
        "mean_old_jaccard":
            mean_metric(
                rows,
                "old_jaccard",
            ),

        "mean_old_precision":
            mean_metric(
                rows,
                "old_precision",
            ),

        "mean_old_recall":
            mean_metric(
                rows,
                "old_recall",
            ),

        "mean_old_f1":
            mean_metric(
                rows,
                "old_f1",
            ),

        "mean_old_reachability_jaccard":
            mean_metric(
                rows,
                "old_reachability_jaccard",
            ),

        "mean_old_hamming_norm":
            mean_metric(
                rows,
                "old_hamming_norm",
            ),

        "fraction_old_exact":
            fraction_true(
                rows,
                "old_exact",
            ),

        # Repaired
        "mean_repaired_jaccard":
            mean_metric(
                rows,
                "repaired_jaccard",
            ),

        "mean_repaired_precision":
            mean_metric(
                rows,
                "repaired_precision",
            ),

        "mean_repaired_recall":
            mean_metric(
                rows,
                "repaired_recall",
            ),

        "mean_repaired_f1":
            mean_metric(
                rows,
                "repaired_f1",
            ),

        "mean_repaired_reachability_jaccard":
            mean_metric(
                rows,
                "repaired_reachability_jaccard",
            ),

        "mean_repaired_hamming_norm":
            mean_metric(
                rows,
                "repaired_hamming_norm",
            ),

        "fraction_repaired_exact":
            fraction_true(
                rows,
                "repaired_exact",
            ),

        # Gains
        "mean_jaccard_gain":
            mean_metric(
                rows,
                "jaccard_gain",
            ),

        "mean_precision_gain":
            mean_metric(
                rows,
                "precision_gain",
            ),

        "mean_recall_gain":
            mean_metric(
                rows,
                "recall_gain",
            ),

        "mean_f1_gain":
            mean_metric(
                rows,
                "f1_gain",
            ),

        "mean_reachability_gain":
            mean_metric(
                rows,
                "reachability_gain",
            ),

        "mean_hamming_reduction":
            mean_metric(
                rows,
                "hamming_reduction",
            ),

        # Residual cycle effect
        "mean_cycle_only_jaccard":
            mean_metric(
                rows,
                "cycle_only_jaccard",
            ),

        "fraction_repaired_mapping_exact":
            fraction_true(
                rows,
                "repaired_mapping_exact",
            ),
    }

    return summary, rows


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
            "JSON files or glob patterns."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "audit_results/"
            "canonical_runtime_repair"
        ),
    )

    args = parser.parse_args()

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

    all_rows = []
    per_file = []

    print("=" * 88)
    print("CANONICAL RUNTIME REPAIR AUDIT")
    print("=" * 88)

    print(
        "files:",
        len(files)
    )

    for path in files:

        summary, rows = (
            analyze_file(
                path
            )
        )

        per_file.append(
            summary
        )

        for row in rows:

            row = dict(row)

            row[
                "source_file"
            ] = path

            all_rows.append(
                row
            )

        print()
        print("-" * 88)

        print(
            Path(path).name
        )

        print(
            "records:",
            summary["records"]
        )

        print(
            "old runtime validation:",
            summary[
                "old_runtime_validation_rate"
            ],
        )

        print(
            "OLD Jaccard:",
            round(
                summary[
                    "mean_old_jaccard"
                ],
                4,
            ),
        )

        print(
            "REPAIRED Jaccard:",
            round(
                summary[
                    "mean_repaired_jaccard"
                ],
                4,
            ),
        )

        print(
            "GAIN:",
            round(
                summary[
                    "mean_jaccard_gain"
                ],
                4,
            ),
        )

        print(
            "cycle-only residual Jaccard:",
            round(
                summary[
                    "mean_cycle_only_jaccard"
                ],
                4,
            ),
        )

    # ========================================================
    # Global summary
    # ========================================================

    global_summary = {
        "files":
            len(files),

        "records":
            len(all_rows),
    }

    keys = [
        "old_jaccard",
        "repaired_jaccard",
        "jaccard_gain",

        "old_precision",
        "repaired_precision",
        "precision_gain",

        "old_recall",
        "repaired_recall",
        "recall_gain",

        "old_f1",
        "repaired_f1",
        "f1_gain",

        "old_reachability_jaccard",
        "repaired_reachability_jaccard",
        "reachability_gain",

        "old_hamming_norm",
        "repaired_hamming_norm",
        "hamming_reduction",

        "cycle_only_jaccard",
    ]

    for key in keys:

        global_summary[
            f"mean_{key}"
        ] = mean_metric(
            all_rows,
            key,
        )

    global_summary[
        "fraction_old_exact"
    ] = fraction_true(
        all_rows,
        "old_exact",
    )

    global_summary[
        "fraction_repaired_exact"
    ] = fraction_true(
        all_rows,
        "repaired_exact",
    )

    global_summary[
        "fraction_repaired_mapping_exact"
    ] = fraction_true(
        all_rows,
        "repaired_mapping_exact",
    )

    # How often repair materially helps
    global_summary[
        "fraction_jaccard_improved"
    ] = float(
        np.mean(
            [
                r["jaccard_gain"] > 0
                for r in all_rows
            ]
        )
    )

    global_summary[
        "fraction_jaccard_gain_gt_0_25"
    ] = float(
        np.mean(
            [
                r["jaccard_gain"] > 0.25
                for r in all_rows
            ]
        )
    )

    global_summary[
        "fraction_jaccard_gain_gt_0_50"
    ] = float(
        np.mean(
            [
                r["jaccard_gain"] > 0.50
                for r in all_rows
            ]
        )
    )

    print()
    print("=" * 88)
    print("GLOBAL SUMMARY")
    print("=" * 88)

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
                    per_file,
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

        for row in all_rows:

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
