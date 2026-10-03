#!/usr/bin/env python3
"""
Final provenance audit for GTD Phase-1 GSM8K labels.

Goal
----
Trace, from released source + generated dataset, whether:

1) raw topology A is passed into Graph as fixed_spatial_masks;
2) utility is derived from actual MAS correctness;
3) cost is derived from the raw graph edge count;
4) the persisted Phase-1 record stores:
       graph = A
       performance.utility = solved(A executed by runtime)
       performance.cost = cost(A_raw)

This script does NOT modify any source files and does NOT call any API.

Usage
-----
PYTHONPATH=. python experiments/audits/audit_phase1_label_provenance.py \
    --runner experiments/run_gsm8k.py \
    --graph-source GDesigner/graph/graph.py \
    --dataset gtd_gsm8k_dataset_training.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def numbered_lines(text: str):
    return list(enumerate(text.splitlines(), start=1))


def print_context(title: str, text: str, pattern: str, before: int = 6, after: int = 10):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)

    lines = text.splitlines()
    rx = re.compile(pattern)

    hits = []
    for i, line in enumerate(lines):
        if rx.search(line):
            hits.append(i)

    if not hits:
        print(f"[NOT FOUND] pattern: {pattern}")
        return []

    shown_ranges = []
    for hit in hits:
        lo = max(0, hit - before)
        hi = min(len(lines), hit + after + 1)

        # avoid duplicate overlapping blocks
        if shown_ranges and lo <= shown_ranges[-1][1]:
            continue

        shown_ranges.append((lo, hi))
        print(f"\n--- around line {hit + 1} ---")
        for j in range(lo, hi):
            marker = ">>" if j == hit else "  "
            print(f"{marker} {j+1:5d}: {lines[j]}")

    return [x + 1 for x in hits]


def find_assignments(text: str):
    """
    AST-based inventory of assignments/calls involving key provenance names.
    This does not attempt whole-program dataflow; it gives exact source locations.
    """
    tree = ast.parse(text)

    records = []

    key_names = {
        "utility",
        "cost",
        "is_solved",
        "predict_answer",
        "adj_matrix",
        "topology_matrix",
        "fixed_spatial_masks",
        "performance",
        "generated_data",
        "dataset_training",
    }

    def src(node):
        try:
            return ast.unparse(node)
        except Exception:
            return node.__class__.__name__

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [src(t) for t in node.targets]
            joined = " ".join(targets)
            value = src(node.value)

            if any(k in joined or k in value for k in key_names):
                records.append(
                    (getattr(node, "lineno", -1), "ASSIGN", ", ".join(targets), value)
                )

        elif isinstance(node, ast.AnnAssign):
            target = src(node.target)
            value = src(node.value) if node.value is not None else ""

            if any(k in target or k in value for k in key_names):
                records.append(
                    (getattr(node, "lineno", -1), "ANNASSIGN", target, value)
                )

        elif isinstance(node, ast.Call):
            call = src(node)
            if (
                "Graph(" in call
                or ".arun(" in call
                or ".append(" in call and ("performance" in call or "graph" in call)
            ):
                records.append(
                    (getattr(node, "lineno", -1), "CALL", "", call)
                )

    return sorted(records)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}") from e
    return rows


def graph_edge_count(graph):
    return sum(int(v) for row in graph for v in row)


def nonself_edge_count(graph):
    n = len(graph)
    return sum(
        int(graph[i][j])
        for i in range(n)
        for j in range(n)
        if i != j
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="experiments/run_gsm8k.py")
    ap.add_argument("--graph-source", default="GDesigner/graph/graph.py")
    ap.add_argument("--dataset", default="gtd_gsm8k_dataset_training.jsonl")
    ap.add_argument(
        "--output",
        default="audit_results/phase1_label_provenance/summary.json",
    )
    args = ap.parse_args()

    runner_path = Path(args.runner)
    graph_path = Path(args.graph_source)
    dataset_path = Path(args.dataset)

    runner_text = read_text(runner_path)
    graph_text = read_text(graph_path)

    print("=" * 96)
    print("GTD PHASE-1 LABEL PROVENANCE AUDIT")
    print("=" * 96)
    print("runner:", runner_path)
    print("graph source:", graph_path)
    print("dataset:", dataset_path)

    # -----------------------------------------------------------------
    # A. Source evidence: raw graph -> Graph(fixed_spatial_masks=...)
    # -----------------------------------------------------------------

    graph_ctor_hits = print_context(
        "A. RAW TOPOLOGY -> Graph(fixed_spatial_masks=...)",
        runner_text,
        r"fixed_spatial_masks",
        before=10,
        after=12,
    )

    # -----------------------------------------------------------------
    # B. Source evidence: actual MAS execution
    # -----------------------------------------------------------------

    arun_hits = print_context(
        "B. MAS EXECUTION -> graph.arun(...)",
        runner_text,
        r"\.arun\(",
        before=10,
        after=12,
    )

    # -----------------------------------------------------------------
    # C. Source evidence: predict_answer / is_solved / utility
    # -----------------------------------------------------------------

    solved_hits = print_context(
        "C1. RAW ANSWER -> predict_answer / is_solved",
        runner_text,
        r"is_solved\s*=",
        before=10,
        after=14,
    )

    utility_hits = print_context(
        "C2. CORRECTNESS -> utility",
        runner_text,
        r"utility\s*=",
        before=10,
        after=14,
    )

    # -----------------------------------------------------------------
    # D. Source evidence: cost construction
    # -----------------------------------------------------------------

    cost_hits = print_context(
        "D. COST CONSTRUCTION",
        runner_text,
        r"cost\s*=",
        before=12,
        after=16,
    )

    # -----------------------------------------------------------------
    # E. Source evidence: persistence of graph/performance
    # -----------------------------------------------------------------

    performance_hits = print_context(
        "E1. STORED performance FIELD",
        runner_text,
        r"[\"']performance[\"']",
        before=14,
        after=18,
    )

    graph_field_hits = print_context(
        "E2. STORED graph FIELD",
        runner_text,
        r"[\"']graph[\"']",
        before=12,
        after=16,
    )

    # -----------------------------------------------------------------
    # F. Runtime source evidence
    # -----------------------------------------------------------------

    init_mask_hits = print_context(
        "F1. Graph: fixed_spatial_masks flattening",
        graph_text,
        r"fixed_spatial_masks\s*=\s*torch\.tensor",
        before=8,
        after=12,
    )

    potential_hits = print_context(
        "F2. Graph: potential spatial edges",
        graph_text,
        r"def init_potential_edges",
        before=4,
        after=32,
    )

    zip_hits = print_context(
        "F3. Graph: runtime zip(potential_spatial_edges, ..., spatial_masks)",
        graph_text,
        r"zip\(self\.potential_spatial_edges",
        before=8,
        after=18,
    )

    decision_hits = print_context(
        "F4. Graph: decision-node aggregation",
        graph_text,
        r"def connect_decision_node",
        before=4,
        after=14,
    )

    # -----------------------------------------------------------------
    # G. AST inventory
    # -----------------------------------------------------------------

    print("\n" + "=" * 96)
    print("G. AST PROVENANCE INVENTORY")
    print("=" * 96)

    ast_rows = find_assignments(runner_text)
    for lineno, kind, lhs, rhs in ast_rows:
        print(f"{lineno:5d} | {kind:9s} | {lhs} <- {rhs[:220]}")

    # -----------------------------------------------------------------
    # H. Dataset-level exact checks
    # -----------------------------------------------------------------

    rows = load_jsonl(dataset_path)

    if not rows:
        raise RuntimeError("Dataset is empty.")

    cost_equals_all_edges = 0
    cost_equals_nonself_edges = 0
    utility_binary = 0
    valid_rows = 0

    utility_counts = Counter()
    cost_counts = Counter()
    raw_topology_counts = Counter()

    mismatches = []

    for idx, row in enumerate(rows):
        graph = row.get("graph")
        perf = row.get("performance", {})
        utility = perf.get("utility")
        cost = perf.get("cost")

        if graph is None or utility is None or cost is None:
            continue

        valid_rows += 1

        all_edges = graph_edge_count(graph)
        nonself_edges = nonself_edge_count(graph)

        if float(cost) == float(all_edges):
            cost_equals_all_edges += 1

        if float(cost) == float(nonself_edges):
            cost_equals_nonself_edges += 1

        if utility in (0, 0.0, 1, 1.0, False, True):
            utility_binary += 1

        utility_counts[float(utility)] += 1
        cost_counts[float(cost)] += 1

        key = tuple(tuple(int(x) for x in r) for r in graph)
        raw_topology_counts[key] += 1

        if float(cost) != float(all_edges):
            mismatches.append(
                {
                    "index": idx,
                    "cost": cost,
                    "all_edges": all_edges,
                    "nonself_edges": nonself_edges,
                }
            )

    print("\n" + "=" * 96)
    print("H. DATASET-LEVEL EXACT CHECKS")
    print("=" * 96)
    print("records:", len(rows))
    print("valid graph/performance records:", valid_rows)
    print(
        "cost == total raw adjacency ones:",
        f"{cost_equals_all_edges}/{valid_rows}",
        f"({cost_equals_all_edges / valid_rows:.1%})" if valid_rows else "",
    )
    print(
        "cost == raw non-self edge count:",
        f"{cost_equals_nonself_edges}/{valid_rows}",
        f"({cost_equals_nonself_edges / valid_rows:.1%})" if valid_rows else "",
    )
    print(
        "utility is binary:",
        f"{utility_binary}/{valid_rows}",
        f"({utility_binary / valid_rows:.1%})" if valid_rows else "",
    )
    print("utility distribution:", dict(utility_counts))
    print("cost distribution:", dict(cost_counts))
    print("unique raw topologies:", len(raw_topology_counts))

    if mismatches:
        print("\nFirst cost mismatches:")
        for x in mismatches[:10]:
            print(x)

    # -----------------------------------------------------------------
    # I. Automated source-level verdict
    # -----------------------------------------------------------------

    source_presence = {
        "fixed_spatial_masks_found": bool(graph_ctor_hits),
        "arun_found": bool(arun_hits),
        "is_solved_found": bool(solved_hits),
        "utility_assignment_found": bool(utility_hits),
        "cost_assignment_found": bool(cost_hits),
        "performance_storage_found": bool(performance_hits),
        "graph_storage_found": bool(graph_field_hits),
        "runtime_mask_flatten_found": bool(init_mask_hits),
        "runtime_zip_found": bool(zip_hits),
        "decision_node_found": bool(decision_hits),
    }

    dataset_checks = {
        "records": len(rows),
        "valid_records": valid_rows,
        "cost_equals_total_raw_ones_count": cost_equals_all_edges,
        "cost_equals_total_raw_ones_rate": (
            cost_equals_all_edges / valid_rows if valid_rows else None
        ),
        "cost_equals_nonself_count": cost_equals_nonself_edges,
        "cost_equals_nonself_rate": (
            cost_equals_nonself_edges / valid_rows if valid_rows else None
        ),
        "utility_binary_count": utility_binary,
        "utility_binary_rate": utility_binary / valid_rows if valid_rows else None,
        "utility_distribution": dict(utility_counts),
        "cost_distribution": dict(cost_counts),
        "unique_raw_topologies": len(raw_topology_counts),
    }

    print("\n" + "=" * 96)
    print("I. AUTOMATIC PROVENANCE CHECK")
    print("=" * 96)

    for k, v in source_presence.items():
        print(f"{k:36s}: {v}")

    print()

    required_source = [
        source_presence["fixed_spatial_masks_found"],
        source_presence["arun_found"],
        source_presence["is_solved_found"],
        source_presence["utility_assignment_found"],
        source_presence["cost_assignment_found"],
        source_presence["performance_storage_found"],
        source_presence["graph_storage_found"],
    ]

    if all(required_source):
        print("SOURCE CHAIN: all expected provenance landmarks were found.")
    else:
        print("SOURCE CHAIN: one or more landmarks were not found; inspect contexts above.")

    if valid_rows and cost_equals_all_edges == valid_rows:
        print(
            "DATA CHECK: performance.cost equals the number of 1s in the stored raw adjacency "
            "for every valid Phase-1 sample."
        )
    else:
        print(
            "DATA CHECK: performance.cost is NOT exactly equal to raw adjacency edge count "
            "for every sample."
        )

    if valid_rows and utility_binary == valid_rows:
        print(
            "DATA CHECK: performance.utility is binary for every valid Phase-1 sample."
        )

    print(
        "\nIMPORTANT: the script intentionally does not infer semantic provenance from variable "
        "names alone. Use the printed source contexts to confirm the exact RHS of utility/cost "
        "and the dict written to the JSONL. That final human-visible trace is the evidence to "
        "quote in the research note."
    )

    # -----------------------------------------------------------------
    # Save machine-readable summary
    # -----------------------------------------------------------------

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "runner": str(runner_path),
        "graph_source": str(graph_path),
        "dataset": str(dataset_path),
        "source_presence": source_presence,
        "dataset_checks": dataset_checks,
        "ast_inventory": [
            {
                "line": line,
                "kind": kind,
                "lhs": lhs,
                "rhs": rhs,
            }
            for line, kind, lhs, rhs in ast_rows
        ],
    }

    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
