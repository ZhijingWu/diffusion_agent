# Experiment index

This file maps the questions in `REPRODUCTION.md` to the scripts and committed outputs used in the audit.

## E01 — Proxy topology sensitivity

**Question.** Does the reproduced Proxy assign meaningfully different scores to structurally different topologies for the same GSM8K question?

**Script**

```text
experiments/audits/core/audit_proxy_topology_sensitivity.py
```

**Output**

```text
audit_results/proxy_topology_sensitivity.json
```

20 GSM8K questions × 8 topology patterns. Mean composite reward range: \(2.74\times10^{-8}\).

## E02 — Guided-candidate near ties

**Question.** Does Proxy-guided candidate selection produce reward separation above numerical near-tie scale?

**Scripts**

```text
experiments/audits/core/audit_guidance_effect.py
experiments/audits/core/analyze_guidance_ties.py
```

**Output**

```text
audit_results/guidance_effect.json
```

Across 100 guided runs, the Proxy-selected candidate equals the first candidate in 89 cases. At tolerance \(10^{-6}\), all 100 runs are tied within tolerance.

## E03 — Raw versus executed topology

**Question.** How closely does the stored/generated adjacency match the graph executed by the released runtime?

**Script**

```text
experiments/audits/core/audit_topology_semantic_fidelity.py
```

**Output**

```text
audit_results/topology_semantic_fidelity/summary.json
```

Across 603 saved records:

- raw-to-mapped Jaccard: 0.164
- mapped-to-executed Jaccard: 0.837
- raw-to-executed Jaccard: 0.152

## E04 — Coordinate mechanism

**Question.** What source-level mechanism explains the low raw-to-executed fidelity?

**Source paths**

```text
GDesigner/graph/graph.py
evidence/source_trace.md
```

A concrete Phase-1 example is retained in:

```text
evidence/key_cases/case_phase1_coordinate_mismatch.json
```

## E05 — Canonical coordinate repair

**Question.** If only the coordinate encoding is aligned to the runtime's edge order, does topology fidelity recover?

**Scripts**

```text
experiments/audits/core/canonical_runtime_adapter.py
experiments/audits/core/audit_canonical_runtime_repair.py
experiments/audits/core/run_gsm8k_canonical_repair.py
```

**Output**

```text
audit_results/canonical_runtime_repair/summary.json
```

Raw-to-executed Jaccard increases from 0.152 to 0.863; the pre-cycle mapping is exact on 603/603 records.

## E06 — Phase-1 label provenance

**Question.** What graph is stored, and where do the Phase-1 utility and cost labels come from?

**Script**

```text
experiments/audits/core/audit_phase1_label_provenance.py
```

**Source**

```text
experiments/run_gsm8k.py
```

**Output**

```text
audit_results/phase1_label_provenance/summary.json
```

Cost equals stored raw edge count in 300/300 records; utility is binary in 300/300 records; the same stored raw graph is reused for Proxy and diffusion supervision.

## E07 — Training graph versus execution semantics

**Question.** How close are the six Phase-1 training topologies to the released and canonical execution semantics?

**Script**

```text
experiments/audits/core/audit_training_execution_semantics.py
```

**Output**

```text
audit_results/training_execution_semantics/summary.json
```

Observed means:

- raw-to-released-execution Jaccard: 0.278
- raw-to-canonical-execution Jaccard: 0.683
- cost vs. raw edge count MAE: 0
- cost vs. released executed-edge count MAE: 2.333

## E08 — Proxy-selected versus first-candidate downstream runs

**Question.** Does the numerically near-tied Proxy guidance produce a stable downstream improvement over choosing the first candidate?

**Supporting script**

```text
experiments/audits/supporting/compare_proxy_vs_first_downstream.py
```

The three matched pairs for seeds `20261002`, `20262000`, and `20263000` are under:

```text
audit_results/downstream_matched_control/
```

The direction of the accuracy difference changes across the three rounds. I therefore treat this comparison as inconclusive.

## E09 — Released versus canonical runtime under all-agent final aggregation

**Question.** Does coordinate repair change final correctness under the released final-aggregation architecture?

**Scripts**

```text
experiments/audits/core/run_gsm8k_matched_control.py
experiments/audits/core/run_gsm8k_canonical_repair.py
```

The matched seed `20266000` files are under:

```text
audit_results/downstream_matched_control/
```

Paired summary:

```text
evidence/paired_downstream_summary.csv
```

Observed result:

- accuracy: 0.96 vs. 0.96
- generated topology same: 49/50
- executed topology same: 0/50
- correctness outcome same: 50/50

## E10 — Sink-only final-decision ablation

**Question.** Does restricting final aggregation to terminal spatial nodes expose more outcome sensitivity to intermediate topology?

**Scripts**

```text
experiments/audits/core/decision_policy_patch.py
experiments/audits/core/run_gsm8k_sink_original.py
experiments/audits/core/run_gsm8k_sink_canonical.py
```

The matched seed `20267000` files are under:

```text
audit_results/downstream_matched_control/
```

Observed result:

- accuracy: 0.96 vs. 0.92
- paired correctness disagreements: 2/50
- exact two-sided McNemar \(p=0.5\)

This experiment is exploratory.

Discordant examples:

```text
evidence/key_cases/case_sink_disagreement_index7.json
evidence/key_cases/case_sink_disagreement_index24.json
```

## E11 — Candidate-budget hypothesis

**Question.** Does the runtime mapping substantially collapse the effective \(K=5\) candidate set?

**Scripts**

```text
experiments/audits/archive/audit_effective_search_budget.py
experiments/audits/archive/audit_execution_semantic_search_budget.py
```

The effective executed-candidate ratio is approximately 0.996 in the audit used to test this hypothesis. I keep this as a negative result rather than as part of the main explanation.
