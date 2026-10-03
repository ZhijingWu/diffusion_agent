# Experiment Index

This index maps each research question to the script, artifact, result, and final status.

## E01 — Proxy topology sensitivity

**Question:** Does the trained Proxy distinguish structurally different topologies for the same task?

- Script: `experiments/audits/core/audit_proxy_topology_sensitivity.py`
- Result: `results/core/proxy_topology_sensitivity.json`
- Design: 20 GSM8K questions × 8 topology patterns
- Key metric: mean composite reward range = \(2.74\times10^{-8}\)
- Status: **CONFIRMED for this reproduced checkpoint**
- Use in author note: supporting mechanism evidence

## E02 — ZO near-tie audit

**Question:** Does Proxy-guided candidate selection produce meaningful reward discrimination?

- Scripts:
  - `experiments/audits/core/audit_guidance_effect.py`
  - `experiments/audits/core/analyze_guidance_ties.py`
- Result: `results/core/guidance_effect.json`
- Design: 100 guided runs
- Key result:
  - Proxy == first candidate: 89/100
  - at epsilon \(10^{-6}\), 0/100 runs show meaningful discrimination
- Status: **CONFIRMED for this reproduced checkpoint**

## E03 — Raw vs executed topology

**Question:** Is the generated adjacency actually the graph executed by the MAS runtime?

- Script: `experiments/audits/core/audit_topology_semantic_fidelity.py`
- Result: `results/core/topology_semantic_fidelity/`
- Design: 603 saved runs
- Key result:
  - raw→executed Jaccard = 0.152
  - raw→mapped Jaccard = 0.164
  - mapped→executed Jaccard = 0.837
- Status: **CONFIRMED**

## E04 — Runtime coordinate probe

**Question:** Why is raw→executed fidelity low?

- Evidence: exhaustive one-hot position probe performed during the reproduction
- Mechanism: flattened \(N^2\) mask is zipped with the effective non-self \(N(N-1)\) potential-edge list
- Supporting source: `evidence/source_trace.md`
- Status: **CONFIRMED**
- Note: the exact probe output was observed interactively; the deterministic mechanism is preserved in the source and repair audits.

## E05 — Canonical runtime repair

**Question:** If only the edge coordinate system is aligned, does topology fidelity recover?

- Scripts:
  - `experiments/audits/core/canonical_runtime_adapter.py`
  - `experiments/audits/core/audit_canonical_runtime_repair.py`
- Result: `results/core/canonical_runtime_repair/`
- Design: 603 saved runs
- Key result:
  - Jaccard 0.152 → 0.863
  - repaired mapping exact = 1.0
- Status: **CONFIRMED**

## E06 — Phase-1 label provenance

**Question:** What exactly are the training graph, utility, and cost labels?

- Script: `experiments/audits/core/audit_phase1_label_provenance.py`
- Result: `results/core/phase1_label_provenance/summary.json`
- Design: source trace + 300 records
- Key result:
  - cost == raw adjacency edge count: 300/300
  - utility is binary correctness: 300/300
  - same raw graph is reused for Proxy and diffusion supervision
- Status: **SOURCE-LEVEL CONFIRMED**

## E07 — Training vs execution semantics

**Question:** Are Phase-1 graph labels closer to canonical or old runtime execution semantics?

- Script: `experiments/audits/core/audit_training_execution_semantics.py`
- Result: `results/core/training_execution_semantics/summary.json`
- Key result:
  - raw→old execution Jaccard = 0.278
  - raw→canonical execution Jaccard = 0.683
  - cost vs raw edges MAE = 0
  - cost vs old executed edges MAE = 2.333
- Status: **CONFIRMED descriptive finding**

## E08 — Proxy vs first-candidate downstream rounds

**Question:** Does the near-tied Proxy guidance produce consistent downstream gains?

- Supporting script: `experiments/audits/supporting/compare_proxy_vs_first_downstream.py`
- Results: `results/supporting/proxy_vs_first_rounds/`
- Three 100-question matched rounds:
  - round 1: 0.91 vs 0.95, McNemar p=0.125
  - round 2: 0.91 vs 0.92, p=1.0
  - round 3: 0.92 vs 0.90, p=0.625
- Status: **INCONCLUSIVE**
- Use: supporting context only; do not headline.

## E09 — All-agent FinalRefer original vs canonical

**Question:** Does repairing execution semantics change downstream correctness under the original final aggregation architecture?

- Core results: `results/core/downstream_matched/` (seed 20266000)
- Design: 50 matched questions
- Key result:
  - accuracy 0.96 vs 0.96
  - generated topology same 49/50
  - executed topology same 0/50
  - correctness same 50/50
- Status: **MODERATE evidence of downstream invariance on this sample**

## E10 — Sink-only FinalRefer ablation

**Question:** Does removing all-agent final aggregation increase sensitivity to communication topology?

- Scripts:
  - `experiments/audits/core/decision_policy_patch.py`
  - `experiments/audits/core/run_gsm8k_sink_original.py`
  - `experiments/audits/core/run_gsm8k_sink_canonical.py`
- Core results: `results/core/downstream_matched/` (seed 20267000)
- Key result:
  - accuracy 0.96 vs 0.92
  - 2/50 paired outcome disagreements, both Original-only correct
  - McNemar p=0.5
- Status: **EXPLORATORY**
- Use: author question / follow-up direction, not a confirmed claim.

## E11 — Effective search-budget audit

**Question:** Does the runtime mapping collapse K=5 candidate diversity?

- Scripts in `experiments/audits/archive/`
- Result: `results/archive/execution_semantic_search_budget/summary.json`
- Key result: effective search ratio ≈ 0.996
- Status: **NEGATIVE RESULT / HYPOTHESIS RULED OUT**
- Reason archived: useful provenance, not part of the final causal story.
