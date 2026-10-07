# GTD GSM8K Reproduction and Execution-Semantics Audit

This branch records my reproduction of the released GTD GSM8K pipeline and a follow-up audit of how generated communication topologies are represented and executed.

The audit started from a simple question: when the diffusion model produces an \(N\times N\) adjacency matrix, does the released multi-agent runtime execute the same directed graph? I traced that matrix through Phase-1 data generation, Proxy and diffusion supervision, runtime graph construction, and final decision aggregation, and then added small controlled interventions to separate the effects of these stages.

The results below refer to the released code path reproduced in this branch. I do not assume that they extend to other benchmarks, checkpoints, or repository revisions without separate verification.

## Technical note

A concise two-page summary of the reproduction and execution-semantics audit is available here:

[GTD Reproduction Technical Note](docs/GTD_Reproduction_Technical_Note_Zhijing_Wu.pdf)

## Reproduction scope

The main reproduction uses GSM8K with four homogeneous `MathSolver` agents.

Audit scripts are grouped under:

```text
experiments/audits/core/
experiments/audits/supporting/
experiments/audits/archive/
```

Selected outputs are kept under:

```text
audit_results/
evidence/
```

The main source paths inspected in the audit are:

```text
experiments/run_gsm8k.py
GDesigner/graph/graph.py
```

## Runtime coordinate mismatch

The released `Graph` constructor accepts `fixed_spatial_masks` as an \(N\times N\) object and immediately flattens it. Later in the class, the effective `init_potential_edges()` constructs only non-self directed edges. `construct_spatial_connection()` then iterates with:

```python
zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks)
```

For four agents, the flattened mask has 16 entries while the potential spatial-edge list has 12 entries.

I checked the resulting coordinate map with a one-hot probe and then measured the effect on saved runs. Across 603 records:

| comparison | mean edge Jaccard |
|---|---:|
| raw adjacency vs. runtime-mapped adjacency | 0.164 |
| runtime-mapped vs. executed adjacency | 0.837 |
| raw adjacency vs. executed adjacency | 0.152 |

The raw and executed graphs were never exactly equal in these 603 records.

This decomposition matters because the larger loss appears before cycle filtering. The dominant discrepancy in these runs comes from how flattened mask coordinates are paired with the non-self edge list, rather than from the later acyclicity check.

Summary:

```text
audit_results/topology_semantic_fidelity/summary.json
```

## Canonical coordinate repair

To isolate the coordinate effect, I added a small adapter that takes the intended non-self adjacency and places those edge indicators in the order consumed by the current runtime.

The intervention does not change the diffusion checkpoint, Proxy checkpoint, candidate set, cycle rule, agent model, or task set.

Across the same 603 records:

| metric | released runtime | canonical coordinate adapter |
|---|---:|---:|
| raw-to-executed Jaccard | 0.152 | 0.863 |
| precision | 0.280 | 1.000 |
| recall | 0.240 | 0.863 |
| F1 | 0.235 | 0.918 |
| reachability Jaccard | 0.243 | 0.823 |

Before cycle filtering, the repaired coordinate mapping is exact on 603/603 records.

Relevant files:

```text
experiments/audits/core/canonical_runtime_adapter.py
experiments/audits/core/audit_canonical_runtime_repair.py
audit_results/canonical_runtime_repair/summary.json
```

I use this as a mechanism check rather than as evidence that task accuracy should necessarily improve. The learned checkpoints and final decision architecture may still make downstream behavior insensitive to this intervention.

## Phase-1 supervision semantics

I traced how the Phase-1 graph-performance pairs are constructed in `experiments/run_gsm8k.py`.

For each static topology, the code:

1. passes `topology_matrix` to `Graph(..., fixed_spatial_masks=topology_matrix)`;
2. executes the multi-agent graph with `arun(...)`;
3. defines `utility` from final-answer correctness;
4. defines `cost` as `sum(sum(row) for row in topology_matrix)`;
5. stores the same raw topology matrix;
6. later reuses that stored raw graph for Proxy graph construction and as a diffusion \(A_0\) target.

In the reproduced 300-record Phase-1 dataset:

- `performance.cost` equals the number of ones in the stored raw adjacency in 300/300 records;
- utility is binary in 300/300 records;
- the utility distribution is 282 successful and 18 unsuccessful examples;
- the dataset contains six distinct static raw topologies.

A compact description is:

\[
(A,C)\mapsto [U(T(A),C),\operatorname{Cost}(A)],
\]

where \(T(A)\) denotes the graph obtained after the runtime interprets the stored matrix \(A\).

This is a provenance distinction, not a claim that the Proxy cannot learn the mapping. Since the runtime transformation is deterministic, \(A\mapsto U(T(A))\) remains learnable in principle. The narrower point is that the stored graph, executed graph, and edge-count cost do not all describe an unchanged topology object.

Relevant summaries:

```text
audit_results/phase1_label_provenance/summary.json
audit_results/training_execution_semantics/summary.json
evidence/source_trace.md
```

## Proxy sensitivity in the reproduced homogeneous setting

A separate audit probes the trained Proxy with deliberately different topologies while keeping the GSM8K question fixed.

Across 20 questions and eight topology patterns per question:

- mean utility range across topologies: \(1.79\times10^{-8}\);
- mean cost range: \(9.54\times10^{-8}\);
- mean composite reward range: \(2.74\times10^{-8}\);
- maximum composite reward range: \(1.67\times10^{-7}\).

In 100 guided runs, the Proxy-selected candidate was the first candidate in 89 cases. With a tolerance of \(10^{-6}\), all 100 runs were tied within tolerance.

This observation is specific to the reproduced checkpoint and the homogeneous four-`MathSolver` GSM8K setting. I do not use it to make a claim about heterogeneous roles or other checkpoints.

Relevant files:

```text
experiments/audits/core/audit_proxy_topology_sensitivity.py
experiments/audits/core/audit_guidance_effect.py
experiments/audits/core/analyze_guidance_ties.py
audit_results/proxy_topology_sensitivity.json
audit_results/guidance_effect.json
```

A separate candidate-budget audit found that candidate diversity itself does not materially collapse after runtime projection; that negative result remains under `experiments/audits/archive/`.

## Downstream checks

### Original final aggregation

I ran a matched 50-question comparison between the released execution path and the canonical coordinate adapter.

For seed `20266000`:

- released-runtime accuracy: 0.96;
- canonical-runtime accuracy: 0.96;
- generated topology matched in 49/50 cases;
- executed topology matched in 0/50 cases;
- correctness outcome matched in 50/50 cases.

Thus, this intervention changes the executed communication graph substantially while leaving task correctness unchanged on this sample.

The corresponding result files are retained under:

```text
audit_results/downstream_matched_control/
```

and the paired summary is in:

```text
evidence/paired_downstream_summary.csv
```

### Sink-only exploratory ablation

After the agent rounds, the released runtime calls `connect_decision_node()`, which connects every agent to the final decision node. I therefore tested a sink-only variant in which only terminal spatial nodes feed that final node.

For seed `20267000`:

- released-coordinate accuracy: 0.96;
- canonical-coordinate accuracy: 0.92;
- the two arms disagree on correctness in 2/50 examples;
- both disagreements favor the released-coordinate arm;
- exact two-sided McNemar \(p=0.5\).

This is an exploratory result. Two discordant examples are not enough to establish a causal topology-bypass effect. I keep it because it identifies a concrete follow-up question about how final aggregation interacts with intermediate communication structure.

Relevant files:

```text
experiments/audits/core/decision_policy_patch.py
experiments/audits/core/run_gsm8k_sink_original.py
experiments/audits/core/run_gsm8k_sink_canonical.py
evidence/key_cases/case_sink_disagreement_index7.json
evidence/key_cases/case_sink_disagreement_index24.json
```

## Reproducing the audits

The core audit directory contains the scripts used for the main claims in this note. The supporting directory contains secondary analyses, and the archive directory contains a tested hypothesis that was not supported.

A file-by-file map is available in:

```text
evidence/experiment_index.md
```

Compact machine-readable summaries are:

```text
evidence/key_metrics.json
evidence/evidence_table.csv
evidence/paired_downstream_summary.csv
```

## Scope and limitations

The source and audit evidence in this branch supports four narrow conclusions:

- the audited released GSM8K runtime interprets the stored adjacency coordinates differently from the corresponding \(A_{ij}\) coordinates;
- aligning only those coordinates restores most raw-to-executed topology fidelity in the 603-record audit;
- Phase-1 edge-count cost is computed on the stored raw adjacency, while utility is observed after runtime execution;
- in one matched 50-question sample, a large change in executed topology does not change final correctness under the original all-agent final aggregation.

The sink-only result is exploratory. The Proxy sensitivity result is limited to the reproduced homogeneous-agent checkpoint.

I do not use these experiments to claim that the paper's reported benchmark results are invalid, that the same behavior occurs on every benchmark or checkpoint, or that these observations are novel relative to all external discussion.
