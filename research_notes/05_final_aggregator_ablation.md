# Finding 05 — Final Aggregator / Topology-Bypass Ablation

## Motivation

After canonical repair, the actually executed topology changed dramatically but matched 50-question GSM8K correctness did not.

Source inspection shows that after agent rounds the runtime calls `connect_decision_node()`, which connects every agent to the final decision node.

This motivates—but does not prove—the hypothesis that global final aggregation can reduce sensitivity to intermediate spatial topology.

## All-agent FinalRefer matched experiment

Seed: `20266000`, N=50.

| Metric | Original runtime | Canonical runtime |
|---|---:|---:|
| Accuracy | 0.96 | 0.96 |
| Avg tokens | 1624.04 | 1696.72 |
| Avg executed edges | 2.94 | 3.02 |
| raw→executed Jaccard | 0.184 | 0.868 |

Paired structure/outcomes:

- same generated topology: 49/50
- same executed topology: 0/50
- same correctness: 50/50
- mean executed-topology Jaccard between arms: 0.174

## Sink-only decision-node ablation

Seed: `20267000`, N=50.

| Metric | Original runtime | Canonical runtime |
|---|---:|---:|
| Accuracy | 0.96 | 0.92 |
| Avg tokens | 1466.40 | 1530.14 |
| Avg executed edges | 3.16 | 3.06 |
| raw→executed Jaccard | 0.179 | 0.820 |

Paired outcomes:

- same generated topology: 49/50
- same executed topology: 0/50
- same correctness: 48/50
- Original-correct / Canonical-wrong: 2
- Canonical-correct / Original-wrong: 0
- exact two-sided McNemar p = 0.5

## Interpretation

The sink-only result is **exploratory**, not statistically confirmatory. It shows that correctness disagreements appear once all-agent final aggregation is removed, but two discordant examples are insufficient to establish a causal bypass effect.

Use this as an author question / follow-up direction, not as a headline claim.

Concrete discordant cases are preserved in:

- `evidence/key_cases/case_sink_disagreement_index7.json`
- `evidence/key_cases/case_sink_disagreement_index24.json`
