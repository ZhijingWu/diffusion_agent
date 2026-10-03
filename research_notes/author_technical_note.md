# Technical Note for GTD Authors — Draft

## Subject

Execution-semantics question in the released GTD GSM8K pipeline

## Short version

While reproducing GTD on GSM8K, we traced the generated adjacency matrix through Phase-1 supervision, Proxy/diffusion training, and the released GDesigner execution path. We found a reproducible representation–execution mismatch: `fixed_spatial_masks` is flattened as an \(N^2\) mask, while the effective potential spatial-edge list contains only \(N(N-1)\) non-self edges and is consumed via `zip`.

For four agents, an exhaustive one-hot probe showed that the 16 raw matrix positions are not interpreted as the corresponding \(A_{ij}\) edges. Across 603 saved runs, the mean generated-vs-executed edge Jaccard was 0.152. A minimal canonical coordinate adapter, leaving the diffusion model, Proxy, candidate sampling, and cycle filtering unchanged, increased this to 0.863; the pre-cycle mapping became exact on all 603 records.

We also traced Phase-1 labels directly from source. The raw `topology_matrix` is passed into the MAS, `utility` is the correctness of the resulting MAS answer, while `cost` is computed as `sum(sum(row) for row in topology_matrix)`. In the generated 300-record training set, `performance.cost` equals raw edge count in 300/300 records. The same stored raw graph is later used for Proxy graph construction and as the diffusion \(A_0\) target.

One additional observation is that after canonical repair, a matched 50-question run changed the executed topology on all 50 questions while correctness stayed identical (0.96 vs 0.96). Since the runtime connects every agent to the final decision node after the agent rounds, we tested a sink-only decision-node ablation. This produced two paired correctness disagreements (0.96 vs 0.92), but the sample is too small for a strong conclusion (exact McNemar p=0.5).

## Questions for clarification

1. Is the intended semantics of the generated \(N\times N\) matrix exactly \(A_{ij}=1\Rightarrow i\to j\) at execution time, or is an implicit mask-to-edge remapping intended?
2. Is Phase-1 `cost` intended to be raw graph edge count, or should it reflect the actually executed communication structure / token consumption?
3. In the intended evaluation, should the final decision node always aggregate outputs from all agents independently of the generated spatial topology?
4. If useful, we can share the minimal reproducer, the canonical adapter, and the per-record audit outputs.

## Evidence summary

| Check | Result |
|---|---:|
| Raw→executed Jaccard, 603 records | 0.152 |
| Canonical-repaired Jaccard | 0.863 |
| Repaired pre-cycle mapping exact | 603/603 |
| Phase-1 cost == raw edge count | 300/300 |
| Proxy mean reward range across 8 topologies | \(2.74\times10^{-8}\) |
| All-agent matched accuracy | 0.96 vs 0.96 |
| Sink-only matched accuracy | 0.96 vs 0.92 (exploratory) |

## Framing

We are treating these as reproduction findings and implementation/semantics questions, not as evidence that the paper's headline benchmark results are invalid. We would especially appreciate clarification on whether our interpretation of the intended topology coordinate system matches the authors' design.
