# GTD Reproduction — Executive Summary

## Scope

This package documents a GSM8K reproduction and execution-semantics audit of the released GTD code path. The goal is **not** to claim that the paper is invalid. The goal is to identify where the mathematical object called a communication topology is represented, supervised, transformed, and executed, and to test whether those semantics remain aligned.

## Main findings

### 1. Proxy / ZO guidance is almost topology-insensitive in the reproduced homogeneous setting

Across 20 GSM8K questions and 8 deliberately different topologies per question, the Proxy's mean composite reward range was:

\[
2.742e-08
\]

and the maximum was:

\[
1.669e-07.
\]

In 100 guided runs, Proxy and first-candidate outputs matched in 89%. Candidate reward spread averaged \(2.54\times10^{-8}\). At tolerance \(10^{-6}\), no run contained a meaningfully discriminated candidate set.

**Interpretation:** candidate diversity exists, but the reproduced Proxy provides negligible ranking signal at this numerical scale.

### 2. The generated adjacency and released runtime edge coordinates are misaligned

The runtime flattens the \(N\times N\) `fixed_spatial_masks`, while the effective `init_potential_edges()` creates only non-self edges. For four agents, this is 16 mask positions versus 12 potential edges. The later `init_potential_edges` definition is the effective Python class method.

Across 603 saved runs:

- raw→mapped Jaccard: **0.164**
- mapped→executed Jaccard: **0.837**
- raw→executed Jaccard: **0.152**
- raw==executed: **0.0%**

The much lower raw→mapped score indicates that most structural distortion precedes cycle filtering.

### 3. A minimal canonical adapter repairs most topology-semantic distortion

Without changing diffusion, Proxy, candidate sampling, or cycle filtering:

\[
J_{raw,exec}:\; 0.152

ightarrow
0.863
\]

with mean gain **+0.711**.

The repaired mapping is exact before cycle filtering on **100.0%** of 603 records.

### 4. Phase-1 supervision mixes raw-graph and executed-graph semantics

Source-level provenance establishes:

1. the raw `topology_matrix` is passed into `Graph(... fixed_spatial_masks=topology_matrix)`;
2. the MAS is executed with `arun`;
3. `utility` is binary correctness of the MAS answer;
4. `cost = sum(sum(row) for row in topology_matrix)`;
5. the raw graph is stored and later reused directly for Proxy graph construction and diffusion \(A_0\) supervision.

Dataset check:

- `performance.cost == raw adjacency edge count`: **300/300**
- utility is binary: **300/300**

The training-semantic audit gives:

\[
J(raw, old\ execution)=0.278
\]

versus:

\[
J(raw, canonical\ execution)=0.683.
\]

A careful formulation is therefore:

> Phase-1 cost is defined directly on the stored raw graph, while utility is produced by MAS execution after runtime interpretation of that graph.

### 5. Correcting execution semantics changes the executed graph far more than task correctness

Matched 50-question all-agent FinalRefer experiment (seed 20266000):

- Original accuracy: **0.96**
- Canonical accuracy: **0.96**
- same generated topology: **49/50**
- same executed topology: **0/50**
- same correctness outcome: **50/50**
- mean executed-topology Jaccard between arms: **0.174**

So a strong intervention on execution topology did not change correctness in this sample.

### 6. Final aggregation is a plausible topology-bypass mechanism, but evidence is exploratory

The runtime connects **every agent** to the final decision node after agent rounds. We therefore ran a sink-only decision-node ablation.

Matched 50-question sink-only experiment (seed 20267000):

- Original accuracy: **0.96**
- Canonical accuracy: **0.92**
- same generated topology: **49/50**
- same executed topology: **0/50**
- correctness disagreements: **2/50**, both Original-correct / Canonical-wrong
- exact two-sided McNemar \(p=0.5\)

This is **not confirmatory statistical evidence**, but it is consistent with the hypothesis that all-agent final aggregation can reduce observable sensitivity to intermediate topology.

## Evidence strength

### Strong / source- or audit-confirmed

- Proxy reward variation is near numerical-tie scale for the reproduced checkpoint.
- Runtime coordinate mismatch exists in the audited released-code path.
- Canonical adapter restores mapping fidelity.
- Phase-1 cost equals raw edge count in all 300 records.
- Raw graph is reused for Proxy and diffusion supervision.

### Moderate

- Large changes in executed topology can coexist with unchanged correctness on the 50-question all-agent sample.

### Exploratory

- Sink-only aggregation increases topology sensitivity.
- FinalRefer is the causal reason for the all-agent invariance.

## What we should not claim

Do not claim that:
- the paper's reported accuracy is invalid;
- GTD “does not work”;
- the issue necessarily affects every benchmark or repository revision;
- this is globally the first discovery without a separate external novelty search.

## Suggested next action

For author outreach, lead with the representation/execution mismatch and canonical repair. Present Proxy insensitivity and final-aggregator bypass as follow-up observations/questions rather than accusations.
