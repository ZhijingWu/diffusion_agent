# Finding 01 — Proxy / ZO Guidance

## Question

Does the reproduced Proxy meaningfully rank different topology candidates, as required by the ZO guidance mechanism?

## Evidence

### Topology sensitivity probe

20 questions × 8 deliberately different topologies:

- max node-feature difference: **0.0e+00**
- mean utility range: **1.788e-08**
- mean cost range: **9.537e-08**
- mean composite reward range: **2.742e-08**
- max composite reward range: **1.669e-07**

### Guided generation audit

100 runs:

- Proxy == first candidate: **89/100**
- mean reward spread: **2.536e-08**
- max reward spread: **2.086e-07**

Tie analysis:

- at epsilon \(10^{-6}\): **100/100** runs were entirely tied within tolerance;
- **0/100** had meaningful discrimination above that threshold.

## Interpretation

This does not mean candidate generation collapses. The separate search-budget audit found \(K_{exec}/K\approx0.996\). Candidate diversity remains; the weak point is the reward signal used to rank candidates.

## Confidence

**Strong for this checkpoint and homogeneous 4×MathSolver GSM8K setting.**

Do not generalize this finding to heterogeneous roles, other checkpoints, or other benchmarks without testing.
