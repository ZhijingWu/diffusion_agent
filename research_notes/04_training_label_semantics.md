# Finding 04 — Phase-1 / Phase-2 Label Semantics

## Direct source trace

`run_gsm8k.py` establishes:

1. raw `topology_matrix` → `Graph(... fixed_spatial_masks=topology_matrix)`;
2. graph → `arun(...)`;
3. MAS answer → parsed prediction;
4. `utility` = binary correctness;
5. `cost = sum(sum(row) for row in topology_matrix)`;
6. raw graph + `[utility, cost]` are stored;
7. the same raw graph is later converted with `dense_to_sparse` for Proxy training;
8. successful raw graphs are appended directly as diffusion \(A_0\) targets.

See `evidence/source_trace.md`.

## Exact dataset checks

300 Phase-1 records:

- cost == number of ones in raw adjacency: **300/300**
- cost == non-self raw edge count: **300/300**
- utility is binary: **300/300**
- utility distribution: **282 success / 18 failure**
- unique raw topologies: **6**

## Semantic audit of the six training topologies

Average fidelity:

\[
J(raw,old\ execution)=0.278
\]

\[
J(raw,canonical\ execution)=0.683.
\]

Cost alignment:

- cost vs raw graph: MAE **0.000**, Pearson **1.000**
- cost vs old executed graph edge count: MAE **2.333**, Pearson **0.913**

## Careful interpretation

The stored reward vector is built from two quantities with different provenance:

\[
utility = correctness\ produced\ after\ runtime\ execution
\]

\[
cost = |E(raw\ topology)|.
\]

Because runtime execution reinterprets the raw mask, it is misleading to treat both labels as measurements of one unchanged topology object without qualification.

This is a semantic-provenance issue, not proof that the Proxy is mathematically incapable of learning: the runtime transformation is deterministic, so \(A\mapsto U(T(A))\) is still a learnable mapping in principle.
