# Finding 03 — Canonical Runtime Repair

## Intervention

The adapter preserves the intended non-self edge coordinate order and converts it into the positions consumed by the released runtime. No changes are made to:

- diffusion weights;
- Proxy weights;
- candidate sampling;
- cycle filtering;
- agent model;
- task set.

## Offline counterfactual, 603 records

| Metric | Old runtime | Canonical repair |
|---|---:|---:|
| raw→executed Jaccard | 0.152 | 0.863 |
| precision | 0.280 | 1.000 |
| recall | 0.240 | 0.863 |
| F1 | 0.235 | 0.918 |
| reachability Jaccard | 0.243 | 0.823 |
| normalized Hamming | 0.391 | 0.054 |

Additional checks:

- mapping exact before cycle filtering: **100.0%**
- Jaccard improved: **99.7%**
- gain > 0.25: **95.0%**
- gain > 0.50: **76.5%**

## Online confirmation

The canonical smoke run reproduced the expected high raw→executed fidelity in real MAS execution.

## Interpretation

This is a controlled recovery experiment: the large restoration in graph semantics follows from changing only the coordinate adapter. It strongly supports the coordinate-mismatch mechanism.

It does **not** imply that downstream accuracy must increase. The learned checkpoints and decision architecture can make downstream behavior insensitive or adapted to other semantics.
