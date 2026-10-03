# Finding 02 — Representation–Execution Coordinate Mismatch

## Source mechanism

In `Graph.__init__`, `fixed_spatial_masks` is flattened. The effective (later-defined) `init_potential_edges()` keeps only non-self pairs. The execution loop then zips potential edges with the flattened mask.

For four agents:

\[
N^2=16\quad\text{mask positions},\qquad N(N-1)=12\quad\text{potential spatial edges}.
\]

An exhaustive one-hot probe confirmed that flattened mask positions are consumed in non-self edge order, causing diagonal and later matrix positions to be reinterpreted or ignored.

## 603-record structural audit

- mean raw edges: **3.536**
- mean executed edges: **2.968**
- raw→mapped Jaccard: **0.164**
- mapped→executed Jaccard: **0.837**
- raw→executed Jaccard: **0.152**
- raw→executed F1: **0.235**
- reachability Jaccard: **0.243**
- raw==executed: **0.0%**
- raw/executed Jaccard < 0.50: **96.2%**

## Why the decomposition matters

The major loss occurs in:

\[
raw \rightarrow mapped
\]

rather than:

\[
mapped \rightarrow executed.
\]

Therefore cycle filtering is not the dominant source of mismatch in these audited runs.

## Concrete Phase-1 case

See:

`evidence/key_cases/case_phase1_coordinate_mismatch.json`

A chain \(0\to1\to2\to3\) has:

- raw→old executed Jaccard = **0**
- raw→canonical executed Jaccard = **1**

while the stored Phase-1 cost remains 3.

## Scope

Claim only:

> The audited released-code path exhibits a systematic representation–execution coordinate mismatch.

Do not infer that every paper benchmark or repository revision is affected without checking.
