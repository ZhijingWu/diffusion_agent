# Evidence

This directory contains compact artifacts used to check the claims in `REPRODUCTION.md` without requiring inspection of every raw run.

- `source_trace.md` records the source paths for the Phase-1 label pipeline, spatial-mask interpretation, and final decision aggregation.
- `experiment_index.md` maps each research question to its script and committed output.
- `key_metrics.json` contains the main summary metrics in machine-readable form.
- `evidence_table.csv` provides a tabular summary of the audit findings.
- `paired_downstream_summary.csv` contains the matched downstream comparisons.
- `key_cases/` contains one Phase-1 coordinate-mismatch example and the two sink-only correctness disagreements.

`audit_results/` contains selected direct outputs from the audit scripts; this directory contains the smaller summaries and examples used to navigate them.

The sink-only ablation is exploratory and is kept separate from the source- and structure-level checks.
