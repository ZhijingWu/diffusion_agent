# Source-Level Provenance Trace

This file records the minimal source path supporting the final technical note. Line numbers refer to the source copies in `source_reference/`.

## Phase-1 generation and labels

`source_reference/experiments/run_gsm8k.py`:

- Lines 137–142: each static `topology_matrix` is passed to `Graph(..., fixed_spatial_masks=topology_matrix)`.
- Line 143: the graph is executed with `gdesigner_graph.arun(...)`.
- Lines 145–146: the model answer is parsed and `utility` is binary task correctness.
- Line 147: `cost = sum(sum(row) for row in topology_matrix)`.
- Lines 149–155: the same raw `topology_matrix` becomes the Proxy graph (`dense_to_sparse`) and is paired with `[utility, cost]`.
- Lines 158–164: the JSONL stores the raw graph and its `performance`.
- Lines 183–190: Phase 2 reloads the stored raw graph for Proxy training.
- Lines 194–198: successful raw graphs are appended directly as diffusion `A0` targets.

This establishes the Phase-1 / Phase-2 provenance chain directly from source.

## Runtime mask interpretation

`source_reference/GDesigner/graph/graph.py`:

- Line 56: `fixed_spatial_masks` is converted to a tensor and flattened with `.view(-1)`.
- Lines 73–74: after node creation, `self.init_potential_edges()` is called.
- The class contains two definitions of `init_potential_edges`; the later definition at lines 498–506 overrides the earlier one in Python class construction.
- Lines 498–505: the effective method creates only non-self potential edges (`if i != j`), giving `N(N-1)` spatial edges for `N` agents.
- Lines 285–292: execution zips `potential_spatial_edges`, `spatial_logits`, and flattened `spatial_masks`; fixed masks are then added subject to cycle checking.

For four agents this means a 16-position flattened raw mask is zipped against 12 non-self potential edges. The exhaustive one-hot probe in our supporting audit established the resulting coordinate mapping.

## Final decision aggregation

`source_reference/GDesigner/graph/graph.py`:

- Lines 277–279: `connect_decision_node()` adds every agent as a predecessor of the decision node.
- Lines 411–416: asynchronous execution calls `connect_decision_node()` after the agent rounds, then executes the final decision node.

This is the source basis for the **topology-bypass hypothesis**. The sink-only experiment is an ablation of this architecture, not a claim about the intended benchmark definition.

## Important wording discipline

The source trace supports:
- a representation/execution coordinate mismatch in the audited released-code path;
- Phase-1 utility and cost being constructed from different semantic objects in the sense that utility is produced after runtime interpretation, while cost is computed from the raw topology matrix;
- direct reuse of the raw graph for Proxy and diffusion supervision.

It does **not** by itself establish:
- that all paper results are invalid;
- that the same issue affects every benchmark or repository revision;
- global novelty relative to all external discussions.
