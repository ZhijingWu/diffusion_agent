# Source trace

This note records the source locations used to establish the execution and supervision paths discussed in `REPRODUCTION.md`.

## Phase-1 graph, utility, and cost

Source: `experiments/run_gsm8k.py`

In the Phase-1 loop:

- each static `topology_matrix` is passed to `Graph(..., fixed_spatial_masks=topology_matrix)`;
- the graph is executed with `gdesigner_graph.arun(...)`;
- the returned answer is parsed and `utility` is set from task correctness;
- `cost` is computed directly as `sum(sum(row) for row in topology_matrix)`;
- the same raw `topology_matrix` is converted with `dense_to_sparse` for the Proxy graph and stored in the generated dataset.

In Phase 2:

- the stored `item['graph']` is loaded again for Proxy graph construction;
- `[utility, cost]` is used as the Proxy target;
- successful stored raw graphs are appended directly to the diffusion \(A_0\) training set.

## Spatial-mask interpretation

Source: `GDesigner/graph/graph.py`

The relevant sequence is:

1. `fixed_spatial_masks` is converted to a tensor and flattened with `.view(-1)`;
2. node creation is followed by `self.init_potential_edges()`;
3. the effective later definition of `init_potential_edges()` creates spatial edges only for `i != j`;
4. `construct_spatial_connection()` iterates over:

   ```python
   zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks)
   ```

   and adds fixed-mask edges subject to cycle checking.

For four agents, this gives 16 flattened mask positions and 12 non-self potential spatial edges.

The one-hot coordinate probe and the 603-record topology audit quantify the resulting mapping rather than relying on source inspection alone.

Relevant summaries:

```text
audit_results/topology_semantic_fidelity/summary.json
audit_results/canonical_runtime_repair/summary.json
```

## Final decision aggregation

Source: `GDesigner/graph/graph.py`

`connect_decision_node()` iterates over all agent nodes and adds the final decision node as a successor.

At the end of asynchronous execution, the runtime calls:

```python
self.connect_decision_node()
await self.decision_node.async_execute(input)
```

This is the source basis for the sink-only ablation. The ablation is used only as an exploratory test of whether global final aggregation can reduce observable sensitivity to intermediate spatial topology.

## Scope

These source locations establish the execution and supervision paths used in this audit.

They do not by themselves establish effects on other benchmarks, checkpoints, or repository revisions, and they do not establish external novelty of the observations.
