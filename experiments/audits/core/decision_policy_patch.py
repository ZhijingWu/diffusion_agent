"""
Decision-node aggregation ablation for GTD/GDesigner.

This file does NOT modify official source files.
Import and monkeypatch Graph.connect_decision_node before running a runner.
"""

from GDesigner.graph.graph import Graph


def connect_decision_node_sink_only(self):
    """
    Connect only spatial sink agents to the final decision node.

    A sink is an agent with no outgoing spatial successor that is itself
    one of the agent nodes in self.nodes.

    The final decision node is intentionally ignored when determining sinks.
    """
    agent_ids = set(self.nodes.keys())
    sinks = []

    for node_id, node in self.nodes.items():
        outgoing_agent_successors = [
            succ
            for succ in node.spatial_successors
            if getattr(succ, "id", None) in agent_ids
        ]

        if len(outgoing_agent_successors) == 0:
            sinks.append(node)

    # A DAG should always have at least one sink.
    # Keep a defensive fallback so the experiment never silently produces
    # a decision node with no predecessors.
    if not sinks:
        sinks = list(self.nodes.values())

    for node in sinks:
        node.add_successor(self.decision_node)


def install_sink_only_policy():
    """
    Monkeypatch the Graph class for the current Python process only.
    """
    Graph.connect_decision_node = connect_decision_node_sink_only
