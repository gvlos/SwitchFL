from itertools import product
from typing import Any, Callable, List, Tuple

import pytest
from switchfl.utils.rail_graph import add_rail_actions
from switchfl.utils.switch_agent import build_rail_action_map
import networkx as nx
from flatland.envs.rail_env import RailEnvActions

GT_ACTIONS = {
    (0.1, 0.1): {
        (0.2, 0.2): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT],
        (0.3, 0.3): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_FORWARD],
        (0.4, 0.4): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT],
    },
    (0.2, 0.2): {
        (0.1, 0.1): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT],
        (0.3, 0.3): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT],
        (0.4, 0.4): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_FORWARD],
    },
    (0.3, 0.3): {
        (0.1, 0.1): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_FORWARD],
        (0.2, 0.2): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT],
        (0.4, 0.4): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT],
    },
    (0.4, 0.4): {
        (0.1, 0.1): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_RIGHT],
        (0.2, 0.2): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_FORWARD],
        (0.3, 0.3): [RailEnvActions.MOVE_FORWARD, RailEnvActions.MOVE_LEFT],
    },
}


def build_complete_switch() -> nx.Graph:
    nodes = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)]

    edges = []
    for node1, node2 in product(nodes, nodes):
        if (node2, node1) in edges or node1 == node2:
            continue
        edges.append((node1, node2))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def build_y_crossing(drop_node: int, drop_edge: int) -> nx.Graph:
    assert drop_node >= 0 and drop_node <= 3
    assert drop_edge >= 0 and drop_edge <= 2

    g = build_complete_switch()
    # remove one node
    nodes = list(g.nodes)
    nodes.pop(drop_node)
    g: nx.Graph = g.subgraph(nodes)

    # remove one edge
    edges = list(g.edges)
    edges.pop(drop_edge)
    g: nx.Graph = g.edge_subgraph(edges)
    return g


def build_intersection(retain_edges: List[int]) -> nx.Graph:
    assert len(retain_edges) >= 0 and len(retain_edges) <= 3
    intersection_edges = [((0.1, 0.1), (0.3, 0.3)), ((0.2, 0.2), (0.4, 0.4))]
    diagonal_edges = [
        ((0.1, 0.1), (0.2, 0.2)),
        ((0.1, 0.1), (0.4, 0.4)),
        ((0.2, 0.2), (0.3, 0.3)),
        ((0.2, 0.2), (0.4, 0.4)),
    ]

    edges = intersection_edges
    for retain_edge in retain_edges:
        assert retain_edge >= 0 and retain_edge <= 3
        edges.append(diagonal_edges[retain_edge])

    g = nx.Graph()
    g.add_edges_from(edges)
    return g


def build_test_data() -> Tuple[Any, List[str]]:
    test_data = [
        lambda: build_y_crossing(drop_node, drop_edge)
        for drop_node, drop_edge in product(range(4), range(3))
    ]
    ids = ["y-crossing"] * len(test_data)

    test_data.append(lambda: build_intersection([]))
    ids.append("intersection-simple")

    test_data.extend(
        [lambda: build_intersection([retain_edge]) for retain_edge in range(4)]
    )
    ids.extend(["intersection-one"] * 4)

    test_data.extend(
        [
            lambda: build_intersection(retain_edge_pair)
            for retain_edge_pair in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
        ]
    )
    ids.extend(["intersection-two"] * 6)
    return test_data, ids


TEST_DATA, ids = build_test_data()


@pytest.mark.parametrize("graph_gen_func", TEST_DATA, ids=ids)
def test_y_switches(graph_gen_func: Callable[[], nx.Graph]):
    switch_graph = graph_gen_func()
    switch_graph = add_rail_actions(switch_graph)
    action_map, outcomes = build_rail_action_map(switch_graph)

    assert len(action_map) == 2 * len(switch_graph.edges)

    for action, outcome in zip(action_map, outcomes):
        source, target = outcome
        gt_action_seq = GT_ACTIONS[source][target]
        for action_node, action_seq in action.items():
            # action node: node from which to perform the action
            if action_node == source:
                assert action_seq == gt_action_seq
            else:
                assert action_seq == [
                    RailEnvActions.STOP_MOVING,
                    RailEnvActions.STOP_MOVING,
                ]
