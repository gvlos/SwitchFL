from typing import Any, Dict, List, Tuple

import networkx as nx
from flatland.envs.rail_env import RailEnvActions


def build_rail_action_map(
    graph: nx.Graph,
) -> Tuple[List[Dict[Any, List[RailEnvActions]]], List[Tuple[Any, Any]]]:
    """computes which discrete environment actions result in which environment actions

    Args:
        graph (nx.Graph): switch graph with actions annotation:
            Each action annotation is stored at the 'action' datafield as a dictionary:
                - key: where the action leads to
                - value: action sequence


    Returns:
        Tuple[List[Dict[Any, List[RailEnvActions]]], List[Any, Any]]: Each entry in the parent list corresponds to one action
            1. List[Dict[Any, List[RailEnvActions]]]: each entry contains commands for trains at each port of the switch
            2. List[Any, Any]: After executing, across which ports will the train transition the switch (source, target)
    """
    actions = []
    target_map = []
    for node, port_actions in graph.nodes.data("actions"):
        for target, a in port_actions.items():
            action = {node: a}
            target_map.append((node, target))

            # add default stop action for all other nodes
            for other_node in graph.nodes:
                if other_node == node:
                    continue
                action[other_node] = [
                    RailEnvActions.STOP_MOVING,
                    RailEnvActions.STOP_MOVING,
                ]

            actions.append(action)
    return actions, target_map
