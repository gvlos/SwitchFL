from typing import Any, Dict, List, Tuple
from tqdm import tqdm
import numpy as np

import networkx as nx
from flatland.envs.rail_env import RailEnv, RailEnvActions
from itertools import product
import matplotlib.colors as mcolors
import matplotlib as mpl
from itertools import combinations


def create_rail_graph(env: RailEnv, cmap="tab20") -> nx.Graph:
    graph = nx.Graph()
    cmap = mpl.colormaps[cmap]

    get_txs = env.rail.get_transitions
    for row, col, dir in tqdm(
        product(range(env.height), range(env.width), range(4)),
        total=env.height * env.width * 4,
    ):
        # [North, East, South, West]
        if bin(env.rail.get_full_transitions(row, col)).count("1") == 0:
            continue
        for dir_idx, tx in enumerate(get_txs(row, col, dir)):
            if tx == 0:
                continue
            # Find next cell coordinates
            next_row = row
            next_col = col
            if dir_idx == 0:  # North
                next_row -= 1
            elif dir_idx == 1:  # East
                next_col += 1
            elif dir_idx == 2:  # South
                next_row += 1
            else:  # West
                next_col -= 1

            # Check if next cell is within the grid
            if (
                next_row >= 0
                and next_row < env.width
                and next_col >= 0
                and next_col < env.height
            ):
                node_color = mcolors.to_hex(
                    mcolors.to_rgba_array(cmap(np.random.randint(20)))
                )
                if not graph.has_node((row, col)):
                    pos = np.array(
                        [col, -row]
                    )  # covert image pixel space to plotting space
                    graph.add_node(
                        (row, col),
                        transition=bin(env.rail.get_full_transitions(row, col))[
                            2:
                        ].zfill(16),
                        node_color=node_color,
                        position=pos,
                        switch_id=(row, col),
                    )
                if not graph.has_node((next_row, next_col)):
                    pos = np.array(
                        [next_col, -next_row]
                    )  # covert image pixel space to plotting space
                    graph.add_node(
                        (next_row, next_col),
                        transition=bin(
                            env.rail.get_full_transitions(next_row, next_col)
                        )[2:].zfill(16),
                        node_color=node_color,
                        position=pos,
                        switch_id=(next_row, next_col),
                    )

                graph.add_edge(
                    (row, col),
                    (next_row, next_col),
                    rail_nodes=[],
                    # rail_node_to_switch={},
                )
    return graph


def insert_switch_proximity_nodes(graph: nx.Graph) -> nx.Graph:
    new_nodes = set()
    for node in list(graph.nodes):
        node_degree = graph.degree(node)
        if node_degree == 2:
            continue

        # add surrounding nodes
        for idx, neighbor in enumerate(list(graph.neighbors(node))):
            new_node = (node[0] + 0.1 * (idx + 1), node[1] + 0.1 * (idx + 1))
            new_nodes.add(new_node)
            pos = (graph.nodes[neighbor]["pos"] + 2 * graph.nodes[node]["pos"]) / 3
            switch_color = graph.nodes[node]["node_color"]
            switch_id = graph.nodes[node]["switch_id"]
            graph.add_node(
                new_node,
                node_color=switch_color,
                position=pos,
                switch_id=switch_id,
                switch_position=switch_id,
                rail_prev_node=graph.nodes[neighbor]["switch_id"],
                approaching_trains=set(),
            )
            graph.add_edge(
                neighbor,
                new_node,
                rail_nodes=[],
            )
            graph.add_edge(
                node,
                new_node,
                rail_nodes=[],
            )
            graph.remove_edge(node, neighbor)
    return graph


def prune_non_switches(graph: nx.Graph) -> nx.Graph:
    assert (
        not graph.is_directed()
    ), "Only applicable for undirected graphs. But given graph is directed."
    for node in list(graph.nodes):
        node_degree = graph.degree(node)
        neighbors_degrees = set([graph.degree(n) for n in graph.neighbors(node)])
        if node_degree == 2 and neighbors_degrees == set([2]):
            prev_node, next_node = list(graph.neighbors(node))

            graph.add_edge(
                prev_node,
                next_node,
                rail_nodes=[
                    node,
                    *graph.edges[(prev_node, node)]["rail_nodes"],
                    *graph.edges[(node, next_node)]["rail_nodes"],
                ],
            )
            graph.remove_edge(prev_node, node)
            graph.remove_edge(node, next_node)
            graph.remove_node(node)
    return graph


def generate_local_switch_graphs(graph: nx.Graph) -> nx.Graph:
    rel_pos = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    directions = "NSEW"
    directions = dict(zip(rel_pos, directions))
    dir_combo = np.array(
        [
            "NN",
            "NE",
            "NS",
            "NW",
            "EN",
            "EE",
            "ES",
            "EW",
            "SN",
            "SE",
            "SS",
            "SW",
            "WN",
            "WE",
            "WS",
            "WW",
        ]
    )

    def find_degree_nodes(G: nx.Graph, k: int, rel=lambda x, y: x == 2) -> bool:
        res = [node for node in G.nodes if rel(G.degree(node), k)]
        return res

    visited = set()
    while True:
        nodes = find_degree_nodes(graph, 2, lambda x, y: x > y)
        if (
            set(graph.nodes) == set(nodes)
            or set(nodes) == visited
            or (set(nodes) - visited) == set()
        ):
            break
        node = (set(nodes) - visited).pop()
        visited.add(node)

        allowed_transitions = graph.nodes[node]["transition"]
        switch_graph = graph.subgraph(nx.ego_graph(graph, node))

        # get relative positions of the subgraphs
        for current_node, next_node in combinations(switch_graph.nodes, 2):
            if current_node == node or next_node == node:
                continue
            visited.add(current_node)
            visited.add(next_node)
            node_pos = graph.nodes[node]["position"]
            current_node_pos = graph.nodes[current_node]["position"]
            next_node_pos = graph.nodes[next_node]["position"]

            train_facing = tuple(np.sign((node_pos - current_node_pos)).tolist())
            train_going = tuple(np.sign((next_node_pos - node_pos)).tolist())
            train_transition = directions[train_facing] + directions[train_going]
            trans_idx = np.where(dir_combo == train_transition)[0].item()
            if allowed_transitions[trans_idx] == "1":
                graph.add_edge(current_node, next_node)

            train_facing = directions[
                tuple(np.sign((node_pos - next_node_pos)).tolist())
            ]
            train_going = directions[
                tuple(np.sign((current_node_pos - node_pos)).tolist())
            ]
            trans_idx = np.where(dir_combo == (train_facing + train_going))[0].item()
            if allowed_transitions[trans_idx] == "1":
                graph.add_edge(
                    next_node, current_node, rail_nodes=[], rail_node_to_switch={}
                )
        graph.remove_node(node)
    return graph


def add_rail_actions(graph: nx.Graph) -> nx.Graph:
    """adds actions to each node.
    Each action annotation is stored at the 'action' datafield as a dictionary:
        - key: where the action leads to
        - value: action sequence

    Example:
    >>> switch_graph = add_rail_actions(switch_graph)
    >>> switch_graph.nodes.data('actions')[<source-node>][<target-node>]

    Args:
        graph (nx.Graph): switch graph without actions

    Returns:
        nx.Graph: switch graph with actions
    """
    actions = {}

    for i, incoming in enumerate(graph.nodes):
        facing_index = (i + 2) % 4  # Opposite direction (facing)
        actions[incoming] = {"actions": {}}  # graph.nodes.data()[incoming]
        for j, target in enumerate(graph.nodes):
            if incoming == target or target not in graph.neighbors(incoming):
                continue  # Cannot go back to where you came from
            relative = (j - facing_index) % 4
            action = [RailEnvActions.MOVE_FORWARD]  # enter switch
            if relative == 0:
                action.append(RailEnvActions.MOVE_FORWARD)
            elif relative == 1:
                action.append(RailEnvActions.MOVE_RIGHT)
            elif relative == 3:
                action.append(RailEnvActions.MOVE_LEFT)
            else:
                action = "invalid"  # Shouldn't occur
            actions[incoming]["actions"][target] = action

    nx.set_node_attributes(G=graph, values=actions)
    return graph
