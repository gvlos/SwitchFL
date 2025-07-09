from typing import Dict, List, Tuple
from gymnasium import Space
import networkx as nx
import pandas as pd
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.agent_utils import EnvAgent as TrainAgent

from environment import NodeId, PortId, TrainAgentHandle
from environment.switch_agents import _Switch, Switch2, get_switch_type
from environment.utils.naming import get_node_id_on_port_id, switch_id2name
from environment.utils.rail_graph import (
    create_rail_graph,
    generate_local_switch_graphs,
    insert_switch_proximity_nodes,
    prune_non_switches,
)


def build_switch_network(rail_network: nx.Graph) -> nx.Graph:
    switch_network = nx.Graph()
    df = pd.DataFrame(
        [attr for _, attr in rail_network.nodes.data()],
        index=[idx for idx, _ in rail_network.nodes.data()],
    )
    edges = set()
    for node, attr in df.groupby("switch_id"):
        switch_graph = rail_network.subgraph(attr.index)

        # find neighbors for edges
        port2neighbor = {}
        for port in switch_graph.nodes:
            neighbors = rail_network.neighbors(port)
            neighbor_port_df = df[df.index.isin(neighbors) & (df["switch_id"] != node)]
            edges.add((node, neighbor_port_df["switch_id"].item()))
            port2neighbor[port] = neighbor_port_df["switch_id"].item()

        switch = get_switch_type(switch_graph)(node, switch_graph, port2neighbor)

        switch_network.add_node(
            node,
            switch_cls=switch,
            node_color=attr["node_color"].iloc[0],
            position=(node[1], -node[0]),
        )

    switch_network.add_edges_from(edges)
    return switch_network


def build_rail_graph(rail_env: RailEnv) -> nx.Graph:
    rail_env.reset()
    graph = create_rail_graph(rail_env)
    graph = insert_switch_proximity_nodes(graph)
    graph = prune_non_switches(graph)
    graph = generate_local_switch_graphs(graph)
    return graph


class RailNetwork:
    def __init__(self, rail_env: RailEnv):
        self.switch_network: nx.Graph

        self.rail_graph = build_rail_graph(rail_env)
        self.switch_network = build_switch_network(self.rail_graph)

        # build lookup dictionaries
        self._pos2port: Dict[Tuple[int, int], PortId] = {
            pos: port for port, pos in self.rail_graph.nodes.data("rail_prev_node")
        }
        """rail position before entering a node"""
        self._pos2switch: Dict[Tuple[int, int], _Switch] = {
            switch_node: switch
            for switch_node, switch in self.switch_network.nodes.data("switch_cls")
        }
        """switch position to switch class"""
        
    def reset(self):
        # reset all switches
        for node, switch in self.switch_network.nodes.data("switch_cls"):
            switch: _Switch
            switch.reset()
        

    def get_switch_on_position(self, switch: NodeId) -> _Switch | None:
        """get switch class for the corresponding position of the switch

        Args:
            switch (NodeId): position in rail grid

        Returns:
            _Switch | None: if given node does not belong to a switch return None. Otherwise the switch object
        """
        if switch not in self.switch_network.nodes:
            return None
        return self.switch_network.nodes.data("switch_cls")[switch]

    def get_neighbor_switch(self, port: PortId) -> _Switch:
        node_id = get_node_id_on_port_id(port)
        switch = self.get_switch_on_position(node_id)
        neighbor_node_id = switch.port2neighbor[port]
        neighbor_switch = self.get_switch_on_position(neighbor_node_id)
        return neighbor_switch

    def get_switch_on_port(self, port: PortId) -> _Switch:
        """get the switch if the position is a node before entering a switch node

        Args:
            position (PortId): position in rail network

        Returns:
            _Switch | None: either a switch or None, if the given position does not belong to a switch
        """
        node_id = get_node_id_on_port_id(port)
        return self._pos2switch[node_id]

    def get_port_on_position(self, position: Tuple[int, int]) -> PortId | None:
        """get the port node if the position is a node before entering a switch node

        Args:
            position (Tuple[int, int]): position in rail network

        Returns:
            PortId | None: either a port or None, if the given position does not belong to a switch
        """
        return self._pos2port.get(position)

    def free_semaphore(self, port: PortId):
        switch = self.get_switch_on_port(port)
        switch.free_port(port)
        node_id = get_node_id_on_port_id(port)
        self.switch_network.update(nodes={node_id: {"switch_cls": switch}})

    def block_semaphore(self, port: PortId):
        switch = self.get_switch_on_port(port)
        switch.block_port(port)
        node_id = get_node_id_on_port_id(port)
        self.switch_network.update(nodes={node_id: {"switch_cls": switch}})

    def transition_semaphore(self, source: PortId, target: PortId):
        self.free_semaphore(source)
        self.block_semaphore(target)

        # account for the case where the next switch is a simple intersections.
        # In this case also the next semaphores have to be switched on
        if isinstance(self.get_neighbor_switch(target), Switch2):
            # find the next switch after this one
            # TODO
            pass

    def get_train_actions(
        self, node: NodeId, action: int, train_agents: List[TrainAgent]
    ) -> Dict[TrainAgentHandle, List[RailEnvActions]]:
        switch = self._pos2switch[node]
        return switch.get_train_action(action, train_agents)

    def get_switch_names(self) -> List[str]:
        res = []
        for node in self.switch_network.nodes:
            res.append(switch_id2name(node))
        return res

    def get_switch_action_space(self, node: NodeId, seed: int = None) -> Space:
        switch = self.get_switch_on_position(node)
        return switch.get_action_space(seed=seed)