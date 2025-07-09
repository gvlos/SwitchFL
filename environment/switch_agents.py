import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Type

import networkx as nx
import numpy as np
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.rail_env import RailEnvActions
from gymnasium import Space, spaces

from environment import NodeId, PortId, TrainAgentHandle
from environment.utils.rail_graph import add_rail_actions
from environment.utils.switch_agent import build_rail_action_map


def build_switch_to_rail_actions(
    switch_graph: nx.Graph,
) -> Tuple[List[Dict[Any, List[RailEnvActions]]], List[Tuple[Any, Any]]]:
    """returns a list of actions for each port.

    Args:
        switch_graph (nx.Graph): switch graph with ports and inter-connectivity's

    Returns:
        Tuple[List[Dict[Any, List[RailEnvActions]]], List[Any, Any]]: Each entry in the parent list corresponds to one action
            1. List[Dict[Any, List[RailEnvActions]]]: each entry contains commands for trains at each port of the switch
            2. List[Any, Any]: After executing, across which ports will the train transition the switch (source, target)
    """
    # add actions to switch_graph
    switch_graph = add_rail_actions(switch_graph)
    action_map, outcomes = build_rail_action_map(switch_graph)
    return action_map, outcomes


class _Switch(ABC):
    def __init__(
        self,
        id: NodeId,
        switch_graph: nx.Graph,
        port2neighbor: Dict[PortId, NodeId] = None,
    ):
        self.id = id
        self.switch_graph = switch_graph
        self.port2neighbor = port2neighbor

        self.n_gaits = len(self.switch_graph.nodes)
        self.n_rails = len(self.switch_graph.edges)

        ab = build_switch_to_rail_actions(self.switch_graph)
        self.actions = ab[0]
        """List of actions per port: self.actions[z]: 2 actions per port"""
        self.action_outcomes = ab[1]
        """List of port mappings. self.action_outcomes[z]: train from port x to port y"""

        self.semaphores: Dict[PortId, bool]
        """which ports are blocked: True, which are free: False"""
        self._port_nodes = OrderedDict(
            {int(str(node[0])[-1]): node for node in self.switch_graph.nodes}
        )
        """to have a ordered list of port nodes"""
        self._pos2port: Dict[Tuple[int, int], PortId] = {
            pos: port for port, pos in self.switch_graph.nodes.data("rail_prev_node")
        }
        """rail position before entering a node"""

        self.reset()

    def reset(self):
        self.semaphores = {port: False for port in self.switch_graph.nodes}

    def block_port(self, port: PortId):
        """indicate a given port is blocked because of an incoming train

        Args:
            port (PortId): which port is blocked
        """
        if port not in self.semaphores.keys():
            logging.error(f"{port=} is not part of switch:{self.id}")
            return
        self.semaphores[port] = True

    def free_port(self, port: PortId):
        """indicate a given port is freed because of an incoming train is already processed

        Args:
            port (PortId): which port is freed
        """
        if port not in self.semaphores.keys():
            logging.error(f"{port=} is not part of switch:{self.id}")
            return
        self.semaphores[port] = False

    def get_action_mask(self) -> np.ndarray:
        """which actions are allowed wrt. incoming train semaphores

        Returns:
            np.ndarray: integer array. 1: action allowed, 0: action forbidden (n_actions, )
        """
        mask = [self.semaphores[target] for _, target in self.action_outcomes]
        mask = (~np.array(mask)).astype(np.int8)
        return mask

    def get_train_action(
        self, action: int, train_agents: List[TrainAgent]
    ) -> Dict[TrainAgentHandle, List[RailEnvActions]]:
        """For the given trains which are about to enter this switch, return the actions sequences for each train

        Args:
            action (int): discrete action
            train_agents (List[TrainAgent]): all trains on the grid

        Returns:
            Dict[TrainAgentHandle, List[RailEnvActions]]: For each train at the switch return actions to perform
        """
        _, target_port = self.action_outcomes[action]
        if self.semaphores[target_port]:
            raise RuntimeError(
                f"Semaphore is blocked for action: {self.action_outcomes[action]}"
            )

        result = {}
        for train_agent in train_agents:
            port_node = self._pos2port.get(train_agent.position)
            if port_node is None:
                # train is not at a port node
                continue
            result[train_agent.handle] = self.actions[action][port_node]

        # If only one train and it is STOP_MOVING
        if (
            len(result) == 1
            and next(iter(result.values()))[0] == RailEnvActions.STOP_MOVING
        ):
            result[next(iter(result))] = [RailEnvActions.STOP_MOVING]
        return result

    def get_port_nodes(self) -> List[PortId]:
        return list(self._port_nodes.values())

    @abstractmethod
    def get_action_space(self, seed: int = None) -> Space:
        raise NotImplementedError


# T or Y junction
class Switch1(_Switch):
    def __init__(self, id, switch_graph, port2neighbor=None):
        super().__init__(id, switch_graph, port2neighbor)

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2
        # switch gait: 3
        # 0  1  2
        # --------
        # g  w  w
        # w  g  w
        # w  w  g1
        # w  w  g2
        # can have a different permutation based on orientation
        return spaces.Discrete(4, seed=seed)


# Intersection
class Switch2(_Switch):
    def __init__(self, id, switch_graph, port2neighbor=None):
        super().__init__(id, switch_graph, port2neighbor)

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2, 3
        # switch gait: None
        # 0  1  2  3
        # ----------
        # g  w  g  w
        # w  g  w  g
        return spaces.Discrete(2, seed=seed)


# Intersection with one pass
class Switch3(_Switch):
    def __init__(self, id, switch_graph, port2neighbor=None):
        super().__init__(id, switch_graph, port2neighbor)

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2, 3
        # switch gait: 0, 3
        # 0  1  2  3
        # ----------
        # g1 w  w  w
        # g2 w  w  w
        # w  g  w  w
        # w  w  g  w
        # w  w  w  g1
        # w  w  w  g2
        return spaces.Discrete(6, seed=seed)


# Intersection with two passes
class Switch4(_Switch):
    def __init__(self, id, switch_graph, port2neighbor=None):
        super().__init__(id, switch_graph, port2neighbor)

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2, 3
        # switch gait: 0, 1, 2, 3
        # 0  1  2  3
        # ----------
        # g1 w  w  w
        # g2 w  w  w
        # w  g1 w  w
        # w  g2 w  w
        # w  w  g1 w
        # w  w  g2 w
        # w  w  w  g1
        # w  w  w  g2
        return spaces.Discrete(8, seed=seed)


SWITCH_AGENT_MAP = {
    (3, 2): Switch1,
    (4, 2): Switch2,
    (4, 3): Switch3,
    (4, 4): Switch4,
}


def get_switch_type(switch_graph: nx.Graph) -> Type[_Switch]:
    n_gaits = len(switch_graph.nodes)
    n_rails = len(switch_graph.edges)
    try:
        return SWITCH_AGENT_MAP[(n_gaits, n_rails)]
    except KeyError:
        raise ValueError(f"No Agent with {n_gaits=} and {n_rails=}")
