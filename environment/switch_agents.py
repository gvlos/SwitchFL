from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Tuple, Type

import networkx as nx
import numpy as np
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.rail_env import RailEnvActions
from gymnasium.spaces import Discrete

from environment.spaces import DiscreteSwitchObsSpace
from environment.utils.rail_graph import add_rail_actions
from environment.utils.switch_agent import build_rail_action_map


class _SwitchAgent(ABC):
    # for the inner representation of an agent

    entity: int
    """what kind of switch are you"""
    n_gaits: int
    """how many rails are connected to the switch"""
    n_rails: int
    """amount of inner connections"""

    def __init__(
        self,
        switch_graph: nx.Graph,
        n_stations: int,
        n_delay_levels: int = 3,
        id: Any = None,
    ):
        super().__init__()
        self.switch_graph = switch_graph
        self.n_stations = n_stations
        self.n_delay_levels = n_delay_levels
        self.id = id
        if self.id is None:
            self.id = list(self.switch_graph.nodes.data("switch_id"))[0][1]

        self.switch_nodes = [
            node for node in self.switch_graph if self.switch_graph.degree(node) == 2
        ]
        self.non_switch_nodes = [
            node for node in self.switch_graph if self.switch_graph.degree(node) == 1
        ]
        self.semaphores = {n: False for n in self.switch_graph.nodes}

        self.action_map, self.outcomes = self._build_switch_rail_actions(
            self.switch_graph
        )

        assert len(self.action_map) == self.get_action_space().n

    def __repr__(self):
        id = self.id
        n_gaits = len(self.switch_graph.nodes)
        n_rails = len(self.switch_graph.edges)
        return f"Switch({id=}, {n_gaits=}, {n_rails=})"

    @classmethod
    def from_switch_graph(
        cls,
        switch_graph: nx.Graph,
        n_stations: int,
        n_delay_levels: int = 3,
        id: Any = None,
    ) -> "_SwitchAgent":
        n_gaits = len(switch_graph.nodes)
        n_rails = len(switch_graph.edges)
        cls = get_agent_abstraction(n_gaits, n_rails)
        return cls(switch_graph, n_stations, n_delay_levels, id=id)

    def get_train_action(
        self, action: int, train_agents: List[TrainAgent]
    ) -> Dict[int, List[RailEnvActions]]:
        res = {}
        for train_agent in train_agents:
            port_node = self._get_port_node_on_position(train_agent.position)
            if port_node is None:
                # train is not at a port node
                continue

            if self.semaphores[self.outcomes[action][1]]:
                raise RuntimeError(
                    f"Semaphore is blocked. Action with transition: {self.outcomes[action]} is not available"
                )
            res[train_agent.handle] = self.action_map[action][port_node]

        # NOTE: if the train is the only train at the switch:
        # just one waiting block -> otherwise we clog up the train_action_plan
        if len(res) == 1 and list(res.values())[0][0] == RailEnvActions.STOP_MOVING:
            res[list(res.keys())[0]] = [RailEnvActions.STOP_MOVING]

        return res

    def block_port(self, port: Any):
        if port not in self.semaphores.keys():
            print(f"{port=} is not part of switch:{self.id}")
            return
        self.semaphores[port] = True

    def free_port(self, port: Any):
        if port not in self.semaphores.keys():
            print(f"{port=} is not part of switch:{self.id}")
            return
        self.semaphores[port] = False

    def get_action_mask(self) -> np.ndarray:
        """which actions are allowed wrt. incoming train semaphores

        Returns:
            np.ndarray: integer array. 1: action allowed, 0: action forbidden (n_actions, )
        """
        mask = [self.semaphores[target] for _, target in self.outcomes]
        mask = (~np.array(mask)).astype(np.int8)
        return mask

    def get_observation_space(self, seed: int = None):
        return DiscreteSwitchObsSpace(
            self.n_gaits, self.n_stations, self.n_delay_levels, seed=seed
        )

    @abstractmethod
    def get_action_space(self, seed: int = None):
        raise NotImplementedError

    @staticmethod
    def _build_switch_rail_actions(
        switch_graph: nx.Graph,
    ) -> List[Dict[Any, List[RailEnvActions]]]:
        """returns a list of actions for each port.

        Args:
            switch_graph (nx.Graph): switch graph with ports and inter-connectivity's

        Returns:
            List[Dict[Any, List[RailEnvActions]]]: each element in the list is a collection of actions for each port:
                Each dictionary: key=port(node)-identifier, value=Sequence of actions(enter switch leave switch)
        """
        # add actions to switch_graph
        switch_graph = add_rail_actions(switch_graph)
        action_map = build_rail_action_map(switch_graph)
        return action_map

    def _get_port_node_on_position(
        self, position: Tuple[int, int]
    ) -> Tuple[float, float] | None:
        for node in self.switch_graph.nodes:
            if self.switch_graph.nodes.data(data="rail_prev_node")[node] == position:
                return node
        return None


# T or Y junction
class SwitchAgent1(_SwitchAgent):
    def __init__(self, switch_graph, n_stations, n_delay_levels=3, id=None):
        super().__init__(switch_graph, n_stations, n_delay_levels, id)
        self.entity = 1
        self.n_gaits = 3
        self.n_rails = 2

    def get_action_space(self, seed: int = None):
        # gaits: 0, 1, 2
        # switch gait: 3
        # 0  1  2
        # --------
        # g  w  w
        # w  g  w
        # w  w  g1
        # w  w  g2
        # can have a different permutation based on orientation
        return Discrete(4, seed=seed)

    @property
    def switch_node(self):
        # only one switch node
        return self.switch_nodes[0]


# T or Y junction
class SwitchAgent1(_SwitchAgent):
    def __init__(self, switch_graph, n_stations, n_delay_levels=3, id=None):
        super().__init__(switch_graph, n_stations, n_delay_levels, id)
        self.entity = 1
        self.n_gaits = 3
        self.n_rails = 2

        # switch orientation (horizontal or vertical)
        self.orientation = self._get_orientation(self.switch_graph)

    def _get_orientation(
        self, switch_graph: nx.Graph
    ) -> Literal["horizontal", "vertical", "v_split"]:
        switch_node_pos = switch_graph.nodes.data(data="pos")[self.switch_node]
        non_switch_node_pos = switch_graph.nodes.data(data="pos")[
            self.non_switch_nodes[0]
        ]

        switch_node_pos = np.array(switch_node_pos)
        non_switch_node_pos = np.array(non_switch_node_pos)

        # move from switch node to non_switch_node
        direction_1 = non_switch_node_pos - switch_node_pos

        non_switch_node_pos = switch_graph.nodes.data(data="pos")[
            self.non_switch_nodes[1]
        ]
        non_switch_node_pos = np.array(non_switch_node_pos)
        direction_2 = non_switch_node_pos - switch_node_pos

        # TODO: does not take rotation if the node into account
        if direction_1[0] == 0 or direction_2[0] == 0:
            return "horizontal"
        elif direction_1[1] == 0 or direction_2[1] == 0:
            return "vertical"
        else:
            return "v_split"

    def get_switch_actions(
        self, source: Any, target: Any
    ) -> Dict[Any, List[RailEnvActions]]:
        switch_policy = {}
        for non_switch_node in set(self.switch_graph.nodes) - set(source):
            switch_policy[non_switch_node] = [
                RailEnvActions.STOP_MOVING,
                RailEnvActions.STOP_MOVING,
            ]

        return switch_policy
        
    def _get_rail_env_actions(self, action):
        # action: which action is there to perform
        # node: which subnode of a swtich subgraph
        # get go straight actions
        switch_policy = {}
        if action == 0:
            switch_policy = self.get_switch_actions(
                self.non_switch_nodes[0], self.switch_node
            )
        elif action == 1:
            switch_policy = self.get_switch_actions(
                self.non_switch_nodes[1], self.switch_node
            )
        elif action == 2:
            switch_policy = self.get_switch_actions(
                self.switch_node, self.non_switch_nodes[0]
            )
        elif action == 3:
            # direction 2 of switch node
            switch_policy = self.get_switch_actions(
                self.switch_node, self.non_switch_nodes[1]
            )
        else:
            raise ValueError("Only for actions [0, ..., 3] available")
        return switch_policy

    def get_action_space(self, seed: int = None):
        # gaits: 0, 1, 2
        # switch gait: 3
        # 0  1  2
        # --------
        # g  w  w
        # w  g  w
        # w  w  g1
        # w  w  g2
        # can have a different permutation based on orientation
        return Discrete(4, seed=seed)

    @property
    def switch_node(self):
        # only one switch node
        return self.switch_nodes[0]


# Intersection
class SwitchAgent2(_SwitchAgent):
    def __init__(self, id, position, n_stations, switch_ports=None, n_delay_levels=3):
        super().__init__(id, position, n_stations, switch_ports, n_delay_levels)
        self.entity = 2
        self.n_gaits = 4
        self.n_rails = 2

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2, 3
        # switch gait: None
        # 0  1  2  3
        # ----------
        # g  w  g  w
        # w  g  w  g
        return Discrete(2, seed=seed)


# Intersection with one pass
class SwitchAgent3(_SwitchAgent):
    def __init__(self, id, position, n_stations, switch_ports=None, n_delay_levels=3):
        super().__init__(id, position, n_stations, switch_ports, n_delay_levels)
        self.entity = 3
        self.n_gaits = 4
        self.n_rails = 3

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
        return Discrete(6, seed=seed)


# Intersection with two passes
class SwitchAgent4(_SwitchAgent):
    def __init__(self, id, position, n_stations, switch_ports=None, n_delay_levels=3):
        super().__init__(id, position, n_stations, switch_ports, n_delay_levels)
        self.entity = 4
        self.n_gaits = 4
        self.n_rails = 4

    def get_action_space(self, seed=None):
        # gaits: 0, 1, 2, 3
        # switch gait: 0, 1, 2, 3
        # 0  1  2  3
        # ----------
        # g1 w  w  w
        # g2 w  w  w
        # w  g1 w  w
        # w  g2 w  ww
        # w  w  g1 w
        # w  w  g2 w
        # w  w  w  g1
        # w  w  w  g2
        return Discrete(8, seed=seed)


def get_agent_abstraction(n_gaits: int, n_rails: int) -> Type[_SwitchAgent]:
    if n_gaits == 3:
        return SwitchAgent1
    elif n_gaits > 4:
        raise ValueError("No agent type with more than 4 rails")

    if n_rails == 2:
        return SwitchAgent2
    elif n_rails == 3:
        return SwitchAgent3
    elif n_rails == 4:
        return SwitchAgent4
    else:
        raise ValueError(f"No Agent with {n_gaits=} and {n_rails=}")
