from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Tuple

from gymnasium import Space
from numpy import ndarray

from switchfl.rail_network import RailNetwork
from switchfl.spaces import MultiDiscreteSwitchObsSpace
from switchfl.switch_agents import _Switch
from switchfl.utils.logging import format_logger
from switchfl.utils.naming import name2switch_id, symmetric_string
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent as TrainAgent
import numpy as np


class _Observer(ABC):
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)
        self.logger = format_logger(self.logger)


    @abstractmethod
    def observe(
        self, agent: str, rail_env: RailEnv, rail_network: RailNetwork
    ) -> Tuple[ndarray, Dict[str, Any]]:
        """Observe the environment and return the observation for the given agent.

        Args:
            agent (str): The ID of the agent.
            rail_env (RailEnv): The rail environment.
            rail_network (RailNetwork): The rail network.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            Tuple[ndarray, Dict[str, Any]]: The observation for the agent.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(
        self, agent: str, rail_env: RailEnv, rail_network: RailNetwork, seed: int = None
    ) -> Space:
        """Get the observation space for the given agent.

        Args:
            agent (str): The ID of the agent.
            rail_env (RailEnv): The rail environment.
            rail_network (RailNetwork): The rail network.
            seed (int, optional): The random seed. Defaults to None.

        Raises:
            NotImplementedError: If the method is not implemented.

        Returns:
            Space: The observation space for the agent.
        """
        raise NotImplementedError

    def human_format(self, observation: Any) -> Dict[str, Any]:
        """Convert the raw observation into a human-readable format.

        Args:
            observation (Any): The raw observation.

        Returns:
            Dict[str, Any]: The human-readable observation.
        """
        if not hasattr(self, "obs_space"):
            self.logger.error(
                "No observation space initialized -> Not able to convert observation"
            )
            return
        if not hasattr(self.obs_space, "human_format"):
            self.logger.error(
                "Given observation space has not method to reinterpret the given observation -> Abort"
            )
            return
        return self.obs_space.human_format(observation)
    

class StandardObserver(_Observer):
    def __init__(self, delay_levels: int = 3, delay_threshold: int = 20):
        super().__init__()

        self.delay_levels = delay_levels

        self.delay_threshold = delay_threshold  # arbitrary value

    def _compute_delay(self, rail_env: RailEnv, train: TrainAgent):
        """
        Returns the delay of the given train

        Args:
            train_id (int): The id of the train

        Returns:
            int: The delay of the given train
        """
        row, col = (
            train.position if train.position is not None else train.initial_position
        )
        min_dist_to_target = rail_env.distance_map.get()[
            train.handle, row, col, train.direction
        ]
        # assume you are moving with max speed=1
        return rail_env._elapsed_steps - train.latest_arrival + min_dist_to_target

    def _discretize_delay(self, train: TrainAgent, delay: int) -> int:
        """
        Discretizes the delay of the given train

        Args:
            train_id (TrainAgent): The id of the train
            delay (int): The delay of the train

        Returns:
            int: The discretized delay of the given train
        """
        available_time = train.latest_arrival - train.earliest_departure
        if delay <= 0:
            return 0
        if delay <= available_time * self.delay_threshold:
            return 1
        return 2

    def observe(self, agent, rail_env, rail_network) -> Tuple[ndarray, Dict[str, Any]]:
        self.logger.debug(symmetric_string(f"obs {agent}", frame="~"))
        node_id = name2switch_id(agent)
        switch: _Switch = rail_network.get_switch_on_position(node_id)

        semaphore = []
        target = []
        delay = []

        train_at_ports = {
            rail_network.get_trains_next_port(train): train for train in rail_env.agents
        }
        logging_var = {k: t.handle for k, t in train_at_ports.items()}
        self.logger.debug(f"train at ports {logging_var}")
        self.logger.debug(f"next port for each train: {rail_network._train2next_port}")
        self.logger.debug(f"ports of switch: {agent} with position")
        train_counter = 0  # debugging
        for port in switch.get_port_nodes():
            self.logger.debug(f"\t{port} {switch.switch_graph.nodes.data("rail_prev_node")[port]}")
            semaphore.append(switch.semaphores[port])

            train = train_at_ports.get(port)
            if train is None:
                # train is not at requested port
                delay.append(-1)
                target.extend([-1, -1])  # 2D coordinates
            else:
                delay.append(
                    self._discretize_delay(train, self._compute_delay(rail_env, train))
                )
                target.extend(train.target)
                train_counter += 1

        if (
            train_counter == 0
        ):  # there is no train at the port -> there is no point for observing it
            # import matplotlib.pyplot as plt
            # from flatland.utils.rendertools import AgentRenderVariant
            # import networkx as nx
            # fig, (ax1, ax2) = plt.subplots(ncols=2)
            # rgb = rail_env.render(
            #     agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS, show_debug=True
            # )
            # ax1.imshow(rgb)
            # nx.draw(
            #     rail_network.rail_graph.to_undirected(),
            #     rail_network.rail_graph.nodes.data("position"),
            #     with_labels=True,
            #     node_color=dict(
            #         rail_network.rail_graph.nodes.data(data="node_color")
            #     ).values(),
            #     edge_color="gray",
            #     node_size=3,
            #     font_size=5,
            #     ax=ax2,
            # )
            # ax1.axis("off")
            # ax2.axis("off")
            # plt.show()


            self.logger.fatal       (
                "Bug detected. No train detected at active switch. Check information in RailNetwork."
            )

        semaphore = np.array(semaphore).astype(int)
        target = np.array(target).astype(int)
        delay = np.array(delay).astype(int)
        observation = np.concatenate([semaphore, target, delay], dtype=np.int64)
        info = {"action_mask": switch.get_action_mask()}
        return observation, info

    def get_observation_space(self, agent, rail_env, rail_network, seed: int = None):
        node_id = name2switch_id(agent)
        switch = rail_network.get_switch_on_position(node_id)
        return MultiDiscreteSwitchObsSpace(
            n_gaits=switch.n_gaits,
            rail_grid_shape=(rail_env.rail.height, rail_env.rail.width),
            n_delay_levels=3,
            seed=seed,
        )
