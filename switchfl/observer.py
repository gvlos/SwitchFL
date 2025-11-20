from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, Tuple

from gymnasium import Space
from numpy import ndarray

from switchfl.rail_network import RailNetwork
from switchfl.spaces import MultiDiscreteSwitchObsSpace
from switchfl.switch_agents import _Switch
from switchfl.utils.logging import format_logger
from switchfl.utils.naming import name2switch_id, symmetric_string, get_node_id_on_port_id
from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent as TrainAgent
import numpy as np

def compute_delay(rail_env: RailEnv, train: TrainAgent, position, direction):
    """
    Returns the delay of the given train

    Args:
        train_id (int): The id of the train

    Returns:
        int: The delay of the given train
    """
    # row, col = (
    #     train.position if train.position is not None else train.initial_position
    # )

    row, col = position

    min_dist_to_target = rail_env.distance_map.get()[
        train.handle, row, col, direction
    ]

    delay = rail_env._elapsed_steps - train.latest_arrival + min_dist_to_target
    # assume you are moving with max speed=1
    return delay

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

    def observe(self, agent, env, rail_network: RailNetwork) -> Tuple[ndarray, Dict[str, Any]]:
        self.logger.debug(symmetric_string(f"obs {agent}", frame="~"))
        node_id = name2switch_id(agent)
        switch: _Switch = rail_network.get_switch_on_position(node_id)

        semaphore = []
        target = []
        delay = []
        # arrived_trains = []

        train_at_ports = rail_network._train2next_port

        # if node_id == (7,5):
        #     print("Eccolo")

        # logging_var = {k: t.handle for k, t in train_at_ports.items()}
        # self.logger.debug(f"train at ports {logging_var}")
        self.logger.debug(f"next port for each train: {rail_network._train2next_port}")
        self.logger.debug(f"current agent: {agent} with position {switch.id}")
        train_counter = 0  # debugging
        active_train_handle = env.active_train
        train = env.rail_env.agents[active_train_handle]
        # node_semaphore_dict = {get_node_id_on_port_id(p): p for p in rail_network.semaphores.keys()}

        
        for port in switch.get_port_nodes():  # Get port IDs of this node
            self.logger.debug(f"port {port} coming from cell {switch.switch_graph.nodes.data('rail_prev_node')[port]}")
            # semaphore.append(switch.semaphores[port])

            # for source, target_port in switch.action_outcomes:
            #     if source == port:
            #         _, next_port = rail_network.get_neighbor_switch(target_port)
            #         if next_port in rail_network.semaphores:
            #             if rail_network.semaphores[next_port] != train.handle:
            #                 semaphore.append(0)
            #                 self.logger.debug(f"Semaphore ON -> next port {next_port} is blocked")
            #             else:
            #                 semaphore.append(1)
            #         else:
            #             semaphore.append(1)

            _, next_port = rail_network.get_neighbor_switch(port)

            # next_node = get_node_id_on_port_id(next_port)
            # if next_node in node_semaphore_dict.keys():
            #     if rail_network.semaphores[node_semaphore_dict[next_node]] != train.handle:
            #         semaphore.append(0)
            #     else:
            #         semaphore.append(1)
            # else:
            #     semaphore.append(1)
            first_blocked = False
            if port in rail_network.semaphores.keys():
                if rail_network.semaphores[port] != train.handle:
                    semaphore.append(0)
                    first_blocked = True

            if not first_blocked:
                if next_port in rail_network.semaphores.keys():
                    if rail_network.semaphores[next_port] != train.handle:
                        semaphore.append(0)
                    else:
                        semaphore.append(1)
                else:
                    semaphore.append(1)
 

            self.logger.debug(f"Semaphore: {semaphore}")

            if train_at_ports[active_train_handle] == port:
                current_port = port
                self.logger.debug(f"train at port {port}")
                delay.append(
                    self._discretize_delay(train, compute_delay(env.rail_env, train, train.position, train.direction))
                )
                target.extend(train.target)
                train_counter += 1
            else:
                delay.append(-1)
                target.extend([-1, -1])  # 2D coordinates

        if (
            train_counter == 0
        ):  

            # a = rail_env.render()
            # fig, ax = plt.subplots(figsize=(8,8))
            # plt.imshow(a)
            # ax.set_xticks(np.arange(0, a.shape[0], a.shape[0]/18), minor=False)
            # ax.set_yticks(np.arange(0, a.shape[0], a.shape[0]/18), minor=False)
            # ax.xaxis.grid(True, which='major', color='black', linestyle='--')
            # ax.yaxis.grid(True, which='major', color='black', linestyle='--')
            # ax.set_xticklabels(np.arange(18))
            # ax.set_yticklabels(np.arange(18))
            # plt.show(block=False)


            self.logger.fatal       (
                "Bug detected. No train detected at active switch. Check information in RailNetwork."
            )


        # for train in env.train_done:
        #     if train == '__all__':
        #         continue
        #     if env.train_done[train] == True:
        #         if train not in arrived_trains:
        #             arrived_trains.append(train)

        semaphore = np.array(semaphore).astype(int)
        target = np.array(target).astype(int)
        delay = np.array(delay).astype(int)
        observation = np.concatenate([node_id, semaphore, target, delay], dtype=np.int64)
        info = {"action_mask": switch.get_action_mask(current_port, semaphore), "active_train": active_train_handle} #, "arrived_trains" : arrived_trains}
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
