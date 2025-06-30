import functools
import logging
from collections import OrderedDict
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from flatland.envs.rail_env import EnvAgent as TrainAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.step_utils.env_utils import apply_action_independent
from flatland.utils.rendertools import AgentRenderVariant
from gymnasium import spaces
from pettingzoo import AECEnv

from environment.switch_agents import _SwitchAgent
from environment.utils.naming import get_switch_id
from environment.utils.rail_graph import (
    create_rail_graph,
    generate_local_switch_graphs,
    insert_switch_proximity_nodes,
    prune_non_switches,
)


class SwitchEnv(AECEnv):
    def __init__(self, rail_env: RailEnv, max_steps: int = 200, seed: int = None):
        super().__init__()
        self.rail_env = rail_env
        self.rail_env.reset()

        self.max_steps = max_steps
        self.seed = seed

        self.switch_network_graph = self._create_switch_graph(self.rail_env)

        pos = self.switch_network_graph.nodes.data(data="pos")
        pos = {k: v for k, v in pos}

        self._node_df = self.get_node_df(self.switch_network_graph)

        # switch agents
        self.possible_agents = [
            get_switch_id(idx) for idx in self._node_df["switch_id"].unique()
        ]
        # all active switch agents
        self.agents = []

        self.train_action_plan = {k.handle: [] for k in self.rail_env.agents}
        self.terminations = {k: False for k in self.possible_agents}
        self.truncations = {k: False for k in self.possible_agents}
        self.rewards = {k: 0 for k in self.possible_agents}
        self._cumulative_rewards = {k: 0 for k in self.possible_agents}
        self.infos = {k: {} for k in self.possible_agents}
        self.step_counter = {k: 0 for k in self.possible_agents}
        self.terminated = False
        self.truncated = False

        self.train_obs = {train.handle: None for train in self.rail_env.agents}
        self.train_reward = {train.handle: 0 for train in self.rail_env.agents}
        self.train_done = {train.handle: False for train in self.rail_env.agents}
        self.train_info = {train.handle: None for train in self.rail_env.agents}

        self.switch_agents = self.build_agent_abstraction(
            self._node_df, self.possible_agents
        )
        self.observation_spaces = self.build_observation_spaces(seed=self.seed)
        self.action_spaces = self.build_action_spaces(seed=self.seed)

        # faster than the pandas DataFrame
        self._switch_signal_map = dict(
            zip(self._node_df["rail_prev_node"], self._node_df.index)
        )
        self._switch_id_switch_pos_map = dict(
            zip(self._node_df.index, self._node_df["switch_pos"])
        )

    def build_agent_abstraction(
        self, node_df: pd.DataFrame, agent_ids: List[str]
    ) -> Dict[str, _SwitchAgent]:
        agents = {}
        for switch_id, group in node_df.groupby("switch_id"):
            subgraph = self.switch_network_graph.subgraph(group.index)
            switch_agent = _SwitchAgent.from_switch_graph(
                subgraph, self.rail_env.rail_generator.max_num_cities
            )
            agents[get_switch_id(switch_id)] = switch_agent
        return agents

    def build_observation_spaces(self, seed: int = None) -> Dict[str, spaces.Space]:
        obs_spaces = {}
        for agent_id, agent in self.switch_agents.items():
            agent: _SwitchAgent
            obs_spaces[agent_id] = agent.get_observation_space(seed)
        return obs_spaces

    def build_action_spaces(self, seed: int = None) -> Dict[str, spaces.Space]:
        act_spaces = {}
        for agent_id, agent in self.switch_agents.items():
            agent: _SwitchAgent
            act_spaces[agent_id] = agent.get_action_space(seed)
        return act_spaces

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_node_df(self, graph: nx.Graph) -> pd.DataFrame:
        df = pd.DataFrame(
            [attr for _, attr in graph.nodes.data()],
            index=[idx for idx, _ in graph.nodes.data()],
        )
        # build intra node_df
        df["intra_switch_index"] = df.index.to_series().apply(
            lambda x: int(str(x[0])[-1])
        )

        return df

    def _create_switch_graph(self, env: RailEnv) -> nx.Graph:
        graph = create_rail_graph(env)
        graph = insert_switch_proximity_nodes(graph)
        graph = prune_non_switches(graph)
        graph = generate_local_switch_graphs(graph)
        return graph

    def _build_semaphores(self):
        """executed in reset()

        finds all train positions and setups all semaphores for all switches.

        Assume:
        - all trains are already on the grid
        - at least one train is in front of a switch
        """

        for train in self.rail_env.agents:
            # simulate steps of a train until they arrive at a switch
            # NOTE: yes this is expensive, but only executed once in reset()
            current_pos = train.position
            current_dir = train.direction
            while True:
                if sum(self._node_df["switch_pos"] == current_pos):
                    # train on switch
                    break
                last_pos = current_pos
                current_pos, current_dir = apply_action_independent(
                    RailEnvActions.MOVE_FORWARD,
                    self.rail_env.rail,
                    current_pos,
                    current_dir,
                )
            # last pos corresponds to rail_prev_node
            switch_id = self._node_df[
                (self._node_df["rail_prev_node"] == last_pos)
                & (self._node_df["switch_pos"] == current_pos)
            ]["switch_id"]
            self.switch_agents[get_switch_id(switch_id.item())].semaphores[
                switch_id.index.item()
            ] = True

    def _compute_delay(self, train: TrainAgent):
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
        min_dist_to_target = self.rail_env.distance_map.get()[
            train.handle, row, col, train.direction
        ]
        # assume you are moving with max speed=1
        return self.rail_env._elapsed_steps - train.latest_arrival + min_dist_to_target

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

    def move_trains(self, action: Dict[int, RailEnvActions] = None):
        base_action = {
            k.handle: RailEnvActions.MOVE_FORWARD for k in self.rail_env.agents
        }

        for train_agent_handle in base_action.keys():
            if len(self.train_action_plan[train_agent_handle]) == 0 or (
                action is not None and train_agent_handle in action.keys()
            ):  # manual overwrite
                continue
            base_action[train_agent_handle] = self.train_action_plan[
                train_agent_handle
            ].pop(0)

        if action is None:
            action = base_action
        else:
            base_action.update(action)
            action = base_action
        self.train_obs, self.train_reward, self.train_done, self.train_info = (
            self.rail_env.step(action)
        )

        # check for all trains being done
        if self.train_done["__all__"]:
            self.terminations = {
                switch_agent: True for switch_agent in self.terminations.keys()
            }
            self.terminated = True

    def _check_active_switch(self):
        # do simulation step and see if a train enters a switch node -> then add the switch to active agents
        for train in self.rail_env.agents:
            if train.position is None:
                continue

            # get next action
            if len(self.train_action_plan[train.handle]) > 0:
                next_action = self.train_action_plan[train.handle][0]
            else:
                next_action = RailEnvActions.MOVE_FORWARD

            # do a simulation step an get the next switch if needed
            new_pos, _ = apply_action_independent(
                next_action,
                self.rail_env.rail,
                train.position,
                train.direction,
            )

            if (self._node_df["switch_pos"] == new_pos).sum():
                switch_id = get_switch_id(new_pos)
                self.agents.append(switch_id)

    def move_trains_to_switch(self):

        # Check active switch (0)
        # Loop:
        #     time t
        #     Train step (t)
        #     Check active switch (t+1)
        #
        # compute switch action (t)
        # perform action (t)

        # self._check_active_switch()

        while len(self.agents) == 0 and not self.terminated:
            # NOTE: after resetting the environment all trains are in a standstill
            # -> they have to be moved first after the reset
            self.move_trains()

            # move trains until they arrive on the grid
            current_positions = [train.position for train in self.rail_env.agents]
            current_positions = list(filter(lambda x: x is not None, current_positions))
            if len(current_positions) == 0:
                continue

            self._check_active_switch()

        # remove duplicates in agents but maintaining order
        self.agents = list(OrderedDict.fromkeys(self.agents))

    def apply_action(self, switch: _SwitchAgent, action: int):
        logging.debug("______")
        logging.debug(f"{action=}")
        rail_env_actions = switch.get_train_action(action, self.rail_env.agents)
        logging.debug(rail_env_actions)
        # transition rail network graph

        # 1. free semaphore on current switch
        source, target = switch.outcomes[action]
        logging.debug(source, target)
        logging.debug(switch.semaphores)
        switch.free_port(source)
        # 2. find next switch (account for simple intersection)
        neighbors = self.switch_network_graph.neighbors(target)
        next_switch = self._node_df[
            self._node_df.index.isin(neighbors)
            & (self._node_df["switch_id"] != switch.id)
        ]
        next_switch_id = get_switch_id(next_switch["switch_id"].item())
        # 3. activate semaphore on next switch
        port = next_switch.index.item()
        self.switch_agents[next_switch_id].block_port(port)
        logging.debug(self.switch_agents[next_switch_id].semaphores)

        # update train_action_plan such that _do_rail_env step can work it down
        for train_agent_handle in rail_env_actions.keys():
            self.train_action_plan[train_agent_handle].extend(
                rail_env_actions[train_agent_handle]
            )
        logging.debug(self.train_action_plan)

    def agent_iter(self, max_iter=2**63):
        while not (self.terminated or self.truncated):
            self.agent_selection = self.agents.pop(0)
            yield self.agent_selection

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        self.rail_env.reset(random_seed=seed, **options)
        self.move_trains_to_switch()
        self._build_semaphores()

        self.rewards = {agent: 0 for agent in self.possible_agents}
        self._cumulative_rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self.observations = {agent: None for agent in self.possible_agents}

        self.num_moves = 0
        self.terminated = False
        self.truncated = False

    def step(self, action):
        self.apply_action(self.switch_agents[self.agent_selection], action)
        if len(self.agents) == 0:
            self.move_trains_to_switch()
            plt.imshow(self.render())
            plt.axis("off")
            plt.show()

        self.step_counter[self.agent_selection] += 1
        if self.n_steps > self.max_steps:
            self.truncations = {
                switch_agent: True for switch_agent in self.truncations.keys()
            }
            self.truncated = True

    def observe(self, agent):
        # get current observation

        # sort switch nodes according to first decimal -> use node_df
        switch_id = tuple(
            map(lambda x: int(x), agent.split("_")[1].strip("()").split(", "))
        )
        df = self._node_df[self._node_df["switch_id"] == switch_id][
            "intra_switch_index"
        ]
        df = df.reset_index(0).set_index("intra_switch_index")

        semaphore = []
        target = []
        delay = []
        train_at_ports = self.switch_agents[switch_id].map_train_to_port(
            self.rail_env.agents
        )
        for node in df["index"]:
            semaphore.append(self.switch_agents[switch_id].semaphores[node])

            train = train_at_ports[node]
            if train is None:
                delay.append(0)
                target.append(0)
            else:
                delay.append(self._discretize_delay(train, self._compute_delay(train)))
                target.append(train.target)
        self.observations[agent] = np.array([*semaphore, *delay, *target])
        self.rewards[agent] = 0  # TODO: still open issue
        self.infos[agent].update(
            {"action_mask": self.switch_agents[agent].get_action_mask()}
        )

    def render(self):
        return self.rail_env.render(
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS, show_debug=True
        )

    def close(self):
        return super().close()

    @property
    def n_steps(self) -> int:
        """accumulated steps from each agents

        Returns:
            int: _description_
        """
        return sum(self.step_counter.values())
