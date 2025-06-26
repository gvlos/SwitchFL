import functools
from collections import OrderedDict
from typing import Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.step_utils.env_utils import apply_action_independent
from flatland.utils.rendertools import AgentRenderVariant
from gymnasium import spaces
from pettingzoo import AECEnv

from environment.switch_agents import _SwitchAgent
from environment.utils import (
    create_rail_graph,
    generate_local_switch_graphs,
    get_switch_id,
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

        self.graph = self._create_switch_graph(self.rail_env)

        pos = self.graph.nodes.data(data="pos")
        pos = {k: v for k, v in pos}

        fig, ax = plt.subplots()
        ax.scatter(1, 1)
        ax.text(1, 2, "N")
        ax.text(1, 0, "S")
        ax.text(2, 1, "E")
        ax.text(0, 1, "W")
        nx.draw(
            self.graph,
            pos,
            # connectionstyle="arc3,rad=0.1",
            with_labels=False,
            node_color=dict(self.graph.nodes.data(data="node_color")).values(),
            edge_color="gray",
            node_size=3,
            # arrowsize=3
            ax=ax,
        )

        self._node_df = self.get_node_df(self.graph)

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
            subgraph = self.graph.subgraph(group.index)
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
        return pd.DataFrame(
            [attr for _, attr in graph.nodes.data()],
            index=[idx for idx, _ in graph.nodes.data()],
        )

    def _create_switch_graph(self, env: RailEnv) -> nx.Graph:
        graph = create_rail_graph(env)
        graph = insert_switch_proximity_nodes(graph)
        graph = prune_non_switches(graph)
        graph = generate_local_switch_graphs(graph)
        return graph

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
        print(action)
        self.rail_env.step(action)

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
                self.agents.append(get_switch_id(new_pos))

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

        while len(self.agents) == 0:
            self.move_trains()
            # move trains until they arrive on the grid
            current_positions = [train.position for train in self.rail_env.agents]
            current_positions = list(filter(lambda x: x is not None, current_positions))
            if len(current_positions) == 0:
                continue

            self._check_active_switch()

        # remove duplicates in agents but maintaining order
        self.agents = list(OrderedDict.fromkeys(self.agents))
        print(self.agents)

    def apply_action(self, switch: _SwitchAgent, action: int):
        rail_env_actions = switch.get_train_action(action, self.rail_env.agents)
        # update train_action_plan such that _do_rail_env step can work it down
        for train_agent_handle in rail_env_actions.keys():
            self.train_action_plan[train_agent_handle].extend(
                rail_env_actions[train_agent_handle]
            )

    def agent_iter(self, max_iter=2**63):
        while not (self.terminated or self.truncated):
            self.agent_selection = self.agents.pop(0)
            yield self.agent_selection

    def reset(self, seed=None, options=None):
        if options is None:
            options = {}
        self.rail_env.reset(random_seed=seed, **options)
        self.move_trains_to_switch()

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
        self.step_counter[self.agent_selection] += 1

    def observe(self, agent):
        return self.observations[agent]

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
