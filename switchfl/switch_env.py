import functools
import logging
from collections import OrderedDict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.step_utils.env_utils import apply_action_independent
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import AgentRenderVariant
from pettingzoo import AECEnv

from switchfl import NodeId, TrainAgentHandle
from switchfl.observer import StandardObserver, _Observer
from switchfl.rail_network import RailNetwork
from switchfl.utils.naming import name2switch_id, switch_id2name


class _SwitchEnv:
    metadata = {"render_mode": ["human", "rgb_array", None]}

    def __init__(
        self,
        rail_env: RailEnv,
        max_steps: int = 200,
        render_mode: str = None,
        observer: _Observer = None,
        seed: int = None,
    ):
        super().__init__()
        self.rail_env = rail_env
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.seed = seed

        self.rail_network = RailNetwork(rail_env)

        self.observer = observer if observer is not None else StandardObserver()
        self.possible_agents = self.rail_network.get_switch_names()
        self.agents = self.possible_agents

        self.active_switch_agents = []

        self.terminated: bool
        self.truncated: bool
        self.train_action_plan: Dict[TrainAgentHandle, List[RailEnvActions]]

        self.rail_env_time: int
        self.train_done: Dict[TrainAgentHandle, bool]
        self.train_obs: Dict[TrainAgentHandle, Any]
        self.train_reward: Dict[TrainAgentHandle, float]
        self.train_info: Dict[TrainAgentHandle, Any]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_space = self.observer.get_observation_space(
            agent=agent,
            rail_env=self.rail_env,
            rail_network=self.rail_network,
            seed=self.seed,
        )
        return obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        node_id = name2switch_id(agent)
        action_space = self.rail_network.get_switch_action_space(
            node_id, seed=self.seed
        )
        return action_space

    def reset(self, seed=None, options=None):
        obs, info = self.rail_env.reset()
        self.rail_network.reset()

        self.terminated = False
        self.truncated = False

        self.terminations = {switch_name: False for switch_name in self.agents}
        self.truncations = {switch_name: False for switch_name in self.agents}
        self.rewards = {switch_name: 0 for switch_name in self.agents}
        self._cumulative_rewards = {switch_name: 0 for switch_name in self.agents}
        self.infos = {switch_name: {} for switch_name in self.agents}
        self.step_counter = {switch_name: 0 for switch_name in self.agents}

        self.train_action_plan = {train.handle: [] for train in self.rail_env.agents}

        self.rail_env_time = 0
        self.train_done = {train.handle: False for train in self.rail_env.agents}
        self.train_obs = {train.handle: None for train in self.rail_env.agents}
        self.train_reward = {train.handle: 0.0 for train in self.rail_env.agents}
        self.train_info = {train.handle: None for train in self.rail_env.agents}

        self._move_trains_to_switch()
        self._init_semaphores()

    def render(self):
        if self.render_mode is None:
            return

        rgb = self.rail_env.render(
            agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS, show_debug=True
        )
        if self.render_mode == "rgb_array":
            return rgb
        elif self.render_mode == "human":
            fig, (ax1, ax2) = plt.subplots(ncols=2)
            ax1.imshow(rgb)
            nx.draw(
                self.rail_network.rail_graph.to_undirected(),
                self.rail_network.rail_graph.nodes.data("position"),
                with_labels=True,
                node_color=dict(
                    self.rail_network.rail_graph.nodes.data(data="node_color")
                ).values(),
                edge_color="gray",
                node_size=3,
                font_size=5,
                ax=ax2,
            )
            ax1.axis("off")
            ax2.axis("off")
            plt.show()

    def obs2human(self, agent: str, observation: Any) -> Dict[str, Any]:
        obs_space = self.observation_space(agent)
        human_format = getattr(obs_space, "human_format", None)
        if not callable(human_format):
            logging.error(
                f"Observation space: {obs_space} has not method to convert observation into a human format."
            )
            return
        return human_format(observation)

    def obs2json(self, agent: str, observation: Any) -> Dict[str, Any]:
        obs = self.obs2human(agent, observation)
        obs = {k: v.tolist() for k, v in obs.items()}
        return obs

    def _apply_action(self, agent_selection: str, action: int) -> NodeId:
        """get train actions and update semaphores at the corresponding switches

        Args:
            agent_selection (str): which switch agent is performing the action
            action (int): which actions is performed

        Returns:
            NodeId: to which switch the action is sending the train
        """
        assert self.action_space(agent_selection).contains(
            action
        ), "Invalid action performed"

        node_id = name2switch_id(agent_selection)
        current_switch = self.rail_network.get_switch_on_position(node_id)
        in_port, out_port = current_switch.action_outcomes[action]
        
        # update train_action_plan such that move_trains step can work it down
        moving_train, train_actions = self.rail_network.get_train_actions(
            node_id, action, self.rail_env.agents
        )
        for train_agent_handle in train_actions.keys():
            self.train_action_plan[train_agent_handle].extend(
                train_actions[train_agent_handle]
            )
        
        # transition a train if there is actually a train moving
        if isinstance(moving_train, TrainAgent):
            # update rail_network how trains are transitioned from edge to edge
            next_switch = self.rail_network.transition_train(moving_train, in_port, out_port)
        else:
            next_switch = current_switch

        return next_switch.id

    def _move_trains(self):
        # NOTE: the time when the train departures is already taken into account
        action = {}
        for train in self.rail_env.agents:
            handle = train.handle
            if self.train_done[handle]:
                # no action for already done trains
                continue

            if len(self.train_action_plan[handle]) == 0:
                # use base action
                action[handle] = RailEnvActions.MOVE_FORWARD
            else:
                # use predetermined action
                action[handle] = self.train_action_plan[handle].pop(0)

        # do rail env step
        self.train_obs, self.train_reward, self.train_done, self.train_info = (
            self.rail_env.step(action)
        )
        self.rail_env_time += 1

        # check for all trains being done
        if self.train_done["__all__"]:
            self.terminations = {
                switch_agent: True for switch_agent in self.terminations.keys()
            }
            self.terminated = True

    def _move_trains_to_switch(self):
        while len(self.active_switch_agents) == 0 and not self.terminated:
            # NOTE: after resetting the environment all trains are in a standstill
            # -> they have to be moved first after the reset
            self._move_trains()
            self._check_active_switch()

        # remove duplicates in agents but maintaining order
        print("active_switches: ", self.active_switch_agents)
        self.active_switch_agents = list(
            OrderedDict.fromkeys(self.active_switch_agents)
        )

    def _check_active_switch(self):
        """do simulation step and see if a train enters a switch node
        -> then add the switch to active agents
        """

        # check if a train is ready to depart -> its position on the grid is known
        current_positions = [
            train.position
            for train in self.rail_env.agents
            if train.position is not None
        ]
        if len(current_positions) == 0:
            # if the train is not on the grid
            return

        for train in self.rail_env.agents:
            # train is not one the grid yet or if it waiting don't execute something on it.
            if train.position is None or train.state == TrainState.WAITING:
                continue

            # get first next action
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

            # if with the next action the train has entered a switch add the switch to the active switches
            if self.rail_network.get_switch_on_position(new_pos) is not None and (
                train.state == TrainState.READY_TO_DEPART
                or train.state == TrainState.MOVING
            ):
                # use new pos because the switch coordinates are its node_id
                switch_id = switch_id2name(new_pos)
                self.active_switch_agents.append(switch_id)

    def _init_semaphores(self):
        """executed in reset()

        finds all train positions and setups all semaphores for all switches.

        Assume:
        - all trains are already on the grid
        - at least one train is in front of a switch
        """
        for train in self.rail_env.agents:
            print("train_handle: ", train.handle)
            # simulate steps of a train until they arrive at a switch
            # NOTE: yes this is expensive, but only executed once in reset()
            current_pos = train.position
            current_dir = train.direction
            last_pos = train.old_position
            last_dir = train.old_direction
            if current_pos is None or current_dir is None:
                current_pos = train.initial_position
                current_dir = train.initial_direction

            while True:
                if self.rail_network.get_switch_on_position(current_pos) is not None:
                    print(
                        "\tnext_switch:",
                        self.rail_network.get_switch_on_position(current_pos).id,
                    )
                    # train on switch
                    break
                last_pos = current_pos
                last_dir = current_dir
                current_pos, current_dir = apply_action_independent(
                    RailEnvActions.MOVE_FORWARD,
                    self.rail_env.rail,
                    current_pos,
                    current_dir,
                )
            # last pos corresponds to rail_prev_node
            # NOTE: getting the port based on position and direction could be bugged
            # if the first switch directly is directly behind a turn in the rail
            port = self.rail_network.get_port_on_position(last_pos, last_dir)
            print("\tarriving_port: ", port)
            self.rail_network.block_semaphore(port)
            self.rail_network.set_trains_next_port(train, port)

    @property
    def n_steps(self) -> int:
        """accumulated steps from each agents

        Returns:
            int: _description_
        """
        return sum(self.step_counter.values())


class ASyncSwitchEnv(_SwitchEnv, AECEnv):
    def __init__(
        self, rail_env, max_steps=200, render_mode=None, observer=None, seed=None
    ):
        super().__init__(rail_env, max_steps, render_mode, observer, seed)

    def agent_iter(self, max_iter=2**63):
        while not (self.terminated or self.truncated):
            self.agent_selection = self.active_switch_agents.pop(0)
            yield self.agent_selection

    def step(self, action) -> Dict[str, Any]:
        # check if current agent is still operating
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
            or action is None
        ):
            # the agent is done
            return {}

        next_switch = self._apply_action(self.agent_selection, action)

        # no switches with non processed trains left -> time to move trains
        if len(self.active_switch_agents) == 0:
            self._move_trains_to_switch()

        # check for done episode
        self.step_counter[self.agent_selection] += 1
        if self.n_steps > self.max_steps:
            self.truncations = {
                switch_agent: True for switch_agent in self.truncations.keys()
            }
            self.truncated = True

        # prepare info dict
        post_step_info = {"next_switch": next_switch}
        return post_step_info

    def observe(self, agent) -> np.ndarray:
        obs, info = self.observer.observe(
            agent=agent,
            rail_env=self.rail_env,
            rail_network=self.rail_network,
        )
        self.infos[agent].update(info)
        return obs

    def close(self):
        return super().close()
