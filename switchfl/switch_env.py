import functools
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from flatland.core.grid.grid4 import Grid4TransitionsEnum
from flatland.envs.agent_utils import EnvAgent as TrainAgent
from flatland.envs.rail_env import RailEnv, RailEnvActions
from flatland.envs.step_utils.states import TrainState
from flatland.utils.rendertools import AgentRenderVariant
from pettingzoo import AECEnv
from switchfl import NodeId, TrainAgentHandle
from switchfl.observer import StandardObserver, _Observer
from switchfl.rail_network import RailNetwork
from switchfl.utils.logging import format_logger
from switchfl.utils.naming import name2switch_id, switch_id2name, symmetric_string


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
        self.logger = logging.getLogger(type(self).__name__)
        self.logger = format_logger(self.logger)

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
            self.logger.error(
                f"Observation space: {obs_space} has not method to convert observation into a human format."
            )
            return
        return human_format(observation)

    def obs2json(self, agent: str, observation: Any) -> Dict[str, Any]:
        obs = self.obs2human(agent, observation)
        obs = {k: v.tolist() for k, v in obs.items()}
        return obs

    def _apply_action(self, agent_selection: str, action: int) -> NodeId:
        """get train actions and update semaphores at the corresponding switches for a SINGLE train transitioning a switch

        Args:
            agent_selection (str): which switch agent is performing the action
            action (int): which actions is performed

        Returns:
            NodeId: to which switch the action is sending the train
        """
        assert self.action_space(agent_selection).contains(
            action
        ), f"Invalid action performed. Allowed action space: {self.action_space(agent_selection)}"

        self.logger.debug(symmetric_string(agent_selection))
        self.logger.debug(f"rail time: {self.rail_env_time}")

        node_id = name2switch_id(agent_selection)
        current_switch = self.rail_network.get_switch_on_position(node_id)
        # in and out port of the traversing switch
        in_port, out_port = current_switch.action_outcomes[action]

        # update train_action_plan such that move_trains step can work it down
        moving_train, train_actions = self.rail_network.get_train_actions(
            node_id, action, self.rail_env.agents
        )
        # transition a train if there is actually a train moving
        if isinstance(moving_train, TrainAgent):
            # update rail_network how trains are transitioned from edge to edge
            next_switch, next_port = self.rail_network.transition_train(
                moving_train, in_port, out_port
            )
        else:
            next_switch = current_switch
            next_port = None

        self.logger.debug(f"existing plan: {self.train_action_plan}")
        self.logger.debug(f"new actions: {train_actions}")
        for train_agent_handle in train_actions.keys():
            # adapt next train actions
            # NOTE: if two switches are direct neighbors, the first given train action for the moving train
            # has to be replaced by the last action of the train_action plan because otherwise the environment
            # will send the train forward which messes up the scheduling and planning
            next_train_actions = train_actions[train_agent_handle]
            if (
                moving_train is not None
                and self.rail_network.get_port_distance(out_port, next_port) == 0
                and len(self.train_action_plan[train_agent_handle]) > 0
            ):
                # cut away the MOVE_FORWARD action of the new plan, because
                # moving into the new switch is done by the action from the previous switch
                next_train_actions.pop(0)
                self.train_action_plan[train_agent_handle].extend(next_train_actions)

            # moving train is None -> stop moving command (stop the train before it enters the next switch)
            # but retain information about how the train will enter the next switch
            # example: move left leads directly on another switch
            elif (
                moving_train is None
                and len(self.train_action_plan[train_agent_handle]) > 0
            ):
                stop_action = next_train_actions[0]
                assert stop_action == RailEnvActions.STOP_MOVING
                self.train_action_plan[train_agent_handle].insert(0, stop_action)
            else:
                self.train_action_plan[train_agent_handle].extend(next_train_actions)

        self.logger.debug(f"updated_plan: {self.train_action_plan}")
        return next_switch.id

    def _move_trains(self):
        """Move all train agents in the RailEnv by one step
        Also take into account if there is a next action, pre-computed for a train to transit the switch
        """
        # NOTE: the time when the train departures is already taken into account
        train_actions = {}
        for train in self.rail_env.agents:
            handle = train.handle
            if self.train_done[handle]:
                # no action for already done trains
                continue

            if len(self.train_action_plan[handle]) == 0:
                # use base action
                train_actions[handle] = RailEnvActions.MOVE_FORWARD
            else:
                # use predetermined action
                train_actions[handle] = self.train_action_plan[handle].pop(0)

        # do rail env step
        (
            self.train_obs,
            self.train_reward,
            self.train_done,
            self.train_info,
        ) = self.rail_env.step(train_actions)
        self.rail_env_time += 1
        self._check_action_execution()

        # check for all trains being done
        if self.train_done["__all__"]:
            self.terminations = {
                switch_agent: True for switch_agent in self.terminations.keys()
            }
            self.terminated = True

    def _move_trains_to_switch(self):
        self.logger.debug(symmetric_string("move trains", 80, "-"))
        while len(self.active_switch_agents) == 0 and not self.terminated:
            # NOTE: after resetting the environment all trains are in a standstill
            # -> they have to be moved first after the reset
            self._move_trains()
            self._check_active_switch()

        # remove duplicates in agents but maintaining order
        train_positions = {
            t.handle: (t.position, Grid4TransitionsEnum(t.direction).name)
            for t in self.rail_env.agents
        }
        self.logger.debug(f"Train pos: {train_positions}")
        self.logger.debug(f"active_switches: {self.active_switch_agents}")
        self.active_switch_agents = list(
            OrderedDict.fromkeys(self.active_switch_agents)
        )

    def _check_active_switch(self):
        """do simulation step and see if a train enters a switch node
        -> then add the switch to active agents
        """
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
            (
                new_cell_valid,
                new_direction,
                new_position,
                transition_valid,
                preprocessed_action,
            ) = self.rail_env.rail.check_action_on_agent(
                next_action,
                train.position,
                train.direction,
            )

            # if with the next action the train has entered a switch add the switch to the active switches
            next_switch = self.rail_network.get_switch_on_position(new_position)
            self.logger.debug(
                f"Next switch for train {train.handle} ({train.state.name, train.position}): {next_switch}"
            )
            if next_switch is not None and (
                train.state == TrainState.READY_TO_DEPART
                or train.state == TrainState.MOVING
            ):
                # use new pos because the switch coordinates are its node_id
                switch_id = switch_id2name(new_position)
                self.active_switch_agents.append(switch_id)

    def _check_action_execution(self):
        """checks if the action has been executed successfully

        This function will log an error if a train deviates from its planned path.
        """
        for train in self.rail_env.agents:
            next_port = self.rail_network.get_trains_next_port(train)
            prev_port = self.rail_network.get_trains_prev_port(train)
            if next_port is None or prev_port is None:
                # train is not at a switch or has no next port planned
                continue
            source_node = self.rail_network.get_switch_on_port(prev_port).id
            target_node = self.rail_network.get_switch_on_port(next_port).id
            rail_pieces = self.rail_network.get_rail_pieces_between_ports(
                source_node, target_node
            )
            if train.position not in [*rail_pieces, source_node, target_node]:
                self.logger.error(f"Train {train.handle} deviated from planned path!")

    def _init_semaphores(self):
        """executed in reset()

        finds all train positions and setups all semaphores for all switches.

        Assume:
        - all trains are already on the grid
        - at least one train is in front of a switch
        """
        for train in self.rail_env.agents:
            # simulate steps of a train until they arrive at a switch
            # NOTE: yes this is expensive, but only executed once in reset()
            current_position = train.position
            current_direction = train.direction
            last_pos = train.old_position
            last_dir = train.old_direction
            if current_position is None or current_direction is None:
                current_position = train.initial_position
                current_direction = train.initial_direction

            while True:
                if (            
                    self.rail_network.get_switch_on_position(current_position)
                    is not None
                ):
                    # train on switch
                    break
                last_pos = current_position
                last_dir = current_direction
                (
                    new_cell_valid,
                    current_direction,
                    current_position,
                    transition_valid,
                    preprocessed_action,
                ) = self.rail_env.rail.check_action_on_agent(
                    RailEnvActions.MOVE_FORWARD,
                    current_position,
                    current_direction,
                )

            # last pos corresponds to rail_prev_node
            # NOTE: getting the port based on position and direction could be bugged
            # if the first switch directly is directly behind a turn in the rail
            port = self.rail_network.get_port_on_position(last_pos, last_dir)
            self.rail_network.block_semaphore(port)
            self.rail_network.set_trains_next_port(train, port)

    @property
    def n_steps(self) -> int:
        """accumulated steps from each agents

        Returns:
            int: _description_
        """
        return sum(self.step_counter.values())

    def get_env_plan(self, path: Path = None):
        """Extract information about each switch as a csv from the environment:
        This csv should contain:
            - switch_id
            - action (int)
            - rail env actions (move forward, move_right, move_left, stop) with their corresponding ports
            - from where to where an action is sending a train
        """
        env_plan = []
        for _, switch in self.rail_network.switches:
            for action_idx in range(switch.n_actions):
                env_plan.append(
                    self.rail_network.get_switch_transition_info(
                        switch.id, action=action_idx
                    )
                )

        if path is not None:
            df = pd.DataFrame(env_plan)
            df.to_csv(path, index=False, sep=";")
            self.logger.info(f"Saved environment plan to {path}")
            return
        else:
            return env_plan


class ASyncSwitchEnv(_SwitchEnv, AECEnv):
    def __init__(
        self,
        rail_env,
        max_steps=200,
        render_mode=None,
        observer=None,
        seed=None,
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
