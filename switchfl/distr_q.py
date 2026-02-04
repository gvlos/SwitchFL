import itertools
from .switch_env import ASyncSwitchEnv, name2switch_id, switch_id2name
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from flatland.utils.rendertools import AgentRenderVariant
from switchfl.utils.naming import get_node_id_on_port_id
import time

# flatland directions
# class Grid4TransitionsEnum(IntEnum):
#     NORTH = 0
#     EAST = 1
#     SOUTH = 2
#     WEST = 3

 # actions
# DO_NOTHING: 0
# MOVE_LEFT: 1
# MOVE_FORWARD: 2
# MOVE_RIGHT: 3
# STOP_MOVING: 4

# switchfl directions
# .1 = EAST
# .2 = NORTH
# .3 = WEST
# .4 = SOUTH

class DistrQLearning:
    """
    DistrQLearning is a class that implements a Distributed Q-learning agent.

    Parameters
    ----------
    gamma : float
        The discount factor.
    default_q : float
        The default value for the Q-table.
    epsilon : float
        The exploration rate.
    epsilon_decay_rate : float
        The decay rate for the exploration rate.
    lr : float
        The learning rate.
    lr_decay_rate : float
        The decay rate for the learning rate.
    seed : int
        The random seed. 
    """
    def __init__(self, env : ASyncSwitchEnv, gamma = 1., epsilon = 0.4, epsilon_decay_rate = 0., lr = 0.4, lr_decay_rate = 0., default_q = 0., seed = 450565):
        self.env = env  # Environment to interact with
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.epsilon = {agent: epsilon for agent in self.env.agents}
        self.epsilon_decay_rate = epsilon_decay_rate
        self.initial_lr = lr
        self.lr = {agent: lr for agent in self.env.agents}
        self.lr_decay_rate = lr_decay_rate
        self.default = [default_q]
        self.q_table = {}
        self.seed = seed
        self.optimal_init = 500.
        self.destination_bonus = 1000.

    def __check_entry(self, state, agent):
        """
        Checks if the state is in the Q-table and adds it if it is not.

        Parameters
        ----------
        state : int
            The state to check.
        """
        if tuple(state) not in self.q_table:
            self.q_table[tuple(state)] = self.default * self.env.action_space(agent).n

    def __decay_epsilon(self, agent, t):
        """
        Decays the epsilon value.

        Parameters
        ----------
        episode : int
            The episode number.
        """
        self.epsilon[agent] = self.initial_epsilon * (self.epsilon_decay_rate ** t)

    def __decay_lr(self, agent, t):
        """
        Decays the learning rate.

        Parameters
        ----------
        episode : int
            The episode number.
        """
        self.lr[agent] = self.initial_lr * (self.lr_decay_rate ** t)

    def __init_q_table(self):
        """
        Initializes the Q-table wiith optimistic values for shortest path actions.
        """
        for agent in self.env.rail_env.agents:
            shortest_path = self.env.rail_env.distance_map.get_shortest_paths(
                max_depth=None, agents = self.env.rail_env.agents, agent_handle=agent.handle)[agent.handle]

            for wp_idx in range(len(shortest_path)):
                wp = shortest_path[wp_idx]
                switch = self.env.rail_network.get_switch_on_position(wp.position)
                if switch is not None:
                    num_ports = len(switch.get_port_nodes())

                    direction = self.env.rail_network.map_inverse_direction(wp.direction)
                    in_port = tuple(pos + direction/10 for pos in wp.position)

                    semaphores = list(itertools.product([0, 1], repeat=num_ports))[1:]  # exclude all red

                    delays = []
                    targets = []

                    for i in range(3):
                        dl = []
                        for port in switch.get_port_nodes():
                            if port == in_port:
                                dl.append(i)
                            else:
                                dl.append(-1)
                        delays.append(tuple(dl))

                    targets = -1 * np.ones(num_ports*2)
                    
                    for i, port in enumerate(switch.get_port_nodes()):
                        if port == in_port:
                            targets[i*2] = agent.target[0]
                            targets[i*2+1] = agent.target[1]

                    prod = list(itertools.product(semaphores, [tuple(targets)], delays))
                    
                    for idx, ele in enumerate(prod):
                        prod[idx] = np.concatenate([wp.position,
                                                    np.array(ele[0]).astype(int),
                                                    np.array(ele[1]).astype(int),
                                                    np.array(ele[2]).astype(int)])

                    next_switch_found = False

                    next_wp_idx = 1
                    while next_wp_idx + wp_idx < len(shortest_path):
                        next_wp = shortest_path[next_wp_idx + wp_idx]
                        next_switch = self.env.rail_network.get_switch_on_position(next_wp.position)
                        if next_switch is not None:
                            next_switch_found = True
                            break
                        next_wp_idx += 1

                    min_distance = float('inf')

                    if next_switch_found:
    
                        for act_idx, action in enumerate(switch.action_outcomes):

                            if in_port == action[0]:
                                _, out_port = action
                                _, next_port = self.env.rail_network.get_neighbor_switch(out_port)

                                next_wp_dir = self.env.rail_network.map_inverse_direction(next_wp.direction)
                                next_port_wp = tuple(pos + next_wp_dir/10 for pos in next_wp.position)
                                if next_port == next_port_wp:
                                    distance = self.env.rail_network.get_port_distance(out_port, next_port)
                                    if distance < min_distance:
                                        optimal_action = act_idx
                                        min_distance = distance

                        for state in prod:
                            self.q_table[tuple(state)] = self.default * self.env.action_space(switch_id2name(switch.id)).n
                            self.q_table[tuple(state)][optimal_action] = self.optimal_init # optimistic initialization

                    else:  # final action towards target
                        
                        for act_idx, action in enumerate(switch.action_outcomes):

                            if in_port == action[0]:

                                _, out_port = action
                                _, next_port = self.env.rail_network.get_neighbor_switch(out_port)

                                edge_data = self.env.rail_network.rail_graph.get_edge_data(out_port, next_port)
                                rail_nodes = edge_data.get("rail_nodes", None)

                                for distance, node in enumerate(rail_nodes):
                                    if node == agent.target:
                                        if distance < min_distance:
                                            min_distance = distance
                                            optimal_action = act_idx
                                        break

                        for state in prod:
                            self.q_table[tuple(state)] = self.default * self.env.action_space(switch_id2name(switch.id)).n
                            self.q_table[tuple(state)][optimal_action] = self.destination_bonus


    def test(self, out_dir, plot=False, save_outputs=True):
        """ Tests the agent in the environment.
            Parameters
            ----------
            out_dir : str
                The output directory to save the results.
            plot : bool
                Whether to plot the environment at each step.
            save_outputs : bool
                Whether to save the outputs to files."""
    
        self.env.reset(seed=self.seed)

        num_iter = 0
        cum_reward = 0.
        for agent in self.env.agent_iter():

            observation, reward, termination, truncation, info = self.env.last()
            reward = reward[self.env.active_train]
            # print("--------------------------------------")
            # print(f"Active train: {self.env.active_train}")
            # print(f"Observation: {observation}")
            # print(f"Reward: {reward}")

            if termination or truncation:
                break
            
            action = self.max_action(observation, agent, info["action_mask"])

            # print(f"Action:{action}")
            post_step_info = self.env.step(action)

            if plot:
                a = self.env.rail_env.render()  # (agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS, show_debug=True)
                fig, ax = plt.subplots(figsize=(8,8))
                plt.imshow(a)
                # ax.set_xticks(np.arange(0, a.shape[0], a.shape[0]/self.env.rail_env.width), minor=False)
                # ax.set_yticks(np.arange(0, a.shape[0], a.shape[0]/self.env.rail_env.height), minor=False)
                # ax.xaxis.grid(True, which='major', color='black', linestyle='--')
                # ax.yaxis.grid(True, which='major', color='black', linestyle='--')
                # ax.set_xticklabels(np.arange(self.env.rail_env.width))
                # ax.set_yticklabels(np.arange(self.env.rail_env.height))
                plt.savefig(os.path.join(out_dir, f"iter_{num_iter}.png"), dpi=300)
                plt.close()

            num_iter += 1
            cum_reward += reward

        arrived_trains = len(post_step_info["arrived_trains"])
        delays = [v[1] for v in list(self.env.train_to_last_node.values())]

        self.env.close()
        if save_outputs:    
            print(f"Terminated in {num_iter} steps ({self.env.rail_env._elapsed_steps} flatland steps), cumulative reward = {cum_reward}")
            print(f"Arrived trains: {arrived_trains} / {self.env.rail_env.get_num_agents()}")
            print(f"Trains at destination: {post_step_info['arrived_trains']}")
            print(f"Delays: {delays}")
            print(f"Num malfunctions: {self.env.num_malfunctions}")

            np.savez_compressed(os.path.join(out_dir, f'cum_reward.npz'), x=cum_reward)
            np.savez_compressed(os.path.join(out_dir, f'trains_at_dest.npz'), x=post_step_info['arrived_trains'])
            np.savez_compressed(os.path.join(out_dir, f'delays.npz'), x=delays)

        return cum_reward, arrived_trains, delays


    def learn(self, num_episodes: int, out_dir: str, checkpoint_freq: int, exploit_freq = None):
        """
        Trains the agent in the environment.

        Parameters
        ----------
        num_episodes : int
            The number of episodes to train the agent.
        out_dir : str
            The output directory to save the results.
        checkpoint_freq : int
            The frequency (in episodes) to save checkpoints.
        exploit_freq : int
            The frequency (in episodes) to exploit the learned policy.
        """

        cum_reward = np.zeros(num_episodes)
        arrived_trains = []
        delays = []
        agent_num_interactions = {agent: 0 for agent in self.env.agents}
        num_malfunctions = []

        cum_reward_exploit = []
        arrived_trains_exploit = []

        rng = np.random.default_rng(self.seed)

        last_time = 0.
        action_selection_time = 0.
        update_time = 0.
        # reset_time = 0.

        for t in range(num_episodes):

            # Exploit round
            if exploit_freq is not None and (t+1) % exploit_freq == 0:
                test_reward, test_arrived_trains, _ = self.test(out_dir=None, plot=False, save_outputs=False)
                cum_reward_exploit.append(test_reward)
                arrived_trains_exploit.append(test_arrived_trains)

            update_dict = {}
            num_iter = 0
            trains_at_destination = []

            # Save checkpoint
            if (t+1) % checkpoint_freq == 0:
                self.save(os.path.join(out_dir, f"checkpoint_{t+1}.pkl"))
                np.savez_compressed(os.path.join(out_dir, f'cum_reward_checkpoint_{t+1}.npz'), x=cum_reward)
                np.savez_compressed(os.path.join(out_dir, f'arrived_trains_checkpoint_{t+1}.npz'), x=arrived_trains)
                np.savez_compressed(os.path.join(out_dir, f'delays_checkpoint_{t+1}.npz'), x=delays)
                np.savez_compressed(os.path.join(out_dir, f'trains_at_dest_checkpoint_{t+1}.npz'), x=trains_at_destination)
                np.savez_compressed(os.path.join(out_dir, f'num_malfunctions_checkpoint_{t+1}.npz'), x=num_malfunctions)                

            # start_reset_time = time.time()
            self.env.reset(seed=self.seed)
            # reset_time += time.time() - start_reset_time

            # Init q table
            if t == 0:
                self.__init_q_table()

            for agent in self.env.agent_iter():

                start_last_time = time.time()
                observation, reward, termination, truncation, info = self.env.last()
                reward = reward[self.env.active_train]
                last_time += time.time() - start_last_time
                # print("--------------------------------------")
                # print(f"Observation: {observation}")
                # print(f"Reward: {reward}")
                
                if termination or truncation:
                    break

                # Epsilon-greedy policy
                start_action_selection_time = time.time()
                self.__decay_epsilon(agent, agent_num_interactions[agent])
                if rng.random() < self.epsilon[agent]:
                    self.env.action_space(agent).seed(int(rng.integers(0, np.iinfo(np.int32).max)))
                    action = self.env.action_space(agent).sample(info["action_mask"])
                    # print(f"Sampled action: {action}")
                else:
                    action = self.max_action(observation, agent, info["action_mask"])
                    # print(f"Max action: {action}")
                action_selection_time += time.time() - start_action_selection_time

                post_step_info = self.env.step(action)
                active_train = info["active_train"]

                start_update_time = time.time()
                agent_id = name2switch_id(agent)
               
               # Update Q-values
                if (agent_id, active_train) in update_dict:
                    previous_obs = update_dict[(agent_id, active_train)][0]
                    previous_act = update_dict[(agent_id, active_train)][1]
                    previous_agent = update_dict[(agent_id, active_train)][2]
                    # print(f"Updating Q-values: agent={previous_agent}, obs={previous_obs}, act={previous_act} with reward={reward}, next state={observation}, next_agent={agent}")
                    self.update(state=previous_obs, action=previous_act,
                                reward=reward, next_state=observation,
                                previous_agent=previous_agent,
                                next_agent=agent,
                                agent_num_interactions=agent_num_interactions)
                    del update_dict[agent_id, active_train]

                next_q_agent = post_step_info["next_switch"]

                update_dict[(next_q_agent, active_train)] = (observation, action, agent)
                
                # destination bonus handling
                for train in post_step_info["arrived_trains"]:
                    if train not in trains_at_destination:
                        trains_at_destination.append(train)

                        for (upd_agent, upd_train), (upd_obs, upd_act, upd_previous_agent) in list(update_dict.items()):
                            if upd_train == train:
                                # print(f"Updating Q-values for ARRIVED TRAIN: agent={upd_previous_agent}, obs={upd_obs}, act={upd_act} with reward={500}, next state=None, next_agent=None")
                                self.update(state=upd_obs, action=upd_act,
                                            reward=self.destination_bonus, next_state=None,
                                            previous_agent=upd_previous_agent,
                                            next_agent=None,
                                            agent_num_interactions=agent_num_interactions)
                                del update_dict[upd_agent, upd_train]

                update_time += time.time() - start_update_time
                
                cum_reward[t] += reward
                num_iter += 1
                agent_num_interactions[agent] += 1

                # a = self.env.rail_env.render(agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS, show_debug=True)
                # fig, ax = plt.subplots(figsize=(8,8))
                # plt.imshow(a)
                # ax.set_xticks(np.arange(0, a.shape[0], a.shape[0]/self.env.rail_env.width), minor=False)
                # ax.set_yticks(np.arange(0, a.shape[0], a.shape[0]/self.env.rail_env.height), minor=False)
                # ax.xaxis.grid(True, which='major', color='black', linestyle='--')
                # ax.yaxis.grid(True, which='major', color='black', linestyle='--')
                # ax.set_xticklabels(np.arange(self.env.rail_env.width))
                # ax.set_yticklabels(np.arange(self.env.rail_env.height))
                # plt.savefig(os.path.join(out_dir, f"learn_iter_{num_iter}.png"), dpi=300)
                # plt.close()

            arrived_trains.append(len(post_step_info["arrived_trains"]))
            delays.append([v[1] for v in list(self.env.train_to_last_node.values())])
            num_malfunctions.append(self.env.num_malfunctions)

        np.savez_compressed(os.path.join(out_dir, 'cum_reward.npz'), x=cum_reward)
        np.savez_compressed(os.path.join(out_dir, 'arrived_trains.npz'), x=arrived_trains)
        np.savez_compressed(os.path.join(out_dir, 'delays.npz'), x=delays)
        np.savez_compressed(os.path.join(out_dir, 'trains_at_dest.npz'), x=post_step_info["arrived_trains"])
        np.savez_compressed(os.path.join(out_dir, 'num_malfunctions.npz'), x=num_malfunctions)
        if exploit_freq is not None:
            np.savez_compressed(os.path.join(out_dir, 'cum_reward_exploit.npz'), x=cum_reward_exploit)
            np.savez_compressed(os.path.join(out_dir, 'arrived_trains_exploit.npz'), x=arrived_trains_exploit)
        self.env.last_time = last_time
        self.env.action_selection_time = action_selection_time
        self.env.update_time = update_time
        # self.env.reset_time = reset_time
        self.env.close()

    def _get_next_q_agent(self, agent, action):
        """ Returns the next Q-learning agent based on the current agent and action.
        
        Parameters
        ----------
        agent : str
            The current agent.
        action : int
            The action taken by the agent."""
        
        switch_id = name2switch_id(agent)
        switch = self.env.rail_network.get_switch_on_position(switch_id)

        _, destination_port = switch.action_outcomes[action]
        next_switch_id = switch.port2neighbor[destination_port][0]
        next_q_agent = switch_id2name(next_switch_id)

        return next_q_agent

    def eval(self, state, action, agent):
        """
        Evaluates the Q-value of a state-action pair.

        Parameters
        ----------
        state : int
            The state.
        action : int
            The action.

        Returns
        -------
        float
            The Q-value of the state-action pair.
        """
        self.__check_entry(state, agent)
        return self.q_table[tuple(state)][action]

    def update(self, state, action, reward, next_state, previous_agent, next_agent, agent_num_interactions):
        """
        Updates the Q-value of a state-action pair.

        Parameters
        ----------
        lr : float
            The learning rate.
        state : int
            The state.
        action : int
            The action.
        reward : float
            The reward.
        next_state : int
            The next state.
        """
        self.__check_entry(state, previous_agent)

        self.__decay_lr(previous_agent, agent_num_interactions[previous_agent])

        if next_agent != previous_agent:
            self.q_table[tuple(state)][action] = \
                (1 - self.lr[previous_agent]) * self.q_table[tuple(state)][action] + \
                self.lr[previous_agent] * (reward + self.gamma * self.max_q(next_state, next_agent))
        else:
            self.q_table[tuple(state)][action] = \
                (1 - self.lr[previous_agent]) * self.q_table[tuple(state)][action] + \
                self.lr[previous_agent] * reward         
        # print(f"New Q-entry: {self.q_table[tuple(state)]}")
        # print("")

    def max_q(self, state, agent):
        """
        Returns the maximum Q-value of a state.

        Parameters
        ----------
        state : int
            The state.

        Returns
        -------
        float
            The maximum Q-value of the state
        """
        if state is None:  # final state
            return 0.
        self.__check_entry(state, agent)
        return max(self.q_table[tuple(state)])
    
    def max_action(self, state, agent, action_mask):
        """
        Returns the action that maximizes the Q-value of a state.
        
        Parameters
        ----------
        state : int
            The state.

        Returns
        -------
        int
            The action that maximizes the Q-value of the state.
        """
        self.__check_entry(state, agent)
        max_q = np.argmax(self.q_table[tuple(state)])
        # print(f"Action mask={action_mask}")
        # print(f"Q-entry: {self.q_table[tuple(state)]}")
        if action_mask[max_q]:
            return max_q
        else:
            # If the action with max Q-value is not allowed, choose among allowed actions
            allowed_actions = np.nonzero(action_mask)[0]
            allowed_q_values = np.array(self.q_table[tuple(state)])[allowed_actions]
            return allowed_actions[np.argmax(allowed_q_values)]

    def save(self, filename: str, mode: str = 'pickle'):
        """
        Dumps the agent to a file.

        Parameters
        ----------
        filename : str
            The name of the file.
        mode : str
            The mode of the dump (pickle, csv, parquet).
        """
        if mode == 'pickle':
            self.__dump_pickle(filename)
        elif mode == 'csv':
            self.__dump_csv(filename)
        elif mode == 'parquet':
            self.__dump_parquet(filename)

    def load(self, filename: str):
        """
        Loads the agent from a file (pickle).
        
        Parameters
        ----------
        filename : str
            The name of the file.
        """
        return self.__load_pickle(filename)

    def __dump_pickle(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def __load_pickle(self, filename: str):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)