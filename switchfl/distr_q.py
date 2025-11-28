from .switch_env import ASyncSwitchEnv, name2switch_id, switch_id2name
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from flatland.utils.rendertools import AgentRenderVariant

# class Grid4TransitionsEnum(IntEnum):
#     NORTH = 0
#     EAST = 1
#     SOUTH = 2
#     WEST = 3

# DO_NOTHING: 0
# MOVE_LEFT: 1
# MOVE_FORWARD: 2
# MOVE_RIGHT: 3
# STOP_MOVING: 4

# directions
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
        self.default = [default_q] # MISSING NUMBER OF ACTIONS
        self.q_table = {}
        self.seed = seed
        # self.rng  = np.random.RandomState(seed)

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

    def test(self, out_dir=None, plot=False):
    
        out_dir = os.path.join(out_dir, "eval")
        os.makedirs(out_dir, exist_ok=True)
        self.env.reset(seed=self.seed)

        num_iter = 0
        cum_reward = 0.
        for agent in self.env.agent_iter():

            observation, reward, termination, truncation, info = self.env.last()
            print("--------------------------------------")
            print(f"Observation: {observation}")
            print(f"Reward: {reward}")

            if termination or truncation:
                break
            
            action = self.max_action(observation, agent, info["action_mask"])

            print(f"Action:{action}")
            post_step_info = self.env.step(action)

            if plot:
                a = self.env.rail_env.render(agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS, show_debug=True)
                fig, ax = plt.subplots(figsize=(8,8))
                plt.imshow(a)
                ax.set_xticks(np.arange(0, a.shape[0], a.shape[0]/self.env.rail_env.width), minor=False)
                ax.set_yticks(np.arange(0, a.shape[0], a.shape[0]/self.env.rail_env.height), minor=False)
                ax.xaxis.grid(True, which='major', color='black', linestyle='--')
                ax.yaxis.grid(True, which='major', color='black', linestyle='--')
                ax.set_xticklabels(np.arange(self.env.rail_env.width))
                ax.set_yticklabels(np.arange(self.env.rail_env.height))
                plt.savefig(os.path.join(out_dir, f"iter_{num_iter}.png"), dpi=300)
                plt.close()

            num_iter += 1
            cum_reward += reward

        self.env.close()
        arrived_trains = len(post_step_info["arrived_trains"])
        print(f"Terminated in {num_iter} steps ({self.env.rail_env._elapsed_steps} flatland steps), cumulative reward = {cum_reward}")
        print(f"Arrived trains: {arrived_trains} / {self.env.rail_env.get_num_agents()}")
        print(f"Delays: {[v[1] for v in list(self.env.train_to_last_node.values())]}")

    def learn(self, num_episodes: int, out_dir: str, checkpoint_freq: int):
        """
        Placeholder for the learning method.

        Parameters
        ----------
        num_timesteps : int
            The number of timesteps to learn.
        """

        cum_reward = np.zeros(num_episodes)
        arrived_trains = np.zeros(num_episodes)
        delays = [[] for _ in range(num_episodes)]
        agent_num_interactions = {agent: 0 for agent in self.env.agents}

        rng = np.random.default_rng(self.seed)

        for t in range(num_episodes):

            if (t+1) % checkpoint_freq == 0:
                self.save(os.path.join(out_dir, f"checkpoint_{t+1}.pkl"))
                np.savez_compressed(os.path.join(out_dir, f'cum_reward_checkpoint_{t+1}.npz'), x=cum_reward)
                np.savez_compressed(os.path.join(out_dir, f'arrived_trains_checkpoint_{t+1}.npz'), x=arrived_trains)
                np.savez_compressed(os.path.join(out_dir, f'delays_checkpoint_{t+1}.npz'), x=delays)

            self.env.reset(seed=self.seed)

            update_dict = {}
            num_iter = 0
            trains_at_destination = []

            for agent in self.env.agent_iter():

                observation, reward, termination, truncation, info = self.env.last()
                # print("--------------------------------------")
                # print(f"Observation: {observation}")
                # print(f"Reward: {reward}")
                
                if termination or truncation:
                    break

                self.__decay_epsilon(agent, agent_num_interactions[agent])
                if rng.random() < self.epsilon[agent]:
                    self.env.action_space(agent).seed(int(rng.integers(0, np.iinfo(np.int32).max)))
                    action = self.env.action_space(agent).sample(info["action_mask"])
                    # print(f"Sampled action: {action}")
                else:
                    action = self.max_action(observation, agent, info["action_mask"])
                    # print(f"Max action: {action}")

                post_step_info = self.env.step(action)
                active_train = info["active_train"]

                agent_id = name2switch_id(agent)
                # Può accadere che le observation consecutive sono di switch non consecutivi quindi bisogna selezionare l'observation giusta per fare l'update
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
                                            reward=500, next_state=None,
                                            previous_agent=upd_previous_agent,
                                            next_agent=None,
                                            agent_num_interactions=agent_num_interactions)
                                del update_dict[upd_agent, upd_train]
                
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

            arrived_trains[t] = len(post_step_info["arrived_trains"])
            delays[t] = [v[1] for v in list(self.env.train_to_last_node.values())]

        np.savez_compressed(os.path.join(out_dir, 'cum_reward.npz'), x=cum_reward)
        np.savez_compressed(os.path.join(out_dir, 'arrived_trains.npz'), x=arrived_trains)
        np.savez_compressed(os.path.join(out_dir, 'delays.npz'), x=delays)
        self.env.close()

    def _get_next_q_agent(self, agent, action):
        
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

        # print(f"Previous Q-entry: {self.q_table[tuple(state)]}")
        self.q_table[tuple(state)][action] = \
            (1 - self.lr[previous_agent]) * self.q_table[tuple(state)][action] + \
            self.lr[previous_agent] * (reward + self.gamma * self.max_q(next_state, next_agent))
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
        return max(self.q_table[tuple(state)]) if state is not None else 0.
    
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
        print(f"Action mask={action_mask}")
        print(f"Q-entry: {self.q_table[tuple(state)]}")
        if action_mask[max_q]:
            return max_q
        else:
            # If the action with max Q-value is not allowed, choose among allowed actions
            allowed_actions = [a for a in range(len(action_mask)) if action_mask[a]]
            allowed_q_values = [self.q_table[tuple(state)][a] for a in allowed_actions]
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