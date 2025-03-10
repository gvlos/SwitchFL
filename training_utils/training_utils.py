from flatland_tools.agent import TQLearningAgent
from flatland_tools.env import Environment
import numpy as np
from typing import Callable
import time
import os
import pandas as pd
import pickle

def log2file(log_file: str, text: str):
    """
    Logs text to a file
    
    Parameters:
    ----------
    log_file: str
        The file to log to
    text: str
        The text to log
    """
    with open(log_file, 'a') as f:
        print(f'{time.ctime()}: {text}', file=f, flush=True)

def train_one_episode(
    agent: TQLearningAgent,
    env: Environment,
    alpha: float,
    epsilon: float,
    rng: np.random.RandomState,
    malf_seed: int = None,
    return_node_interactions: bool = False
):
    """
    Does one episode of training of the environment

    Parameters:
    ----------
    agent: TQLearningAgent
        The agent to train
    env: Environment
        The environment to train the agent in
    alpha: float
        The learning rate
    epsilon: float
        The epsilon value for epsilon-greedy action selection
    rng: np.random.RandomState
        The random number generator to use for epsilon-greedy action selection
    malf_seed: int
        The seed for the malfunction generator for this episode, if any. 
        If None, malfunction generator is not seeded, and may be random. 
        Defaults to None.
    return_node_rewards: bool
        Whether to return node interaction data. Defaults to False.

    Returns:
    -------
    cumulative_reward: float
        The cumulative reward for the episode
    node_interactions: list
        A list of node interactions if return_node_interactions is True, else None
    """
    if malf_seed is not None:
        env.rail_env.malfunction_generator.seed = malf_seed
        reset_malfunctions = True
    else:
        reset_malfunctions = False
    env.reset(step_until_action_required=True, reset_malfunctions=reset_malfunctions)
    cumulative_reward = 0.0
    train_to_obs_action_node_id = {}
    if return_node_interactions:
        node_interactions = []
    else:
        node_interactions = None
    low_level_time = 0
    training_time = time.time()
    while True:
        actions = []
        for train_id, node_id, obs, old_reward in env.observe():
            if train_id in train_to_obs_action_node_id:
                last_obs, last_action, old_node_id = train_to_obs_action_node_id[train_id]
                agent.update(alpha, last_obs, last_action, old_reward, obs)
                if return_node_interactions:
                    node_interactions.append((last_obs, last_action, old_reward, obs, old_node_id, node_id))
            if rng.rand() < epsilon:
                action = rng.randint(3)
            else:
                action = agent.max_action(obs)
            cumulative_reward += old_reward # Log reward
            actions.append((train_id, node_id, action))
            train_to_obs_action_node_id[train_id] = (obs, action, node_id)
        done, sll = env.step(actions, step_until_action_required=True)
        low_level_time += sll
        if done:
            break
    last_rewards = env.end_episode()
    for train_id, (obs, action, node_id) in train_to_obs_action_node_id.items():
        r = last_rewards[node_id]
        agent.update(alpha, obs, action, r)
        cumulative_reward += r
        if return_node_interactions:
            node_interactions.append((obs, action, r, None, node_id, None))
    training_time = time.time() - training_time
    return cumulative_reward, node_interactions, low_level_time, training_time

def train(
    env: Environment,
    agent: TQLearningAgent,
    n_episodes: int,
    alphas: np.ndarray[float],
    epsilons: np.ndarray[float],
    with_malfunctions: bool,
    save_node_interactions: bool,
    rng_master_seed: int,
    log_every: int,
    save_every: int,
    rewards_out_dir: str,
    qtables_out_dir: str,
    log_file: str
):
    """
    Trains the agent in the environment (fully reproducible)

    Parameters:
    ----------
    env: Environment
        The environment to train the agent in
    agent: TQLearningAgent
        The agent to train
    n_episodes: int
        The number of episodes to train the agent for
    alphas: np.ndarray[float]
        The learning rates for each episode
    epsilons: np.ndarray[float]
        The epsilon values for each episode
    with_malfunctions: bool
        Whether to train with malfunctions (provides seeds for malfunction generator)
    save_node_interactions: bool
        Whether to save node interactions data for each episode.
    rng_master_seed: int
        The master seed for reproducibility
    log_every: int
        How often to print the cumulative reward
    save_every: int
        How often to save the agent
    rewards_out_dir: str
        The directory to save the training rewards to
    qtables_out_dir: str
        The directory to save intermediate qtables to
    log_file: str
        The file to log the training to

    Returns:
    -------
    None
    """
    # Start time
    start_time = time.time()

    master_rng = np.random.RandomState(rng_master_seed)
    rng_seeds = master_rng.randint(2**32, size=n_episodes)
    if with_malfunctions:
        malf_seeds = master_rng.randint(2**32, size=n_episodes)

    # Cumulative rewards
    cumulative_rewards = []
    node_interactions = []
    low_level_time_list = []
    training_time_list = []

    for ep in range(n_episodes):
        cum_rew, node_int, low_level_time, training_time = train_one_episode(
            agent, env, alphas[ep], epsilons[ep],
            np.random.RandomState(rng_seeds[ep]),
            malf_seed = malf_seeds[ep] if with_malfunctions else None,
            return_node_interactions = save_node_interactions
        )
        cumulative_rewards.append((ep+1, cum_rew))
        low_level_time_list.append(low_level_time)
        training_time_list.append(training_time)

        if save_node_interactions:
            node_interactions += node_int
        # Logging
        time_to_log = (ep == 0) or ((ep+1) % log_every == 0) or (ep+1 == n_episodes)
        if time_to_log:
            log2file(log_file, f'Episode {ep+1}/{n_episodes}: Cumulative reward = {cum_rew}')
            log2file(log_file, f'Episode {ep+1}/{n_episodes}: Low-level time = {low_level_time:.2f} seconds, Training time = {training_time:.2f} seconds')
        # Saving
        time_to_save = ((ep+1) % save_every == 0) or (ep+1 == n_episodes)
        if time_to_save:
            # Save qtable
            agent.dump(os.path.join(qtables_out_dir, f'qtable_{ep+1}.pkl'))
            # Save rewards
            pd.DataFrame(cumulative_rewards, columns=['Episode', 'Cumulative reward'])\
                .set_index('Episode')\
                .to_parquet(os.path.join(rewards_out_dir, 'tr_rewards.parquet'))
            
            with open(os.path.join(rewards_out_dir, 'low_level_time.npy'), 'wb') as f:
                np.save(f, np.array(low_level_time_list))
            with open(os.path.join(rewards_out_dir, 'training_time.npy'), 'wb') as f:
                np.save(f, np.array(training_time_list))
            
            # Save node interactions
            if save_node_interactions:
                with open(os.path.join(rewards_out_dir, 'node_interactions.pkl'), 'wb') as f:
                    pickle.dump(node_interactions, f)
    # Final log
    log2file(log_file, f'Training completed in {time.time() - start_time:.2f} seconds')


def eval_once(
    env: Environment,
    agent: TQLearningAgent,
    malf_seed: int = None,
    return_node_interactions: bool = False
):
    """
    Evaluates the agent in the environment once with the greedy policy.

    Parameters:
    ----------
    env: Environment
        The environment to evaluate the agent in
    agent: TQLearningAgent
        The agent to evaluate
    malf_seed: int
        The seed for the malfunction generator for this episode, if any. 
        If None, malfunction generator is not seeded, and may be random. 
        Defaults to None.
    return_node_interactions: bool
        Whether to return node interaction data. Defaults to False.

    Returns:
    -------
    cumulative_reward: float
        The cumulative reward for the episode
    delays: dict[int, int | None]
        A dictionary mapping train IDs to the delay at the final node if the train arrived, else None
    n_arrived: int
        The number of trains that arrived at their destination
    n_arrived_on_time: int
        The number of trains that arrived at their destination on time
    node_interactions: list
        A list of node interactions if return_node_interactions is True, else None
    """
    if malf_seed is not None:
        env.rail_env.malfunction_generator.seed = malf_seed
        reset_malfunctions = True
    else:
        reset_malfunctions = False
    env.reset(step_until_action_required=True, reset_malfunctions=reset_malfunctions)
    cumulative_reward = 0.0
    train_to_obs_action_node_id = {}
    if return_node_interactions:
        node_interactions = []
    else:
        node_interactions = None
    while True:
        actions = []
        for train_id, node_id, obs, old_reward in env.observe():
            if train_id in train_to_obs_action_node_id:
                last_obs, last_action, old_node_id = train_to_obs_action_node_id[train_id]
                if return_node_interactions:
                    node_interactions.append((last_obs, last_action, old_reward, obs, old_node_id, node_id))
            action = agent.max_action(obs)
            cumulative_reward += old_reward # Log reward
            actions.append((train_id, node_id, action))
            train_to_obs_action_node_id[train_id] = (obs, action, node_id)
        if env.step(actions, step_until_action_required=True):
            break
    last_rewards = env.end_episode()
    for train_id, (obs, action, node_id) in train_to_obs_action_node_id.items():
        r = last_rewards[node_id]
        cumulative_reward += r
        if return_node_interactions:
            node_interactions.append((obs, action, r, None, node_id, None))
    
    delays = {}
    n_arrived = 0
    n_arrived_on_time = 0
    for train_id, (_, final_delay) in env.train_to_last_node.items():
        if env.trains_arrived[train_id]:
            delays[train_id] = final_delay
            n_arrived += 1
            if final_delay <= 0:
                n_arrived_on_time += 1
        else:
            delays[train_id] = None
    return cumulative_reward, delays, n_arrived, n_arrived_on_time, node_interactions