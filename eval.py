from flatland_tools.env import Environment
from flatland_tools.agent import TQLearningAgent
import pandas as pd
import numpy as np


import sys 
import os 
sys.path.append(os.path.abspath('../'))

from flatland_tools.agent import TQLearningAgent

from train import generate_env

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


def eval_batch(
    malf_rate: float,
    malf_min: int,
    malf_max: int,
    malf_seeds: np.ndarray[int],
    exp_id: int,
    first_eval_id: int,
    agent_path: str,
    out_dir: str
):
    env = generate_env(malf_rate, malf_min, malf_max)
    agent=TQLearningAgent.load(agent_path)

    eval_df = []
    eval_df_columns = \
        ['Experiment id', 'Eval id', 'Eval seed', 'Cumulative reward'] + \
        [f'Delay {i}' for i in range(env.rail_env.get_num_agents())] + \
        ['# arrived', '# arrived on time']
    for idx, malf_seed in enumerate(malf_seeds):
        cumulative_reward, delays, n_arrived, \
            n_arrived_on_time, _ = eval_once(
                env, agent, malf_seed, False
            )
        eval_df.append([
            exp_id,
            first_eval_id + idx,
            malf_seed,
            cumulative_reward,
        ] + [
            delays[train_id] if train_id in delays else None for train_id in range(env.rail_env.get_num_agents())
        ] + [
            n_arrived,
            n_arrived_on_time
        ])

    print(n_arrived_on_time)
    print(n_arrived)
    print(delays)


if __name__ == '__main__':


    agent_path = "./experiments/reproduce_deterministic/2/qtables/qtable_400000.pkl"

    agent=TQLearningAgent.load(agent_path)

    eval_batch(
        malf_rate=0.0,
        malf_min=0,
        malf_max=0,
        malf_seeds=[0],
        exp_id=1,
        first_eval_id=10,
        agent_path=agent_path,
        out_dir="."
    )



