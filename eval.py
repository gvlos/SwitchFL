from flatland_tools.env import Environment
from flatland_tools.agent import TQLearningAgent
import pandas as pd
import numpy as np


import sys 
import os 
sys.path.append(os.path.abspath('../'))

from flatland_tools.agent import TQLearningAgent

from train import generate_env
from training_utils.training_utils import eval_once


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


    agent_path = "./experiments/reproduce_deterministic/1/qtables/qtable_400000.pkl"

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



