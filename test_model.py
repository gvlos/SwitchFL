from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from switchfl.switch_env import ASyncSwitchEnv
import matplotlib.pyplot as plt
import numpy as np
import os
from switchfl.distr_q import DistrQLearning
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen



stochastic_data = MalfunctionParameters(
    malfunction_rate=0,  # Rate of malfunction occurence
    min_duration=0,  # Minimal duration of malfunction
    max_duration=0  # Max duration of malfunction
)
mf = ParamMalfunctionGen(stochastic_data)


if __name__=='__main__':

    out_dir = '/home/gianvito/Desktop/debug_q_learning'
    os.makedirs(out_dir, exist_ok=True)


    random_seed = 450565
    rail_env = RailEnv(
        width=18,
        height=18,
        rail_generator=sparse_rail_generator(
            max_num_cities=5,
            grid_mode=True,
            max_rails_between_cities=1,
            max_rail_pairs_in_city=1,
            seed=random_seed,
        ),
        line_generator=sparse_line_generator(seed=random_seed),
        number_of_agents=2,
        malfunction_generator=mf
    )

    # random_seed = 15
    # rail_env = RailEnv(
    #     width=40,
    #     height=40,
    #     rail_generator=sparse_rail_generator(
    #         max_num_cities=7,
    #         grid_mode=True,
    #         max_rails_between_cities=1,
    #         max_rail_pairs_in_city=1,
    #         seed=random_seed,
    #     ),
    #     line_generator=sparse_line_generator(seed=random_seed),
    #     number_of_agents=5,
    #     malfunction_generator=mf
    # )

    env = ASyncSwitchEnv(rail_env, render_mode="human", max_steps=100_000)

    model = DistrQLearning(env=env,
                           gamma = 1.,
                           epsilon = 1.0,
                           epsilon_decay_rate = 0.9999,
                           lr = 0.1,
                           lr_decay_rate = 0.9999,
                           default_q = 0.,
                           seed = random_seed)
    
    
    model.learn(num_episodes=100, out_dir=out_dir, checkpoint_freq=50)

    model.save(os.path.join(out_dir, "distr_q_model.pkl"))