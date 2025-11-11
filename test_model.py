from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from switchfl.switch_env import ASyncSwitchEnv
import matplotlib.pyplot as plt
import numpy as np
from switchfl.distr_q import DistrQLearning
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen



stochastic_data = MalfunctionParameters(
    malfunction_rate=0,  # Rate of malfunction occurence
    min_duration=0,  # Minimal duration of malfunction
    max_duration=0  # Max duration of malfunction
)
mf = ParamMalfunctionGen(stochastic_data)


if __name__=='__main__':

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

    env = ASyncSwitchEnv(rail_env, render_mode="human", max_steps=1000)

    model = DistrQLearning(env=env,
                           gamma = 1.,
                           epsilon = 0.4,
                           epsilon_decay_rate = 0.,
                           lr = 0.4,
                           lr_decay_rate = 0.,
                           default_q = 0.,
                           seed = random_seed)
    
    
    model.learn(num_episodes=100)

    model.save("distr_q_model.pkl")

    env.render()