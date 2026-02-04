from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from switchfl.switch_env import ASyncSwitchEnv
import matplotlib.pyplot as plt
import numpy as np
import os
from switchfl.distr_q import DistrQLearning
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
import time



stochastic_data = MalfunctionParameters(
    malfunction_rate=0.01,  # Rate of malfunction occurence
    min_duration=5,  # Minimal duration of malfunction
    max_duration=15  # Max duration of malfunction
)
mf = ParamMalfunctionGen(stochastic_data)


if __name__=='__main__':


    # Output directory for experiment results
    out_dir = '/home/gianvito/Desktop/debug'
    os.makedirs(out_dir, exist_ok=True)

    # Environment setup
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

    num_episodes = 5



    # -------------------------------------------------------------------------------------
    # DO NOT MODIFY BELOW THIS LINE
    # -------------------------------------------------------------------------------------

    env = ASyncSwitchEnv(rail_env, render_mode="human", max_steps=100_000)

    model = DistrQLearning(env=env,
                           gamma = 1.,
                           epsilon = 0.5,
                           epsilon_decay_rate = 0.9997,
                           lr = 0.1,
                           lr_decay_rate = 1.0,
                           default_q = 0.,
                           seed = random_seed)
    
    
    start_time = time.time()

    model.learn(num_episodes=num_episodes, out_dir=out_dir, checkpoint_freq=10000)

    model.save(os.path.join(out_dir, "distr_q_model.pkl"))

    elapsed_time = time.time() - start_time
    print("DONE!")
    print(f"TOTAL TIME: {elapsed_time:.1f} seconds")
    print(f"Seconds per episode: {elapsed_time / num_episodes:.1f}")
    print(f"Flatland step time: {env.flatland_step_time:.1f} seconds")
    print(f"Total step time: {env.step_time:.1f} seconds")
    print(f"Total last time: {env.last_time:.1f} seconds")
    print(f"Action selection time: {env.action_selection_time:.1f} seconds")
    print(f"Update time: {env.update_time:.1f} seconds")
    print(f"Flatland reset time: {env.reset_time:.1f} seconds")
    print(f"Total reset time: {env.reset_total_time:.1f} seconds")