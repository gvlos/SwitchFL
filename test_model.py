from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from switchfl.switch_env import ASyncSwitchEnv
import matplotlib.pyplot as plt
import numpy as np
from switchfl.distr_q import DistrQLearning

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
    )

    env = ASyncSwitchEnv(rail_env, render_mode="human")

    env.reset(seed=random_seed)

    model = DistrQLearning(env=env,
                           gamma = 1.,
                           epsilon = 0.4,
                           epsilon_decay_rate = 0.,
                           lr = 0.4,
                           lr_decay_rate = 0.,
                           default_q = 0.,
                           seed = random_seed)
    
    
    model.learn(num_episodes=10)
    
    a = env.rail_env.render()
    fig, ax = plt.subplots(figsize=(8,8))
    plt.imshow(a)
    ax.set_xticks(np.arange(0, a.shape[0], a.shape[0]/18), minor=False)
    ax.set_yticks(np.arange(0, a.shape[0], a.shape[0]/18), minor=False)
    ax.xaxis.grid(True, which='major', color='black', linestyle='--')
    ax.yaxis.grid(True, which='major', color='black', linestyle='--')
    ax.set_xticklabels(np.arange(18))
    ax.set_yticklabels(np.arange(18))
    plt.show(block=True)


    model.save("distr_q_model.pkl")

    env.render()