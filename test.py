from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from switchfl.switch_env import ASyncSwitchEnv
import matplotlib.pyplot as plt
import numpy as np

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

    num_iter = 0
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            break
        
        # Replace with your custom switch policy
        env.action_space(agent).seed(random_seed * num_iter + 1)
        action = env.action_space(agent).sample(info["action_mask"])

        env.step(action)

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

        num_iter += 1

    env.close()