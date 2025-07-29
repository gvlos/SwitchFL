from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
import json
from switchfl.switch_env import ASyncSwitchEnv

random_seed = 41
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
for agent in env.agent_iter():
    print("===================================================")
    print(f"{env.rail_env_time=}")   
    observation, reward, termination, truncation, info = env.last()
    print(agent)
    print(json.dumps(env.obs2json(agent, observation)))

    if termination or truncation:
        break
    # this is where you would insert your policy

    action = env.action_space(agent).sample(info["action_mask"])
    env.render()
    env.step(action)
env.close()
