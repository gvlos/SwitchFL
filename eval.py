import configparser
import os
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from switchfl.switch_env import ASyncSwitchEnv
import os
from switchfl.distr_q import DistrQLearning
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
import time
import argparse


if __name__=='__main__':

    config_path = "/home/gianvito/Desktop/flatland_exp/5_agents_with_init/exp_0/seed_0/config.ini"
    model_path = "/home/gianvito/Desktop/flatland_exp/5_agents_with_init/exp_0/seed_0/distr_q_model.pkl"
    out_dir = "/home/gianvito/Desktop/flatland_exp/5_agents_with_init/exp_0/seed_0"

    config = configparser.ConfigParser()
    config.read(config_path)

    checkpoint_freq = int(config["MISC"]["checkpoint_freq"])

    stochastic_data = MalfunctionParameters(
        malfunction_rate=float(config["ENV"]["malfunction_rate"]),  # Rate of malfunction occurence
        min_duration=int(config["ENV"]["min_duration"]),  # Minimal duration of malfunction
        max_duration=int(config["ENV"]["max_duration"])  # Max duration of malfunction
    )
    mf = ParamMalfunctionGen(stochastic_data)


    random_seed = config["MISC"]["random_seed"]
    rail_env = RailEnv(
        width=int(config["ENV"]["width"]),
        height=int(config["ENV"]["height"]),
        rail_generator=sparse_rail_generator(
            max_num_cities=int(config["ENV"]["max_num_cities"]),
            grid_mode=True,
            max_rails_between_cities=int(config["ENV"]["max_rails_between_cities"]),
            max_rail_pairs_in_city=int(config["ENV"]["max_rail_pairs_in_city"]),
            seed=int(config["MISC"]["random_seed"]),
        ),
        line_generator=sparse_line_generator(seed=int(config["MISC"]["random_seed"])),
        number_of_agents=int(config["ENV"]["number_of_agents"]),
        malfunction_generator=mf
    )

    env = ASyncSwitchEnv(rail_env, render_mode="human", max_steps=100_000)

    model = DistrQLearning(env=env,
                        gamma = float(config["MODEL"]["gamma"]),
                        epsilon = float(config["MODEL"]["epsilon"]),
                        epsilon_decay_rate = float(config["MODEL"]["epsilon_decay_rate"]),
                        lr = float(config["MODEL"]["lr"]),
                        lr_decay_rate = float(config["MODEL"]["lr_decay_rate"]),
                        default_q = float(config["MODEL"]["default_q"]),
                        seed = int(config["MISC"]["random_seed"]))
    
    model.load(model_path)
    model.test(out_dir=out_dir, plot=True)