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


    # Set of experiment directories to evaluate
    exp_dir_list = [
        "/home/gianvito/Desktop/flatland_exp/100_agents_challenge/exp_0/seed_0",
        "/home/gianvito/Desktop/flatland_exp/100_agents_challenge/exp_0/seed_1",
        "/home/gianvito/Desktop/flatland_exp/100_agents_challenge/exp_0/seed_2",
        "/home/gianvito/Desktop/flatland_exp/100_agents_challenge/exp_0/seed_3",
        "/home/gianvito/Desktop/flatland_exp/100_agents_challenge/exp_0/seed_4",
    ]

    # Name of the trained model file to load
    distr_q_model_name = "checkpoint_3000.pkl"



    # -------------------------------------------------------------------------------------
    # DO NOT MODIFY BELOW THIS LINE
    # -------------------------------------------------------------------------------------
    for exp_dir in exp_dir_list:

        print(f"Evaluating {exp_dir}")
        config_path = os.path.join(exp_dir, "config.ini")
        model_path = os.path.join(exp_dir, distr_q_model_name)
        out_dir = exp_dir

        config = configparser.ConfigParser()
        config.read(config_path)

        checkpoint_freq = int(config["MISC"]["checkpoint_freq"])

        malfunction_rate= float(config["ENV"]["malfunction_rate"])  # Rate of malfunction occurence
        min_duration= int(config["ENV"]["min_duration"])  # Minimal duration of malfunction
        max_duration= int(config["ENV"]["max_duration"])  # Max duration of malfunction

        stochastic_data = MalfunctionParameters(
            malfunction_rate=malfunction_rate,
            min_duration=min_duration,
            max_duration=max_duration
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

        # Multiple evals if malfunctions
        num_evals = 10 if malfunction_rate > 0 else 1

        for i in range(num_evals):

            print(f"Eval {i+1}")

            out_dir = os.path.join(exp_dir, f"eval_{i}")
            os.makedirs(out_dir, exist_ok=True)

            model.test(out_dir=out_dir, plot=False)

            print("")