import configparser
import os
import subprocess
from itertools import product

if __name__=='__main__':
    
    random_seed = 18
    out_dir = "/home/gianvito/Desktop/debug_15_agents"
    
    num_episodes = 5000
    checkpoint_freq = 2000

    width = 80
    height = 80
    max_num_cities = 25
    max_rails_between_cities = 2
    max_rail_pairs_in_city = 2
    number_of_agents = 15
    malfunction_rate = 0.
    min_duration = 0
    max_duration = 0

    # width = 18
    # height = 18
    # max_num_cities = 5
    # max_rails_between_cities = 1
    # max_rail_pairs_in_city = 1
    # number_of_agents = 2
    # malfunction_rate = 0.
    # min_duration = 0
    # max_duration = 0

    hyperparams = {
        "epsilon" : [1.0],
        "epsilon_decay_rate" : [0.9998],
        "lr" : [0.1],
        "lr_decay_rate" : [1.0],
    }

    gamma = 1.
    default_q = 0.

    # ----------------------------------------------------------------------
    
    model_param_list = list((dict(zip(hyperparams.keys(), values)) 
                        for values in product(*hyperparams.values())))
    
    for idx, params in enumerate(model_param_list):

        exp_dir = os.path.join(out_dir, f"exp_{idx}")
        os.makedirs(exp_dir, exist_ok=True)

        config = configparser.ConfigParser()

        config["MISC"] = {
            "random_seed" : random_seed,
            "out_dir" : exp_dir,
            "checkpoint_freq" : checkpoint_freq
        }

        config["ENV"] = {
            "width" : width,
            "height" : height,
            "max_num_cities" : max_num_cities,
            "max_rails_between_cities" : max_rails_between_cities,
            "max_rail_pairs_in_city" : max_rail_pairs_in_city,
            "number_of_agents" : number_of_agents,
            "malfunction_rate" : malfunction_rate,
            "min_duration" : min_duration,
            "max_duration" : max_duration
        }

        config["MODEL"] = {
            "gamma" : gamma,
            "epsilon" : params["epsilon"],
            "epsilon_decay_rate" : params["epsilon_decay_rate"],
            "lr" : params["lr"],
            "lr_decay_rate" : params["lr_decay_rate"],
            "default_q" : default_q,
            "num_episodes" : num_episodes
        }

        config_path = os.path.join(exp_dir, 'config.ini')
        with open(config_path, 'w') as configfile:
            config.write(configfile)

        # cmd = f"source {venv_dir}/bin/activate && which python3 &> out.out && python3 main.py -c {config_path} 1>{os.path.join(exp_dir, "stdout.out")} 2>{os.path.join(exp_dir, "stderr.err")} &"        
        cmd = f"python main.py -c {config_path} 1>{os.path.join(exp_dir, "stdout.out")} 2>{os.path.join(exp_dir, "stderr.err")} &"

        try:
            print()
            print('------------------------------------------------')
            print(f"Starting experiment {idx+1}/{len(model_param_list)}")
            subprocess.Popen(cmd, shell=True, executable='/bin/bash')
        except subprocess.CalledProcessError as e:
            print(str(e)) 