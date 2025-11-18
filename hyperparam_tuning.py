import configparser
import os
import subprocess
from itertools import product

if __name__=='__main__':
    
    random_seed = 15
    out_dir = "/home/gianvito/Desktop/test_config"
    checkpoint_freq = 50

    width = 40
    height = 40
    max_num_cities = 7
    max_rails_between_cities = 1
    max_rail_pairs_in_city = 1
    number_of_agents = 5
    malfunction_rate = 0.
    min_duration = 0
    max_duration = 0

    hyperparams = {
        "epsilon" : [0.7, 0.4],
        "epsilon_decay_rate" : [0.],
        "lr" : [0.4],
        "lr_decay_rate" : [0.],
    }

    gamma = 1.
    default_q = 0.

    num_episodes = 100

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


    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for config_path, elapsed_time in zip(config_list, executor.map(launch_experiment, config_list)):
    #         print(f"Exp {config_path.split('/')[-2]} done in {round(elapsed_time,2)} seconds")

        cmd = f"python main.py -c {config_path} 1>{os.path.join(exp_dir, "stdout.out")} 2>{os.path.join(exp_dir, "stderr.err")} &"
        
        try:
            print()
            print('------------------------------------------------')
            print(f"Starting experiment {idx+1}/{len(model_param_list)}")
            end = subprocess.run(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print(str(e)) 