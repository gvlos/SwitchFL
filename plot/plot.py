import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from os import listdir
import glob

plt.style.use('./fig.mplstyle')


def running_avg(x, ws):
    '''Computes a running average over x'''
    # ws = int(len(x)/n_windows)  # size of each window
    return [np.sum(x[i*ws:i*ws+ws])/ws for i in range(int(len(x)/ws))]


decay_name = {
    "0.99997" : "fast",
    "0.999965" : "medium",
    "0.999975" : "faster",
    "0.99999" : "slow",
    "1.0" : "none",
    # "0.9999899999999999": "strange"
}


if __name__=='__main__':

    resdir = '../experiments/reproduce_deterministic'

    parq_list = glob.glob(join(resdir, '*/tr_rewards.parquet'))
    config_list = glob.glob(join(resdir, '*/*_config.csv'))


    config_list = [
    '../experiments/reproduce_deterministic/1/1_config.csv', 
    '../experiments/reproduce_deterministic/2/2_config.csv',  
    '../experiments/reproduce_deterministic/3/3_config.csv',
    '../experiments/reproduce_deterministic/4/4_config.csv',
    '../experiments/reproduce_deterministic/5/5_config.csv',
    ]

    ws = 5000

    plt.figure(figsize=(15,7), dpi=80)

    for parq, config in zip(parq_list, config_list):
        rewards = pd.read_parquet(parq)["Cumulative reward"].to_numpy()
        mv_avg_rewards = running_avg(rewards, ws)

        params = pd.read_csv(config, names=['param', 'value'])
        params.drop(0, inplace=True)
        params = dict(zip(params["param"].to_list(), params["value"].to_list()))
        print(params)

        x_axis = ws * np.arange(len(mv_avg_rewards))
        label = rf'$\epsilon_{{\text{{decay}}}}$ = {decay_name[str(params["epsilon decay"])]}, ' + \
        rf'$\alpha$={params["alpha"]}, ' + \
        rf'$\alpha_{{\text{{decay}}}}$ = {decay_name[str(params["alpha decay"])]}'
        plt.plot(x_axis, mv_avg_rewards, label=label)


    plt.hlines(1000, xmin=0, xmax=4e5, colors='black', ls='--', lw=2, label='Upper bound')
    plt.hlines(800, xmin=0, xmax=4e5, colors='grey', ls='--', lw=2, label='Empirically good')
    plt.grid(True, which="both", lw=0.5, ls='--')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward')
    plt.tight_layout()
    plt.savefig('training.png')