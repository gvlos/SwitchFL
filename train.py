from flatland_tools.env import Environment, Node
from flatland_tools.agent import TQLearningAgent
from training_utils.training_utils import train, eval_once, log2file
from typing import Tuple
from flatland.envs.agent_utils import TrainState
import pandas as pd
import numpy as np
import concurrent.futures
import os
import uuid
import time
import multiprocessing

class ModifiedEnv(Environment):
    """
   A wrapper to the Flatland environment that produces a slightly different observation:
    - station_id: the station identifier of the target station
    - node_id: the node identifier of the current node
    - delay: the delay of the train
    - semaphore_edge_0: 
        - 0: no train coming towards me in edge 0
        - 1: train coming towards me in edge 0
        - 2: no train coming towards me, but there is a malfunctioning train
    - semaphore_edge_1:
        same as semaphore_edge_0 but for edge 1
        
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def observe(self):
        """
        Returns [(train_id, node_id, obs, old_reward)]
        """
        observations = []
        for train_id, node in self.graph.trainsThatAreOnNodes(self.rail_env.agents).items():
            node_id = self.node_to_id[node]
            obs = self.observe_node_v2(node, train_id)
            self.last_obs[node_id] = obs
            old_reward = self._get_reward_for_node(node)
            observations.append((train_id, node_id, obs, old_reward))
        return observations

    def observe_node_v2(self, node: Node, train_id: int) -> Tuple[int, int, int, int, int]:
        """
        Preconditions: train_id exists and is on a node
        Returns (station_id, node_id, delay, semaphore edge 0, sempahore edge 1)
        """
        train = self.rail_env.agents[train_id]

        # Station identifier
        station_id = self.stations[train.target]

        # Train delay
        delay = self._discretize_delay(train_id, self._compute_delay(train_id))

        # Semaphores
        sem_0 = self.compute_semaphore_v2(node, 0)
        sem_1 = self.compute_semaphore_v2(node, 1)

        return station_id, self.node_to_id[node], delay, sem_0, sem_1
    
    def compute_semaphore_v2(self, node: Node, edge_idx: int) -> int:
        """
        Returns the semaphore value for the given node and edge index.
            - 0: no train coming towards me in edge 0
            - 1: train coming towards me in edge 0
            - 2: no train coming towards me, but there is a malfunctioning train
        """
        _, edge_data = self.graph.fw_star(node)[edge_idx]
        malf_present = False
        for train in self.rail_env.agents:
            row, col = train.position if train.position is not None else train.initial_position
            if row == node.row and col == node.col:
                continue
            # if train is currently on this edge and its direction is opposite to the one that is available on this edge starting from the given node
            # or if train is malfunctioning
            # semaphore is activated (True)
            train_on_edge = edge_data.direction_mask[row, col] != -1
            opposite_dir = train.direction != edge_data.direction_mask[row, col]
            malfunction = train.state == TrainState.MALFUNCTION
            if train_on_edge and opposite_dir:
                return 1
            elif malfunction:
                malf_present = True
        return 2 if malf_present else 0
    
def generate_env(malf_rate, malf_min, malf_max, malf_seed=0):
    """
    Returns a modified environment with the given malfunction parameters

    Parameters
    ----------
    malf_rate: float
        The malfunction rate
    malf_min: int
        The minimum malfunction duration
    malf_max: int
        The maximum malfunction duration
    malf_seed: int
        The seed for the malfunction generator

    Returns
    -------
    ModifiedEnv
        The modified environment
    """
    return ModifiedEnv(
        env_width=150,
        env_height=150,
        env_n_cities=37,
        env_n_trains=400,
        seed=13,
        destination_bonus=200,
        deadlock_penalty=-200,
        delay_threshold=0.2,
        malfunction_rate=malf_rate,
        malfunction_min_duration=malf_min,
        malfunction_max_duration=malf_max,
        malfunction_seed=malf_seed
    )

def train_launcher(*args, **kwargs):
    """
    Wrapper for train function that returns metadata
    """
    metadata = kwargs.pop('metadata')
    train(*args, **kwargs)
    return metadata 

def eval_batch(
    malf_rate: float,
    malf_min: int,
    malf_max: int,
    agent: TQLearningAgent,
    malf_seeds: np.ndarray[int],
    exp_id: int,
    first_eval_id: int,
    out_dir: str
):
    """
    Evaluates the agent on a batch of environments with the given malfunction parameters

    Parameters
    ----------
    malf_rate: float
        The malfunction rate
    malf_min: int
        The minimum malfunction duration
    malf_max: int
        The maximum malfunction duration
    agent: TQLearningAgent
        The agent to evaluate
    malf_seeds: np.ndarray[int]
        The seeds for the malfunction generators
    exp_id: int
        The experiment id
    first_eval_id: int
        The id of the first evaluation in this batch
    out_dir: str
        The output directory
    """
    env = generate_env(malf_rate, malf_min, malf_max)
    eval_df = []
    eval_df_columns = \
        ['Experiment id', 'Eval id', 'Eval seed', 'Cumulative reward'] + \
        [f'Delay {i}' for i in range(env.rail_env.get_num_agents())] + \
        ['# arrived', '# arrived on time']
    for idx, malf_seed in enumerate(malf_seeds):
        cumulative_reward, delays, n_arrived, \
            n_arrived_on_time, _ = eval_once(
                env, agent, malf_seed, False
            )
        eval_df.append([
            exp_id,
            first_eval_id + idx,
            malf_seed,
            cumulative_reward,
        ] + [
            delays[train_id] if train_id in delays else None for train_id in range(env.rail_env.get_num_agents())
        ] + [
            n_arrived,
            n_arrived_on_time
        ])
    pd.DataFrame(eval_df, columns=eval_df_columns)\
        .to_parquet(os.path.join(out_dir, f'{uuid.uuid4()}.parquet'))

# Malfunction configurations
malf_configs = pd.DataFrame([
    [0., 0, 0],
    # [1e-3, 5, 15],
    # [1e-3, 15, 30],
    # [5e-3, 5, 15],
    # [5e-3, 15, 30]
], columns=['rate', 'min', 'max'])

# Hyperparameters
hp_configs = pd.DataFrame([
    # Best hps from exp 12
    [1.0, 0.99997, 0.1, 0.99999, int(1e2)],  # blue
    # [1.0, 0.999965, 0.1, 0.99999, int(4e5)],  # orange
    # [1.0, 0.99997, 0.01, 1., int(4e5)],  # green
    # [1.0, 0.999965, 0.01, 1.00000, int(4e5)],  # red
    # [1.0, 0.999975, 0.1, 0.99999, int(4e5)],  # purple
    # [1.0, 0.999975, 0.01, 1, int(4e5)]  # brown
], columns=['epsilon', 'epsilon decay', 'alpha', 'alpha decay', 'n_episodes'])

# Other parameters
out_dir = 'experiments/test_time'
n_workers = multiprocessing.cpu_count()
master_seed = 666
log_every = 10
save_every = 10
n_evals_calc = lambda _: 1 # lambda malf_rate: int(300 / malf_rate)
eval_batch_size = 10_000

if __name__ == '__main__':
    start_time = time.time()
    os.makedirs(out_dir, exist_ok=True)
    master_rng = np.random.RandomState(master_seed)
    master_log_file = os.path.join(out_dir, 'log.txt')
    log2file(master_log_file, f'Starting training with {n_workers} workers.')
    executor = concurrent.futures.ProcessPoolExecutor(n_workers)
    tasks = []

    exp_id = 0
    for hp_idx in range(len(hp_configs)):
        for malf_idx in range(len(malf_configs)):
            exp_id += 1
            hp = hp_configs.iloc[hp_idx]
            malf = malf_configs.iloc[malf_idx]
            env = generate_env(malf['rate'], malf['min'], malf['max'])
            agent = TQLearningAgent()
            # Experiment directory
            exp_dir = os.path.join(out_dir, f'{exp_id}')
            os.makedirs(exp_dir, exist_ok=True)
            # Save config
            pd.concat([hp, malf])\
                .to_csv(os.path.join(exp_dir, f'{exp_id}_config.csv'))
            log2file(master_log_file, f'Experiment {exp_id} started, with hyperparameters:\n{hp}\nand malfunction config:\n{malf}\n')
            # Log file
            log_file = os.path.join(exp_dir, 'log.txt')
            # Q-tables directory
            qtables_out_dir = os.path.join(exp_dir, 'qtables')
            os.makedirs(qtables_out_dir, exist_ok=True)
            # Rng seeds
            train_seed = master_rng.randint(2**32)
            eval_seed = master_rng.randint(2**32)
            np.save(os.path.join(exp_dir, 'train_rng_seed.npy'), train_seed)
            np.save(os.path.join(exp_dir, 'eval_rng_seed.npy'), eval_seed)
            # Launch experiment
            task = executor.submit(
                train_launcher,
                env,
                agent,
                int(hp['n_episodes']),
                alphas=hp['alpha'] * hp['alpha decay'] ** np.arange(hp['n_episodes']),
                epsilons=hp['epsilon'] * hp['epsilon decay'] ** np.arange(hp['n_episodes']),
                with_malfunctions=True,
                save_node_interactions=False,
                rng_master_seed=train_seed,
                log_every=log_every,
                save_every=save_every,
                rewards_out_dir=exp_dir,
                qtables_out_dir=qtables_out_dir,
                log_file=log_file,
                metadata = {
                    'exp_id': exp_id,
                    'hp': hp,
                    'malf': malf,
                    'eval_seed': eval_seed,
                    'agent_path': os.path.join(qtables_out_dir, f'qtable_{int(hp["n_episodes"])}.pkl'),
                    'start_time': time.time()
                }
            )
            tasks.append(task)

    eval_tasks = []
    eval_df_dir = os.path.join(out_dir, 'eval_results')
    os.makedirs(eval_df_dir, exist_ok=True)
    for task in concurrent.futures.as_completed(tasks):
        metadata = task.result()
        malf = metadata['malf']
        n_evals = n_evals_calc(malf['rate'])
        eval_seeds = np.random.RandomState(metadata['eval_seed']).randint(2**32, size=n_evals)
        
        log2file(master_log_file, f'Experiment {metadata["exp_id"]} completed. Took {time.time() - metadata["start_time"]:.2f} seconds')
        log2file(master_log_file, f'Launching {n_evals} evaluations for experiment {metadata["exp_id"]}')

        for eval_id in range(0, n_evals, eval_batch_size):
            eval_task = executor.submit(
                eval_batch,
                malf_rate=malf['rate'],
                malf_min=malf['min'],
                malf_max=malf['max'],
                agent=TQLearningAgent.load(metadata['agent_path']),
                malf_seeds=eval_seeds[eval_id:eval_id+eval_batch_size],
                exp_id=metadata['exp_id'],
                first_eval_id=eval_id,
                out_dir=eval_df_dir
            )
            eval_tasks.append(eval_task)

    concurrent.futures.wait(eval_tasks)
    executor.shutdown()

    log2file(master_log_file, f'All experiments completed. Training and evaluation took {time.time() - start_time:.2f} seconds with {n_workers} workers.')

