import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Flatland imports
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv

import os

# --- Iperparametri ---
LEARNING_RATE = 5e-5
GAMMA = 1.00
EPS_CLIP = 0.2
K_EPOCHS = 20
UPDATE_TIMESTEP = 200
MAX_EPISODES = 5000
CHECKPOINT_INTERVAL = 500  # Salva modelli e dati ogni N episodi

# Flatland environment parameters
GRID_WIDTH = 80
GRID_HEIGHT = 80
N_AGENTS = 15
N_CITIES = 25
MAX_RAILS_BETWEEEN_CITIES = 2
MAX_RAILS_PAIRS_IN_CITY = 2
SEED = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_dir = "./ppo_independent_results"
os.makedirs(out_dir, exist_ok=True)

# --- 1. Definizione dell'Agente PPO (Invariata) ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared Feature Extractor (Opzionale: puoi anche separare completamente le reti)
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh()
        )
        
        # Actor head: produce la distribuzione di probabilità delle azioni
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head: produce lo scalare V(s)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        """
        Implementazione standard del forward pass.
        Restituisce la probabilità delle azioni e il valore dello stato.
        """
        features = self.feature_layer(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        # Utilizza il forward per campionare un'azione
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, state, action):
        # Utilizza il forward per valutare azioni specifiche durante l'update
        action_probs, state_values = self.forward(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, agent_id, lr=3e-4, gamma=0.99, K_epochs=4, eps_clip=0.2, mini_batch_size=64):
        super(PPOAgent, self).__init__()
        self.agent_id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.mini_batch_size = mini_batch_size # Dimensione del minibatch
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.loss_criterion = nn.MSELoss()
        self.buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state_tensor)
        
        self.buffer['states'].append(state_tensor)
        self.buffer['actions'].append(torch.tensor(action).to(device))
        self.buffer['logprobs'].append(action_logprob)
        return action

    def store_outcome(self, reward, is_done):
        self.buffer['rewards'].append(reward)
        self.buffer['is_terminals'].append(is_done)

    def update(self):
        if not self.buffer['rewards']:
            return
        
        # print(f"Aggiornamento Agente {self.agent_id} con {len(self.buffer['states'])} transizioni.")
            
        # 2. Conversione dell'intero buffer in Tensori
        rewards = torch.tensor(self.buffer['rewards'], dtype=torch.float32).to(device).view(-1)
        old_states = torch.stack(self.buffer['states']).detach().to(device)
        old_actions = torch.stack(self.buffer['actions']).detach().to(device).view(-1)
        old_logprobs = torch.stack(self.buffer['logprobs']).detach().to(device).view(-1)

        # 3. Calcolo dell'Advantage sull'intero batch (per stabilità)
        with torch.no_grad():
            _, state_values = self.policy.forward(old_states)
            advantages = rewards - state_values.view(-1)
            
            # --- FIX PER IL WARNING std() ---
            # Controlliamo che ci sia più di un elemento per calcolare lo std correttamente
            if advantages.numel() > 1:
                std = advantages.std()
                # Se lo std è quasi zero (es. tutti i reward uguali), evitiamo la divisione
                if std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (std + 1e-7)
                else:
                    advantages = advantages - advantages.mean()
            else:
                # Se c'è un solo elemento, lo scarto dalla media è per definizione zero
                advantages = advantages - advantages.mean()

        # --- LOGICA MINIBATCH ---
        dataset_size = old_states.size(0)
        indices = np.arange(dataset_size)

        for _ in range(self.K_epochs):
            # Mischiamo gli indici ad ogni epoca
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]
                
                # Estraiamo il minibatch
                mb_states = old_states[batch_idx]
                mb_actions = old_actions[batch_idx]
                mb_logprobs = old_logprobs[batch_idx]
                mb_advantages = advantages[batch_idx]
                mb_rewards = rewards[batch_idx]

                # Valutazione della policy attuale sui campioni del minibatch
                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                
                # Calcolo Ratio e PPO Loss
                ratios = torch.exp(logprobs.view(-1) - mb_logprobs)
                
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                
                loss = -torch.min(surr1, surr2) + \
                       0.5 * self.loss_criterion(state_values.view(-1), mb_rewards) - \
                       0.01 * dist_entropy.mean()

                # Aggiornamento gradienti per questo minibatch
                self.optimizer.zero_grad()
                loss.mean().backward()

                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        # Aggiornamento della vecchia policy e pulizia buffer
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Puliamo il buffer per il prossimo ciclo di raccolta dati
        self.clear_buffer()

    def clear_buffer(self):
        """Svuota la memoria locale dell'agente."""
        for key in self.buffer:
            self.buffer[key] = []

# --- Setup Ambiente e Utility ---
def create_env():
    obs_builder = GlobalObsForRailEnv()
    env = RailEnv(
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        rail_generator=sparse_rail_generator(max_rails_between_cities=MAX_RAILS_BETWEEEN_CITIES, max_rail_pairs_in_city=MAX_RAILS_PAIRS_IN_CITY, 
                                             max_num_cities=N_CITIES, seed=SEED, grid_mode=True),
        line_generator=sparse_line_generator(seed=SEED),
        number_of_agents=N_AGENTS,
        obs_builder_object=obs_builder,
        random_seed=SEED
    )
    return env

def preprocess_obs(obs, width, height):
    # if obs is None:
    #     return np.zeros(width * height * 5)
    # return np.array(obs).flatten()
    obs_flattened = np.concatenate([o.flatten() for o in obs])
    return obs_flattened


# --- Training Loop con Agenti Indipendenti ---
if __name__ == "__main__":
    env = create_env()
    dummy_obs, _ = env.reset()
    
    # Calcolo dimensioni
    if dummy_obs[0] is not None:
        obs_shape = [dummy_obs[0][0].size, dummy_obs[0][1].size, dummy_obs[0][2].size]
        state_dim = np.sum(obs_shape)
    else:
        state_dim = GRID_WIDTH * GRID_HEIGHT * 5 # fallback
        
    action_dim = 5

    # --- MODIFICA CHIAVE: Creazione Dizionario di Agenti ---
    # Creiamo un agente separato per ogni handle (0, 1, ..., N-1)
    agents = {}
    for handle in range(env.get_num_agents()):
        # print(f"Inizializzazione Agente PPO indipendente per il Treno {handle}")
        agents[handle] = PPOAgent(state_dim, action_dim, agent_id=handle)

    timestep_counter = 0
    arrived_trains = np.zeros(MAX_EPISODES)

    for i_episode in range(MAX_EPISODES):
        obs_dict, info_dict = env.reset(regenerate_rail=False, regenerate_schedule=False, random_seed=SEED)
        
        total_rewards = {h: 0 for h in range(N_AGENTS)}
        done_dict = {a: False for a in range(N_AGENTS)}
        done_dict['__all__'] = False


        while True:
            actions_dict = {}
            active_agents_this_step = []
            
        # 1. Chi può agire?
            for handle in range(env.get_num_agents()):
                # Un agente può agire solo se non ha finito l'episodio
                if not done_dict.get(handle, False):
                    # Ottieni osservazione
                    obs = obs_dict.get(handle)
                    if obs is not None:
                        state = preprocess_obs(obs, GRID_WIDTH, GRID_HEIGHT)
                        # Questo aggiunge 1 elemento a states, actions, logprobs
                        action = agents[handle].select_action(state)
                        actions_dict[handle] = action
            
            # 2. Step ambiente
            next_obs_dict, rewards_dict, dones, info = env.step(actions_dict)
            
            for handle in range(env.get_num_agents()):
                if not done_dict.get(handle, False):
                    obs = obs_dict.get(handle)
                    if obs is not None:
                        state = preprocess_obs(obs, GRID_WIDTH, GRID_HEIGHT)
                        action = agents[handle].select_action(state)
                        actions_dict[handle] = action
                        # Segniamoci che questo agente ha appena prodotto uno STATO/AZIONE
                        active_agents_this_step.append(handle)

            next_obs_dict, rewards_dict, dones, info = env.step(actions_dict)

            # SALVATAGGIO REWARD: Solo per chi ha agito!
            for handle in active_agents_this_step:
                reward = rewards_dict[handle]
                is_done = dones[handle]
                # Ora avrai SEMPRE 1 reward per ogni 1 stato
                agents[handle].store_outcome(reward, is_done)

            obs_dict = next_obs_dict
            done_dict = dones

            if dones['__all__']:
                done_dict = dones
                break

            # 4. Aggiornamento PPO (Iteriamo su TUTTI gli agenti)
            if (timestep_counter+1) % UPDATE_TIMESTEP == 0:
                for handle in agents:
                    agents[handle].update()

            obs_dict = next_obs_dict
            done_dict = dones
            timestep_counter += 1
                
        # Calcolo reward medio tra gli agenti per stampare statistiche
        avg_reward = sum(total_rewards.values()) / N_AGENTS
        # print(f"Episodio {i_episode} | Reward Medio Agenti: {avg_reward:.2f} | Rewards: {list(total_rewards.values())}")

        arrived_trains[i_episode] = len([train.handle for train in env.agents \
                if train.position == None and train.arrival_time != None])
        
        if (timestep_counter+1) % CHECKPOINT_INTERVAL == 0:
            np.savez_compressed(os.path.join(out_dir, f'cum_reward_{i_episode}.npz'), x=list(total_rewards.values()))
            np.savez_compressed(os.path.join(out_dir, f'arrived_trains_{i_episode}.npz'), x=arrived_trains)

            # Salvataggio modelli intermedi
            for handle in agents:
                model_path = os.path.join(out_dir, f'ppo_independent_agent_{handle}_checkpoint_{i_episode}.pkl')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(agents[handle].state_dict(), model_path)

    # print("Addestramento Independent PPO completato.")


    np.savez_compressed(os.path.join(out_dir, f'cum_reward.npz'), x=list(total_rewards.values()))
    np.savez_compressed(os.path.join(out_dir, f'arrived_trains.npz'), x=arrived_trains)

    # Salvataggio modelli finali
    for handle in agents:
        model_path = os.path.join(out_dir, f'ppo_independent_agent_{handle}_final.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agents[handle].state_dict(), model_path)