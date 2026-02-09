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
from tqdm import tqdm
import os

# Flatland environment parameters
GRID_WIDTH = 80
GRID_HEIGHT = 80
N_AGENTS = 15
N_CITIES = 25
MAX_RAILS_BETWEEEN_CITIES = 2
MAX_RAILS_PAIRS_IN_CITY = 2
SEED = 64

MAX_EPISODES = 2
CHECKPOINT_INTERVAL = 500  # Salva modelli e dati ogni N episodi
UPDATE_TIMESTEP = 200  # Frequenza aggiornamento (step totali)
K_EPOCHS = 4        # Numero di epoche per ogni update
MINI_BATCH_SIZE = 64   # Dimensione dei minibatch
LEARNING_RATE = 1e-4   # LR ridotto per stabilità
GAMMA = 0.99
EPS_CLIP = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_dir = "./ppo_new_results"
os.makedirs(out_dir, exist_ok=True)

# --- 1. ARCHITETTURA ACTOR-CRITIC ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Feature extractor comune
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh()
        )
        
        # Actor head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (Value function)
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        features = self.feature_layer(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value
    
    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def evaluate(self, state, action):
        action_probs, state_values = self.forward(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, state_values, dist_entropy

# --- 2. AGENTE PPO INDIPENDENTE ---
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim, agent_id):
        super(PPOAgent, self).__init__()
        self.agent_id = agent_id
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
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
        # 0. Sincronizzazione Buffer (Risolve il problema Size Mismatch)
        n_states = len(self.buffer['states'])
        n_rewards = len(self.buffer['rewards'])
        min_len = min(n_states, n_rewards)
        
        if min_len < 2: return # Troppo pochi dati per l'update

        # Taglio dei dati alla lunghezza minima comune
        b_states = torch.stack(self.buffer['states'][:min_len]).detach().to(device)
        b_actions = torch.stack(self.buffer['actions'][:min_len]).detach().to(device).view(-1)
        b_logprobs = torch.stack(self.buffer['logprobs'][:min_len]).detach().to(device).view(-1)
        b_rewards_raw = self.buffer['rewards'][:min_len]
        b_terms = self.buffer['is_terminals'][:min_len]

        # 1. Calcolo dei Ritorni (G_t)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(b_rewards_raw), reversed(b_terms)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device).view(-1)

        # 2. Calcolo Advantage Globale (Anti-NaN e Anti-Warning std)
        with torch.no_grad():
            _, state_values = self.policy.forward(b_states)
            advantages = rewards - state_values.view(-1)
            if advantages.numel() > 1:
                std = advantages.std()
                if std > 1e-8:
                    advantages = (advantages - advantages.mean()) / (std + 1e-7)
            else:
                advantages = advantages - advantages.mean()

        # 3. Minibatch SGD
        indices = np.arange(min_len)
        for _ in range(K_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, min_len, MINI_BATCH_SIZE):
                idx = indices[start : start + MINI_BATCH_SIZE]
                if len(idx) < 2: continue # Salta minibatch troppo piccoli per std()

                # Estrazione minibatch
                mb_states = b_states[idx]
                mb_actions = b_actions[idx]
                mb_logprobs = b_logprobs[idx]
                mb_advantages = advantages[idx]
                mb_rewards = rewards[idx]

                # Valutazione
                logprobs, state_values, dist_entropy = self.policy.evaluate(mb_states, mb_actions)
                
                # PPO Loss
                ratios = torch.exp(logprobs.view(-1) - mb_logprobs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * mb_advantages
                
                loss = -torch.min(surr1, surr2) + \
                       0.5 * self.loss_criterion(state_values.view(-1), mb_rewards) - \
                       0.01 * dist_entropy.mean()

                # Backpropagation con Gradient Clipping (Anti-NaN)
                self.optimizer.zero_grad()
                if not torch.isnan(loss.mean()):
                    loss.mean().backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                    self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': []}

# --- 3. UTILITY ---
def create_env():
    obs_builder = GlobalObsForRailEnv()
    return RailEnv(
        width=GRID_WIDTH, height=GRID_HEIGHT,
        rail_generator=sparse_rail_generator(max_num_cities=2, seed=SEED),
        line_generator=sparse_line_generator(),
        number_of_agents=N_AGENTS,
        obs_builder_object=obs_builder,
        random_seed=SEED
    )

def preprocess_obs(obs, width, height):
    obs_flattened = np.concatenate([o.flatten() for o in obs])
    return obs_flattened

# --- 4. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    env = create_env()
    obs_dict, _ = env.reset()
    
    # Inizializzazione dimensioni e agenti
    dummy_obs = preprocess_obs(obs_dict[0], GRID_WIDTH, GRID_HEIGHT)
    state_dim = len(dummy_obs)
    action_dim = 5
    
    agents = {h: PPOAgent(state_dim, action_dim, h) for h in range(N_AGENTS)}
    
    print(f"Training iniziato. Device: {device}")
    total_step_counter = 0
    arrived_trains = np.zeros(MAX_EPISODES)
    cum_reward = []


    for i_episode in tqdm(range(MAX_EPISODES)):
        obs_dict, _ = env.reset()
        done_dict = {i: False for i in range(N_AGENTS)}
        done_dict['__all__'] = False
        ep_reward = {i: 0 for i in range(N_AGENTS)}

        while not done_dict['__all__']:
            actions_dict = {}
            active_this_step = []

            # 1. Scelta Azioni
            for h in range(N_AGENTS):
                if not done_dict.get(h, False):
                    obs = obs_dict.get(h)
                    if obs is not None:
                        state = preprocess_obs(obs, GRID_WIDTH, GRID_HEIGHT)
                        action = agents[h].select_action(state)
                        actions_dict[h] = action
                        active_this_step.append(h)

            # 2. Step Ambiente
            next_obs_dict, rewards_dict, dones, _ = env.step(actions_dict)

            # 3. Memorizzazione Sincronizzata
            for h in active_this_step:
                agents[h].store_outcome(rewards_dict[h], dones[h])
                ep_reward[h] += rewards_dict[h]

            obs_dict = next_obs_dict
            done_dict = dones
            total_step_counter += 1

            # 4. Update Periodico
            if total_step_counter % UPDATE_TIMESTEP == 0:
                for h in agents:
                    agents[h].update()

        if i_episode % 5 == 0:
            avg_r = sum(ep_reward.values()) / N_AGENTS
            print(f"Episodio {i_episode} | Reward Medio: {avg_r:.2f}")

        cum_reward.append(ep_reward)

        arrived_trains[i_episode] = len([train.handle for train in env.agents \
                if train.position == None and train.arrival_time != None])
        
        if (total_step_counter+1) % CHECKPOINT_INTERVAL == 0:
            np.savez_compressed(os.path.join(out_dir, f'cum_reward_{i_episode}.npz'), x=cum_reward)
            np.savez_compressed(os.path.join(out_dir, f'arrived_trains_{i_episode}.npz'), x=arrived_trains)

            # Salvataggio modelli intermedi
            for handle in agents:
                model_path = os.path.join(out_dir, f'ippo_agent_{handle}_checkpoint_{i_episode}.pkl')
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(agents[handle].state_dict(), model_path)

    # print("Addestramento Independent PPO completato.")


    np.savez_compressed(os.path.join(out_dir, f'cum_reward.npz'), x=cum_reward)
    np.savez_compressed(os.path.join(out_dir, f'arrived_trains.npz'), x=arrived_trains)

    # Salvataggio modelli finali
    for handle in agents:
        model_path = os.path.join(out_dir, f'ippo_agent_{handle}_final.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(agents[handle].state_dict(), model_path)

    print("Addestramento completato!")