import numpy as np
import torch
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.observations import GlobalObsForRailEnv

# --- IMPORTA LE TUE CLASSI (Assicurati che siano uguali al training) ---
# Se le hai nello stesso file non serve, altrimenti importale.
# Qui le rimetto in versione ridotta per completezza.
from ppo_baseline import ActorCritic, preprocess_obs
import os


# Flatland environment parameters
GRID_WIDTH = 25
GRID_HEIGHT = 25
N_AGENTS = 2
N_CITIES = 15
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

def test_agents(render=True):
    # 1. Setup Ambiente (stessa configurazione del training)
    env = create_env()
    obs_dict, _ = env.reset(random_seed=SEED)
    dummy_obs = preprocess_obs(obs_dict[0], GRID_WIDTH, GRID_HEIGHT)
    state_dim = len(dummy_obs)
    action_dim = 5

    # 2. Caricamento Modelli
    models = {}
    for h in range(N_AGENTS):
        model = ActorCritic(state_dim, action_dim).to(device)
        try:
            model.load_state_dict(torch.load(f"/home/gianvito/Documents/SwitchFL-1/ppo_new_results/ippo_agent_{h}_final.pth", map_location=device))
            model.eval() # Modalità valutazione: disattiva dropout/batchnorm
            models[h] = model
            print(f"Modello Agente {h} caricato con successo.")
        except FileNotFoundError:
            print(f"Attenzione: file ippo_agent_{h}_final.pth non trovato. L'agente userà pesi casuali.")
            models[h] = model

    # 3. Ciclo di Valutazione
    num_test_episodes = 1
    for ep in range(num_test_episodes):
        obs_dict, _ = env.reset(random_seed=SEED)
        done_dict = {i: False for i in range(N_AGENTS)}
        done_dict['__all__'] = False
        
        total_rewards = 0
        agents_arrived = 0
        steps = 0

        while not done_dict['__all__']:
            actions_dict = {}
            
            for h in range(N_AGENTS):
                if not done_dict.get(h, False):
                    obs = obs_dict.get(h)
                    if obs is not None:
                        state = preprocess_obs(obs, GRID_WIDTH, GRID_HEIGHT)
                        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
                        
                        with torch.no_grad():
                            # Scegliamo l'azione con la probabilità più alta (Greedy)
                            action_probs, _ = models[h](state_tensor)
                            action = torch.argmax(action_probs, dim=-1).item()
                            actions_dict[h] = action
            
            next_obs_dict, rewards_dict, dones, info = env.step(actions_dict)
            
            for h in range(N_AGENTS):
                total_rewards += rewards_dict.get(h, 0)
                # Verifichiamo se l'agente è arrivato (completato il percorso)
                if dones[h] and not done_dict[h]:
                    agents_arrived += 1
            
            obs_dict = next_obs_dict
            done_dict = dones
            steps += 1

        print(f"Test Episodio {ep+1} | Arrivati: {agents_arrived}/{N_AGENTS} | Reward Totale: {total_rewards:.2f} | Step: {steps}")

        delays = []
        for train in env.agents:
            delay = train.get_current_delay(elapsed_steps=steps, distance_map=env.distance_map)
            delays.append(delay)
            print(f"Agente {train.handle} | Ritardo: {delay}")

        with open(os.path.join(out_dir, f"/home/gianvito/Documents/SwitchFL-1/ppo_new_results/final_delays_eval.npy"), 'wb') as f:
            np.save(f, delays)
        

if __name__ == "__main__":
    test_agents()