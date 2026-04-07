"""
Energetická optimalizácia inteligentnej budovy - Q-Learning Agent s HEATING + COOLING
Úloha: Naučiť RL agenta optimalizovať spotrebu energie (chladenie a vykurovanie)
Algoritmus: Q-Learning
Porovnanie: Fixná stratégia vs. Q-Learning stratégia
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


# ============================================================================
# KONFIGURÁCIA
# ============================================================================

DATASET_DIR = 'CityLearn/data/datasets/citylearn_challenge_2023_phase_3_3'
BUILDING_FILE = 'Building_1.csv'
WEATHER_FILE = 'weather.csv'

# Q-Learning hyperparametry
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# Trénovanie
TRAIN_EPISODES = 2000
EPISODE_LENGTH = 24 * 7  # Jeden týždeň

# Stavový priestor - binnovánie
N_OCCUPANCY_BINS = 3
N_TEMP_DIFF_BINS = 4  # Rozdiel medzi indoor a outdoor teplotou
N_SOLAR_BINS = 3
N_DEMAND_BINS = 3  # Heating/Cooling demand

# Akčný priestor - HVAC režimy
ACTIONS = {
    0: 'off',           # Bez zásahu (off)
    1: 'low',           # Nízka intenzita (50%)
    2: 'high'           # Vysoká intenzita (100%)
}
INTENSITY_MULTIPLIERS = {0: 0.0, 1: 0.5, 2: 1.0}

RANDOM_SEED = 42


# ============================================================================
# UTILITY FUNKCIE
# ============================================================================

def set_seed(seed):
    np.random.seed(seed)


def get_column(frame, column_name, default_value=0.0):
    """Bezpečne načítaj stĺpec z DataFrame"""
    if column_name in frame.columns:
        return frame[column_name].values
    return np.full(len(frame), default_value, dtype=np.float64)


def digitize_value(value, edges):
    """Mapuj kontinuálnu hodnotu na bin index"""
    return min(len(edges) - 2, max(0, np.digitize(value, edges, right=False) - 1))


def compute_bins(values, n_bins):
    """Vypočítaj bin hrany pre dané hodnoty"""
    if len(np.unique(values)) <= 1:
        return np.array([np.min(values) - 1, np.max(values) + 1])
    return np.quantile(values, np.linspace(0, 1, n_bins + 1))


# ============================================================================
# NAČÍTANIE A PRÍPRAVA ÚDAJOV
# ============================================================================

def load_data(dataset_dir):
    """Načítaj údaje z CityLearn datasetu"""
    building_path = os.path.join(dataset_dir, BUILDING_FILE)
    weather_path = os.path.join(dataset_dir, WEATHER_FILE)
    
    print(f"Načítavam: {building_path}")
    print(f"Načítavam: {weather_path}")
    
    building = pd.read_csv(building_path)
    weather = pd.read_csv(weather_path)
    
    # Výber relevantných premenných - OBOJE heating a cooling
    data = pd.DataFrame({
        'occupant_count': get_column(building, 'occupant_count'),
        'cooling_demand': get_column(building, 'cooling_demand'),
        'heating_demand': get_column(building, 'heating_demand'),
        'non_shiftable_load': get_column(building, 'non_shiftable_load'),
        'dhw_demand': get_column(building, 'dhw_demand'),
        'solar_generation': get_column(building, 'solar_generation'),
        'indoor_temp': get_column(building, 'indoor_dry_bulb_temperature'),
        'indoor_temp_cool_sp': get_column(building, 'indoor_dry_bulb_temperature_cooling_set_point'),
        'indoor_temp_heat_sp': get_column(building, 'indoor_dry_bulb_temperature_heating_set_point'),
        'outdoor_temp': get_column(weather, 'outdoor_dry_bulb_temperature'),
        'outdoor_temp_pred_1': get_column(weather, 'outdoor_dry_bulb_temperature_predicted_1'),
        'solar_irrad': get_column(weather, 'direct_solar_irradiance') + 
                       get_column(weather, 'diffuse_solar_irradiance'),
    })
    
    # Výpočet podstatných features
    data['temp_forecast'] = data['outdoor_temp_pred_1']
    data['temp_diff'] = data['indoor_temp'] - data['outdoor_temp']  # Rozdiel teplôt
    data['total_hvac_demand'] = data['cooling_demand'] + data['heating_demand']  # Celkový dopyt HVAC
    
    return data


def prepare_state_features(data):
    """Priprav features na binnovánie"""
    features = {
        'occupancy': data['occupant_count'].values.astype(np.int32),
        'temp_diff': data['temp_diff'].values,
        'solar_irrad': data['solar_irrad'].values,
        'hvac_demand': data['total_hvac_demand'].values,
    }
    
    # Návrhové bin hrany
    features['occupancy_edges'] = np.array([0, 1, 2, 3, 4])
    features['temp_diff_edges'] = compute_bins(features['temp_diff'], N_TEMP_DIFF_BINS)
    features['solar_edges'] = compute_bins(features['solar_irrad'], N_SOLAR_BINS)
    features['demand_edges'] = compute_bins(features['hvac_demand'], N_DEMAND_BINS)
    
    print(f"\n  Temp diff rozsah: [{features['temp_diff_edges'][0]:.1f}, {features['temp_diff_edges'][-1]:.1f}]")
    print(f"  Solar irrad rozsah: [{features['solar_edges'][0]:.1f}, {features['solar_edges'][-1]:.1f}]")
    print(f"  HVAC demand rozsah: [{features['demand_edges'][0]:.2f}, {features['demand_edges'][-1]:.2f}]")
    
    return features


def state_to_index(occupancy, temp_diff, solar_irrad, hvac_demand, features):
    """Preveď kontinuálny stav na diskrétny index"""
    occ_bin = digitize_value(occupancy, features['occupancy_edges'])
    temp_bin = digitize_value(temp_diff, features['temp_diff_edges'])
    solar_bin = digitize_value(solar_irrad, features['solar_edges'])
    demand_bin = digitize_value(hvac_demand, features['demand_edges'])
    
    # Kombinovaný index
    state_index = (occ_bin * (N_TEMP_DIFF_BINS * N_SOLAR_BINS * N_DEMAND_BINS) + 
                   temp_bin * (N_SOLAR_BINS * N_DEMAND_BINS) + 
                   solar_bin * N_DEMAND_BINS +
                   demand_bin)
    return state_index


# ============================================================================
# PROSTREDIE (SIMULÁCIA BUDOVY S HEATING + COOLING)
# ============================================================================

class BuildingEnvironment:
    """Simulácia budovy - energia, chladenie a vykurovanie"""
    
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.n_timesteps = len(data)
        self.current_step = 0
        
    def reset(self, start_hour=0):
        """Vyresetuj prostredie na začiatočný čas"""
        self.current_step = start_hour % self.n_timesteps
        
    def step(self, hvac_action):
        """
        Vykonaj akciu HVAC a získaj reward
        
        Logika:
        - Ak je potrebné vykurovanie (heating_demand > 0), aplikuj heating
        - Ak je potrebné chladenie (cooling_demand > 0), aplikuj cooling
        - Akcia určuje intenzitu (0=off, 1=low, 2=high)
        
        Args:
            hvac_action: Index akcie (0=off, 1=low 50%, 2=high 100%)
            
        Returns:
            next_state_info: dict s informáciami o novom stave
            reward: Odmena (negatívna = vysoká spotreba)
            done: Či je epizóda skončená
        """
        if self.current_step >= self.n_timesteps - 1:
            self.current_step = self.n_timesteps - 1
            done = True
        else:
            done = False
        
        # Získaj údaje pre aktuálny čas
        row = self.data.iloc[self.current_step]
        
        # Výber intenzity podľa akcie
        intensity = INTENSITY_MULTIPLIERS.get(hvac_action, 0.0)
        
        # Vypočítaj spotrebu energie - vznik z heating alebo cooling dopytu
        cooling_demand = row['cooling_demand']
        heating_demand = row['heating_demand']
        
        # Rozhodni podľa dopytu: ak je oboje, uprednostni väčšie
        if cooling_demand > heating_demand:
            # Chladenie je viac potrebované
            hvac_energy = cooling_demand * intensity
            hvac_mode = 'cooling'
        else:
            # Vykurovanie je viac potrebované alebo rovnaké
            hvac_energy = heating_demand * intensity
            hvac_mode = 'heating'
        
        # Celková spotreba energie
        total_energy = (row['non_shiftable_load'] + 
                       row['dhw_demand'] + 
                       hvac_energy)
        
        # Komfort: penalizácia ak HVAC je vypnutý ale je veľký dopyt
        total_hvac_demand = cooling_demand + heating_demand
        if hvac_action == 0 and total_hvac_demand > 0.5:
            comfort_penalty = -total_hvac_demand * 0.2  # Penalizácia za nepohodlie
        else:
            comfort_penalty = 0
        
        # Reward: negatívna energia (chceme minimalizovať) + komfort
        energy_penalty = -total_energy
        reward = energy_penalty + comfort_penalty
        
        self.current_step += 1
        
        # Stav ďalšieho kroku
        if self.current_step < self.n_timesteps:
            next_row = self.data.iloc[self.current_step]
            next_state_info = {
                'occupancy': next_row['occupant_count'],
                'temp_diff': next_row['temp_diff'],
                'solar_irrad': next_row['solar_irrad'],
                'hvac_demand': next_row['total_hvac_demand'],
            }
        else:
            next_state_info = None
        
        return next_state_info, reward, done, total_energy
    
    def get_current_state(self):
        """Aktuálny stav prostredia"""
        row = self.data.iloc[self.current_step]
        return {
            'occupancy': row['occupant_count'],
            'temp_diff': row['temp_diff'],
            'solar_irrad': row['solar_irrad'],
            'hvac_demand': row['total_hvac_demand'],
        }


# ============================================================================
# Q-LEARNING AGENT
# ============================================================================

class QLearningAgent:
    """Q-Learning agent pre optimalizáciu energie (heating + cooling)"""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount=0.95):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount = discount
        
        # Q-tabuľka inicializovaná na 0
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        
        self.epsilon = EPSILON_START
        
    def choose_action(self, state, training=True):
        """Epsilon-greedy výber akcie"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self.Q[state]
            return np.argmax(q_values)
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update"""
        current_q = self.Q[state][action]
        
        if done:
            max_next_q = 0
        else:
            max_next_q = np.max(self.Q[next_state])
        
        new_q = current_q + self.learning_rate * (reward + self.discount * max_next_q - current_q)
        self.Q[state][action] = new_q
    
    def decay_epsilon(self):
        """Zníž exploráciu"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# ============================================================================
# FIXNÉ STRATÉGIE (BASELINE)
# ============================================================================

class FixedBaselinePolicy:
    """Fixná stratégia na porovnanie"""
    
    def __init__(self, action=1):  # 1 = low (stredný režim)
        self.action = action
    
    def choose_action(self, state):
        """Vždy vykonaj tú istú akciu"""
        return self.action


# ============================================================================
# TRÉNOVANIE A EVALUÁCIA
# ============================================================================

def train_q_learning(env, agent, features, n_episodes=100):
    """Trénovanie Q-Learning agenta"""
    episode_rewards = []
    episode_energies = []
    
    for episode in tqdm(range(n_episodes), desc="Trénovanie Q-Learning"):
        env.reset(start_hour=0)
        
        episode_reward = 0
        episode_energy = 0
        
        for step in range(EPISODE_LENGTH):
            # Aktuálny stav
            state_info = env.get_current_state()
            state_idx = state_to_index(
                state_info['occupancy'],
                state_info['temp_diff'],
                state_info['solar_irrad'],
                state_info['hvac_demand'],
                features
            )
            
            # Výber akcie
            action = agent.choose_action(state_idx, training=True)
            
            # Vykonaj akciu
            next_state_info, reward, done, total_energy = env.step(action)
            
            # Ďalší stav
            if next_state_info is not None:
                next_state_idx = state_to_index(
                    next_state_info['occupancy'],
                    next_state_info['temp_diff'],
                    next_state_info['solar_irrad'],
                    next_state_info['hvac_demand'],
                    features
                )
            else:
                next_state_idx = state_idx
            
            # Q-Learning update
            agent.update(state_idx, action, reward, next_state_idx, done)
            
            episode_reward += reward
            episode_energy += total_energy
            
            if done:
                break
        
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        episode_energies.append(episode_energy)
    
    return episode_rewards, episode_energies


def evaluate_policy(env, policy_func, features, n_episodes=10, training=False):
    """Evaluuj politiku (Q-Learning alebo Fixnú)"""
    total_energy = 0
    total_reward = 0
    episode_energies = []
    
    for episode in range(n_episodes):
        env.reset(start_hour=0)
        episode_energy = 0
        episode_reward = 0
        
        for step in range(EPISODE_LENGTH):
            state_info = env.get_current_state()
            state_idx = state_to_index(
                state_info['occupancy'],
                state_info['temp_diff'],
                state_info['solar_irrad'],
                state_info['hvac_demand'],
                features
            )
            
            action = policy_func(state_idx)
            next_state_info, reward, done, total_energy = env.step(action)
            
            episode_energy += total_energy
            episode_reward += reward
            
            if done:
                break
        
        total_energy += episode_energy
        total_reward += episode_reward
        episode_energies.append(episode_energy)
    
    avg_energy = total_energy / n_episodes
    avg_reward = total_reward / n_episodes
    
    return {
        'avg_energy': avg_energy,
        'avg_reward': avg_reward,
        'episode_energies': episode_energies
    }


# ============================================================================
# HLAVNÁ FUNKCIA
# ============================================================================

def main():
    print("="*70)
    print("Energetická optimalizácia - Q-Learning s HEATING + COOLING")
    print("="*70)
    
    set_seed(RANDOM_SEED)
    
    # 1. NAČÍTANIE ÚDAJOV
    print("\n[1] Načítavam údaje...")
    data = load_data(DATASET_DIR)
    print(f"    Počet timestepov: {len(data)}")
    print(f"    Heating demand - priemer: {data['heating_demand'].mean():.4f}, max: {data['heating_demand'].max():.4f}")
    print(f"    Cooling demand - priemer: {data['cooling_demand'].mean():.4f}, max: {data['cooling_demand'].max():.4f}")
    
    # 2. PRÍPRAVA STAVOV
    print("\n[2] Pripravujem stavový priestor...")
    features = prepare_state_features(data)
    n_states = (N_OCCUPANCY_BINS * N_TEMP_DIFF_BINS * N_SOLAR_BINS * N_DEMAND_BINS)
    n_actions = len(ACTIONS)
    print(f"    Počet stavov: ~{n_states}")
    print(f"    Počet akcií: {n_actions} {list(ACTIONS.values())}")
    
    # 3. INICIALIZÁCIA
    print("\n[3] Inicializujem Q-Learning agenta...")
    env = BuildingEnvironment(data)
    q_agent = QLearningAgent(n_states, n_actions, 
                              learning_rate=LEARNING_RATE,
                              discount=DISCOUNT_FACTOR)
    
    # 4. TRÉNOVANIE Q-LEARNING
    print(f"\n[4] Trénujem Q-Learning agenta ({TRAIN_EPISODES} epizód)...")
    train_rewards, train_energies = train_q_learning(env, q_agent, features, TRAIN_EPISODES)
    
    print(f"    Priemerná energia v tréningovej fáze: {np.mean(train_energies[-10:]):.4f}")
    
    # 5. EVALUÁCIA - Q-LEARNING
    print(f"\n[5] Evaluujem Q-Learning agenta...")
    q_policy = lambda state: q_agent.choose_action(state, training=False)
    q_results = evaluate_policy(env, q_policy, features, n_episodes=20, training=False)
    
    print(f"    Priemerná spotreba energie: {q_results['avg_energy']:.4f} J")
    
    # 6. EVALUÁCIA - FIXNÉ STRATÉGIE
    print(f"\n[6] Testovanie fixných stratégií (baseline)...")
    baseline_results = {}
    
    for action_id, action_name in ACTIONS.items():
        fixed_func = lambda state, a=action_id: a
        
        results = evaluate_policy(env, fixed_func, features, n_episodes=20)
        baseline_results[action_name] = results
        print(f"    {action_name:12s} - Spotreba: {results['avg_energy']:.4f} J")
    
    # 7. POROVNANIE VÝSLEDKOV
    print(f"\n[7] POROVNANIE ENERGETICKÝCH ÚSPOR")
    print("="*70)
    
    # Najlepšia fixná stratégia
    best_fixed = min(baseline_results.items(), 
                    key=lambda x: x[1]['avg_energy'])
    best_fixed_name, best_fixed_results = best_fixed
    
    energy_savings = best_fixed_results['avg_energy'] - q_results['avg_energy']
    energy_savings_pct = (energy_savings / best_fixed_results['avg_energy']) * 100 if best_fixed_results['avg_energy'] > 0 else 0
    
    print(f"\nQ-Learning spotreba:        {q_results['avg_energy']:.4f} J")
    print(f"Najlepšia fixná stratégia:  {best_fixed_results['avg_energy']:.4f} J ({best_fixed_name})")
    print(f"Absolútne úspory:           {energy_savings:.4f} J")
    print(f"Relatívne úspory:           {energy_savings_pct:.2f}%")
    
    if energy_savings > 0:
        print(f"\n✓ Q-Learning je efektívnejší o {energy_savings_pct:.2f}%!")
    else:
        print(f"\n✗ Fixná stratégia ({best_fixed_name}) je efektívnejšia.")
    
    # Porovnanie rewards
    print(f"\n[8] DETAILNÉ POROVNANIE VÝKONNOSTI")
    print("="*70)
    print(f"{'Metóda':<15} {'Energia (J)':<15} {'Reward':<15}")
    print("-"*70)
    for name, results in baseline_results.items():
        print(f"{name:<15} {results['avg_energy']:>14.4f} {results['avg_reward']:>14.4f}")
    print(f"{'Q-Learning':<15} {q_results['avg_energy']:>14.4f} {q_results['avg_reward']:>14.4f}")
    
    # 9. VIZUALIZÁCIA
    print(f"\n[9] Generujem grafy...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Graf 1: Trénovacia krivka
    axes[0, 0].plot(train_energies, linewidth=2)
    axes[0, 0].fill_between(range(len(train_energies)), train_energies, alpha=0.3)
    axes[0, 0].set_title('Energia počas tréningu Q-Learning', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epizóda')
    axes[0, 0].set_ylabel('Energia [J]')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Graf 2: Porovnanie energií
    methods = list(baseline_results.keys()) + ['Q-Learning']
    energies = [baseline_results[m]['avg_energy'] for m in baseline_results.keys()] + [q_results['avg_energy']]
    colors = ['#ff9999', '#ffcc99', '#ccffcc', '#99ff99']
    
    bars = axes[0, 1].bar(methods, energies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_title('Porovnanie energetickej spotreby', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Energia [J]')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Hodnoty na baroch
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{energy:.1f}',
                       ha='center', va='bottom', fontsize=9)
    
    # Graf 3: Rozdelenie energií Q-Learning
    axes[1, 0].hist(q_results['episode_energies'], bins=10, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].set_title('Distribúcia energií - Q-Learning', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Energia [J]')
    axes[1, 0].set_ylabel('Frekvencia')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Graf 4: Relatívne úspory
    savings_pct = []
    for m in baseline_results.keys():
        if baseline_results[m]['avg_energy'] > 0:
            saving = ((baseline_results[m]['avg_energy'] - q_results['avg_energy']) / 
                     baseline_results[m]['avg_energy'] * 100)
        else:
            saving = 0
        savings_pct.append(saving)
    
    colors_savings = ['green' if s > 0 else 'red' for s in savings_pct]
    axes[1, 1].barh(list(baseline_results.keys()), savings_pct, color=colors_savings, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_title('Úspory Q-Learning oproti fixným stratégiám', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Úspory [%]')
    axes[1, 1].axvline(x=0, color='black', linewidth=1)
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # Hodnoty na baroch
    for i, (bar, saving) in enumerate(zip(axes[1, 1].patches, savings_pct)):
        width = bar.get_width()
        axes[1, 1].text(width, bar.get_y() + bar.get_height()/2.,
                       f'{saving:.1f}%',
                       ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('energy_optimization_comparison_heating_cooling.png', dpi=100, bbox_inches='tight')
    print("    ✓ Uložený: energy_optimization_comparison_heating_cooling.png")
    
    print("\n" + "="*70)
    print("SKONČENÉ - Q-Learning s podporou HEATING a COOLING")
    print("="*70)


if __name__ == '__main__':
    main()
