"""
Energy optimization of a smart building - Q-Learning (version 03)
Goal: Train RL agent using ONLY occupancy + weather forecasts as state inputs.
Algorithm: Q-Learning
Comparison: Fixed baseline strategies vs Q-Learning
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================================
# CONFIG
# ============================================================================

DATASET_DIR = 'CityLearn/data/datasets/citylearn_challenge_2023_phase_3_3'
BUILDING_FILE = 'Building_1.csv'
WEATHER_FILE = 'weather.csv'

RANDOM_SEED = 42

# Training
TRAIN_EPISODES = 3000
EPISODE_LENGTH = 24 * 7  # one week

# Q-learning
ALPHA = 0.10
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995

# Discretization (state built only from occupancy + weather forecasts)
N_OCCUPANCY_BINS = 3
N_TEMP_MEAN_BINS = 5
N_TEMP_TREND_BINS = 3
N_SOLAR_MEAN_BINS = 4
N_HUMIDITY_MEAN_BINS = 3

# Actions: HVAC intensity only
ACTIONS = {
    0: 'off',
    1: 'low',
    2: 'high',
}
INTENSITY = {
    0: 0.0,
    1: 0.5,
    2: 1.0,
}


# ============================================================================
# HELPERS
# ============================================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def get_column(frame: pd.DataFrame, column_name: str, default_value: float = 0.0) -> np.ndarray:
    if column_name in frame.columns:
        return frame[column_name].to_numpy(dtype=np.float64)
    return np.full(len(frame), default_value, dtype=np.float64)


def compute_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        if v_min == v_max:
            v_max = v_min + 1e-6
        edges = np.linspace(v_min, v_max, n_bins + 1)
    return edges


def bin_value(value: float, edges: np.ndarray) -> int:
    idx = int(np.digitize(value, edges[1:-1], right=False))
    return max(0, min(idx, len(edges) - 2))


def moving_average(values, window=30):
    if window <= 1:
        return np.asarray(values)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode='same')


# ============================================================================
# DATA
# ============================================================================

def load_data(dataset_dir: str) -> pd.DataFrame:
    building_path = os.path.join(dataset_dir, BUILDING_FILE)
    weather_path = os.path.join(dataset_dir, WEATHER_FILE)

    print(f'Loading building file: {building_path}')
    print(f'Loading weather file:  {weather_path}')

    building = pd.read_csv(building_path)
    weather = pd.read_csv(weather_path)

    # Weather forecast inputs (only these + occupancy go into state)
    t1 = get_column(weather, 'outdoor_dry_bulb_temperature_predicted_1')
    t2 = get_column(weather, 'outdoor_dry_bulb_temperature_predicted_2', default_value=float(np.mean(t1)))
    t3 = get_column(weather, 'outdoor_dry_bulb_temperature_predicted_3', default_value=float(np.mean(t2)))

    h1 = get_column(weather, 'outdoor_relative_humidity_predicted_1')
    h2 = get_column(weather, 'outdoor_relative_humidity_predicted_2', default_value=float(np.mean(h1)))
    h3 = get_column(weather, 'outdoor_relative_humidity_predicted_3', default_value=float(np.mean(h2)))

    d1 = get_column(weather, 'diffuse_solar_irradiance_predicted_1')
    d2 = get_column(weather, 'diffuse_solar_irradiance_predicted_2', default_value=float(np.mean(d1)))
    d3 = get_column(weather, 'diffuse_solar_irradiance_predicted_3', default_value=float(np.mean(d2)))

    r1 = get_column(weather, 'direct_solar_irradiance_predicted_1')
    r2 = get_column(weather, 'direct_solar_irradiance_predicted_2', default_value=float(np.mean(r1)))
    r3 = get_column(weather, 'direct_solar_irradiance_predicted_3', default_value=float(np.mean(r2)))

    solar1 = d1 + r1
    solar2 = d2 + r2
    solar3 = d3 + r3

    df = pd.DataFrame({
        # State inputs
        'occupant_count': get_column(building, 'occupant_count'),
        'temp_pred_1': t1,
        'temp_pred_2': t2,
        'temp_pred_3': t3,
        'humidity_pred_1': h1,
        'humidity_pred_2': h2,
        'humidity_pred_3': h3,
        'solar_pred_1': solar1,
        'solar_pred_2': solar2,
        'solar_pred_3': solar3,

        # Environment/reward internal signals (not used as state)
        'cooling_demand': get_column(building, 'cooling_demand'),
        'heating_demand': get_column(building, 'heating_demand'),
        'non_shiftable_load': get_column(building, 'non_shiftable_load'),
        'dhw_demand': get_column(building, 'dhw_demand'),
    })

    # Derived features used in state encoding
    df['forecast_temp_mean'] = df[['temp_pred_1', 'temp_pred_2', 'temp_pred_3']].mean(axis=1)
    df['forecast_temp_trend'] = df['temp_pred_3'] - df['temp_pred_1']
    df['forecast_solar_mean'] = df[['solar_pred_1', 'solar_pred_2', 'solar_pred_3']].mean(axis=1)
    df['forecast_humidity_mean'] = df[['humidity_pred_1', 'humidity_pred_2', 'humidity_pred_3']].mean(axis=1)

    return df


def build_state_edges(df: pd.DataFrame):
    return {
        'occupancy_edges': compute_bins(df['occupant_count'].to_numpy(), N_OCCUPANCY_BINS),
        'temp_mean_edges': compute_bins(df['forecast_temp_mean'].to_numpy(), N_TEMP_MEAN_BINS),
        'temp_trend_edges': compute_bins(df['forecast_temp_trend'].to_numpy(), N_TEMP_TREND_BINS),
        'solar_mean_edges': compute_bins(df['forecast_solar_mean'].to_numpy(), N_SOLAR_MEAN_BINS),
        'humidity_mean_edges': compute_bins(df['forecast_humidity_mean'].to_numpy(), N_HUMIDITY_MEAN_BINS),
    }


def state_to_index(state_info, edges):
    o = bin_value(state_info['occupancy'], edges['occupancy_edges'])
    tm = bin_value(state_info['forecast_temp_mean'], edges['temp_mean_edges'])
    tt = bin_value(state_info['forecast_temp_trend'], edges['temp_trend_edges'])
    sm = bin_value(state_info['forecast_solar_mean'], edges['solar_mean_edges'])
    hm = bin_value(state_info['forecast_humidity_mean'], edges['humidity_mean_edges'])

    base4 = N_HUMIDITY_MEAN_BINS
    base3 = N_SOLAR_MEAN_BINS * base4
    base2 = N_TEMP_TREND_BINS * base3
    base1 = N_TEMP_MEAN_BINS * base2

    return o * base1 + tm * base2 + tt * base3 + sm * base4 + hm


# ============================================================================
# ENVIRONMENT
# ============================================================================

class BuildingEnv:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.t = 0

    def reset(self, random_start=True):
        if random_start:
            max_start = max(0, self.n_steps - EPISODE_LENGTH - 1)
            self.t = int(np.random.randint(0, max_start + 1))
        else:
            self.t = 0
        return self.get_state_info()

    def get_state_info(self):
        row = self.df.iloc[self.t]
        # Only occupancy + weather forecasts
        return {
            'occupancy': float(row['occupant_count']),
            'forecast_temp_mean': float(row['forecast_temp_mean']),
            'forecast_temp_trend': float(row['forecast_temp_trend']),
            'forecast_solar_mean': float(row['forecast_solar_mean']),
            'forecast_humidity_mean': float(row['forecast_humidity_mean']),
        }

    def step(self, action: int):
        row = self.df.iloc[self.t]
        intensity = INTENSITY[action]

        cooling = float(row['cooling_demand'])
        heating = float(row['heating_demand'])
        base_load = float(row['non_shiftable_load']) + float(row['dhw_demand'])

        # Internal plant model: HVAC spends energy proportional to the active demand.
        hvac_demand = max(cooling, heating)
        hvac_energy = intensity * hvac_demand

        total_energy = base_load + hvac_energy

        # Reward = negative energy + mild comfort penalty for turning HVAC off under high demand.
        comfort_penalty = 0.0
        if action == 0 and hvac_demand > 0.5:
            comfort_penalty = -0.2 * hvac_demand

        reward = -total_energy + comfort_penalty

        self.t += 1
        done = (self.t >= self.n_steps - 1)

        next_state = self.get_state_info() if not done else None
        return next_state, reward, done, total_energy


# ============================================================================
# AGENT
# ============================================================================

class QLearningAgent:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
        self.epsilon = EPSILON_START

    def act(self, state_idx: int, training=True):
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        return int(np.argmax(self.q[state_idx]))

    def update(self, s, a, r, s_next, done):
        q_sa = self.q[s][a]
        max_next = 0.0 if done else float(np.max(self.q[s_next]))
        target = r + GAMMA * max_next
        self.q[s][a] = q_sa + ALPHA * (target - q_sa)

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


# ============================================================================
# TRAIN / EVAL
# ============================================================================

def train(env: BuildingEnv, agent: QLearningAgent, edges):
    rewards_hist = []
    energies_hist = []

    for _ in tqdm(range(TRAIN_EPISODES), desc='Training Q-learning'):
        state = env.reset(random_start=True)
        ep_reward = 0.0
        ep_energy = 0.0

        for _step in range(EPISODE_LENGTH):
            s = state_to_index(state, edges)
            a = agent.act(s, training=True)
            next_state, reward, done, step_energy = env.step(a)
            s_next = s if next_state is None else state_to_index(next_state, edges)

            agent.update(s, a, reward, s_next, done)

            ep_reward += reward
            ep_energy += step_energy

            if done:
                break
            state = next_state

        agent.decay_epsilon()
        rewards_hist.append(ep_reward)
        energies_hist.append(ep_energy)

    return rewards_hist, energies_hist


def evaluate(env: BuildingEnv, policy_fn, edges, n_episodes=20):
    ep_energies = []
    ep_rewards = []

    for _ in range(n_episodes):
        state = env.reset(random_start=False)
        ep_energy = 0.0
        ep_reward = 0.0

        for _step in range(EPISODE_LENGTH):
            s = state_to_index(state, edges)
            a = policy_fn(s)
            next_state, reward, done, step_energy = env.step(a)

            ep_energy += step_energy
            ep_reward += reward

            if done:
                break
            state = next_state

        ep_energies.append(ep_energy)
        ep_rewards.append(ep_reward)

    return {
        'avg_energy': float(np.mean(ep_energies)),
        'avg_reward': float(np.mean(ep_rewards)),
        'episode_energies': ep_energies,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print('=' * 72)
    print('Q_learning0704_03 - state uses ONLY weather forecast + occupancy')
    print('=' * 72)

    set_seed(RANDOM_SEED)

    print('\n[1] Loading data...')
    df = load_data(DATASET_DIR)
    print(f'    Time steps: {len(df)}')

    print('\n[2] Building state discretization edges...')
    edges = build_state_edges(df)
    n_states = (
        N_OCCUPANCY_BINS
        * N_TEMP_MEAN_BINS
        * N_TEMP_TREND_BINS
        * N_SOLAR_MEAN_BINS
        * N_HUMIDITY_MEAN_BINS
    )
    print(f'    Approx state count: {n_states}')
    print(f'    Actions: {list(ACTIONS.values())}')

    print('\n[3] Training Q-learning agent...')
    env = BuildingEnv(df)
    agent = QLearningAgent(n_actions=len(ACTIONS))
    train_rewards, train_energies = train(env, agent, edges)
    print(f'    Last-50 mean training energy: {np.mean(train_energies[-50:]):.4f}')

    print('\n[4] Evaluating Q-learning policy...')
    q_policy = lambda s: agent.act(s, training=False)
    q_res = evaluate(env, q_policy, edges, n_episodes=20)
    print(f'    Q-learning avg energy: {q_res["avg_energy"]:.4f}')

    print('\n[5] Evaluating fixed baselines...')
    baseline_results = {}
    for a_id, a_name in ACTIONS.items():
        fixed = lambda _s, aa=a_id: aa
        res = evaluate(env, fixed, edges, n_episodes=20)
        baseline_results[a_name] = res
        print(f'    {a_name:10s} avg energy: {res["avg_energy"]:.4f}')

    print('\n[6] Energy savings vs best fixed baseline')
    best_name, best_res = min(baseline_results.items(), key=lambda kv: kv[1]['avg_energy'])
    savings = best_res['avg_energy'] - q_res['avg_energy']
    savings_pct = 100.0 * savings / best_res['avg_energy'] if best_res['avg_energy'] > 0 else 0.0

    print('-' * 72)
    print(f'Q-learning energy:      {q_res["avg_energy"]:.4f}')
    print(f'Best fixed ({best_name}): {best_res["avg_energy"]:.4f}')
    print(f'Absolute savings:       {savings:.4f}')
    print(f'Relative savings:       {savings_pct:.2f}%')
    print('-' * 72)

    print('\n[7] Saving plot...')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(train_energies, alpha=0.5, label='episode energy')
    axes[0, 0].plot(moving_average(train_energies, window=50), linewidth=2, label='ma(50)')
    axes[0, 0].set_title('Training energy')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    methods = list(baseline_results.keys()) + ['q_learning']
    energies = [baseline_results[m]['avg_energy'] for m in baseline_results.keys()] + [q_res['avg_energy']]
    colors = ['#c0c0c0', '#9ecae1', '#6baed6', '#2ca25f']
    axes[0, 1].bar(methods, energies, color=colors)
    axes[0, 1].set_title('Average energy comparison')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].hist(q_res['episode_energies'], bins=10, color='#2ca25f', alpha=0.8, edgecolor='black')
    axes[1, 0].set_title('Q-learning episode energy distribution')
    axes[1, 0].set_xlabel('Energy')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3)

    savings_vs_each = []
    labels = []
    for name, res in baseline_results.items():
        labels.append(name)
        val = 100.0 * (res['avg_energy'] - q_res['avg_energy']) / res['avg_energy'] if res['avg_energy'] > 0 else 0.0
        savings_vs_each.append(val)

    bar_colors = ['#2ca25f' if v >= 0 else '#de2d26' for v in savings_vs_each]
    axes[1, 1].barh(labels, savings_vs_each, color=bar_colors)
    axes[1, 1].axvline(0.0, color='black', linewidth=1)
    axes[1, 1].set_title('Q-learning savings vs fixed strategies [%]')
    axes[1, 1].set_xlabel('Savings [%]')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('energy_optimization_q_learning0704_03.png', dpi=120, bbox_inches='tight')
    print('    Saved: energy_optimization_q_learning0704_03.png')


if __name__ == '__main__':
    main()
