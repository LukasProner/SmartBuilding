import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Simple tuning knobs (edit here)
# -----------------------------
RANDOM_SEED = 42

# Dataset
DATASET_DIR = (
    'CityLearn/data/datasets/'
    'citylearn_challenge_2023_phase_3_3'
)

# Training knobs
TRAIN_EPISODES = 700
TRAIN_EPISODE_LENGTH = 24 * 28
EPSILON_START = 0.5
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.999
AUTO_EPSILON_DECAY = True
ALPHA = 0.07
GAMMA = 0.995

# Policy comparison knob
FIXED_BASELINE_ACTION = 1  # 0=eco, 1=normal, 2=comfort

# Environment/reward knobs
COOLING_MULTIPLIERS = np.array([0.3, 1.0, 1.2])
HOT_TEMP_THRESHOLD = 25.0
COMFORT_PENALTY_WEIGHT = 4
OVERCOOL_PENALTY_WEIGHT = 2.5
OVERCOOL_TOLERANCE = 0.2
# Relative comfort sensitivity per action: eco is penalized the most,
# comfort mode the least, but all modes are penalized if under-cooled.
COMFORT_ACTION_WEIGHTS = np.array([1.0, 0.4, 0.15])


# -----------------------------
# 1) Data loading
# -----------------------------
def load_citylearn_data(dataset_dir):
    building_path = os.path.join(dataset_dir, 'Building_1.csv')
    weather_path = os.path.join(dataset_dir, 'weather.csv')

    building = pd.read_csv(building_path)
    weather = pd.read_csv(weather_path)

    data = pd.DataFrame({
        # Calendar / control context
        'month': building['month'].to_numpy(),
        'hour': building['hour'].to_numpy(),
        'day_type': building['day_type'].to_numpy(),
        'hvac_mode': building['hvac_mode'].to_numpy(),

        # Occupancy signal (directly available in this dataset)
        'occupant_count': building['occupant_count'].to_numpy(),

        # Building states for comfort
        'indoor_temp': building['indoor_dry_bulb_temperature'].to_numpy(),
        'cool_setpoint': building['indoor_dry_bulb_temperature_cooling_set_point'].to_numpy(),
        'avg_unmet_cooling': building['average_unmet_cooling_setpoint_difference'].to_numpy(),

        # Energy terms
        'non_shiftable_load': building['non_shiftable_load'].to_numpy(),
        'dhw_demand': building['dhw_demand'].to_numpy(),
        'cooling_demand': building['cooling_demand'].to_numpy(),
        'solar_generation': building['solar_generation'].to_numpy(),

        # Weather forecasts
        'outdoor_temp_pred_1': weather['outdoor_dry_bulb_temperature_predicted_1'].to_numpy(),
        'outdoor_humidity_pred_1': weather['outdoor_relative_humidity_predicted_1'].to_numpy(),
        'diffuse_solar_pred_1': weather['diffuse_solar_irradiance_predicted_1'].to_numpy(),
        'direct_solar_pred_1': weather['direct_solar_irradiance_predicted_1'].to_numpy(),
    })

    data['solar_pred_1'] = data['diffuse_solar_pred_1'] + data['direct_solar_pred_1']

    return data


# -----------------------------
# 2) Simple environment
# -----------------------------
@dataclass
class EnvConfig:
    month_bins: int = 4
    hour_bins: int = 6
    day_type_bins: int = 2
    temp_bins: int = 6
    humidity_bins: int = 4
    solar_bins: int = 4
    occupant_bins: int = 4
    hvac_mode_bins: int = 4


class BuildingQEnv:
    """
    A richer but still tabular environment:
    - State uses multiple available predictors (calendar + occupancy + weather + mode).
    - Action: 0=eco, 1=normal, 2=comfort
    - Reward: -net_grid - comfort_penalty
    """

    def __init__(
        self,
        data,
        config=None,
        cooling_multipliers=None,
        hot_temp_threshold=HOT_TEMP_THRESHOLD,
        comfort_penalty_weight=COMFORT_PENALTY_WEIGHT,
        overcool_penalty_weight=OVERCOOL_PENALTY_WEIGHT,
        overcool_tolerance=OVERCOOL_TOLERANCE,
        comfort_action_weights=None,
    ):
        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)
        self.config = config or EnvConfig()

        self.action_names = ['eco', 'normal', 'comfort']
        self.cooling_multipliers = np.array(
            cooling_multipliers if cooling_multipliers is not None else COOLING_MULTIPLIERS,
            dtype=np.float64,
        )
        self.hot_temp_threshold = float(hot_temp_threshold)
        self.comfort_penalty_weight = float(comfort_penalty_weight)
        self.overcool_penalty_weight = float(overcool_penalty_weight)
        self.overcool_tolerance = float(overcool_tolerance)
        self.comfort_action_weights = np.array(
            comfort_action_weights if comfort_action_weights is not None else COMFORT_ACTION_WEIGHTS,
            dtype=np.float64,
        )

        self.month_edges = self._make_edges(self.data['month'].to_numpy(), self.config.month_bins)
        self.hour_edges = np.linspace(1, 24, self.config.hour_bins + 1)
        self.temp_edges = self._make_edges(self.data['outdoor_temp_pred_1'].to_numpy(), self.config.temp_bins)
        self.humidity_edges = self._make_edges(self.data['outdoor_humidity_pred_1'].to_numpy(), self.config.humidity_bins)
        self.solar_edges = self._make_edges(self.data['solar_pred_1'].to_numpy(), self.config.solar_bins)
        self.occupant_edges = self._make_edges(self.data['occupant_count'].to_numpy(), self.config.occupant_bins)

        self.state_shape = (
            self.config.month_bins,
            self.config.hour_bins,
            self.config.day_type_bins,
            self.config.temp_bins,
            self.config.humidity_bins,
            self.config.solar_bins,
            self.config.occupant_bins,
            self.config.hvac_mode_bins,
        )
        self.total_states = int(np.prod(self.state_shape))
        self.total_actions = len(self.action_names)

        self.t = 0

    def _make_edges(self, values, n_bins):
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(values, q)
        edges = np.unique(edges)

        if len(edges) < 3:
            v_min, v_max = float(np.min(values)), float(np.max(values))
            if v_min == v_max:
                v_max = v_min + 1e-6
            edges = np.linspace(v_min, v_max, n_bins + 1)

        return edges

    def reset(self, start_index=0):
        self.t = int(start_index)
        return self._get_state(self.t)

    def _digitize(self, value, edges, n_bins):
        idx = int(np.digitize(value, edges[1:-1], right=False))
        return max(0, min(idx, n_bins - 1))

    def _day_type_bin(self, day_type):
        # Simple split: weekday-like vs weekend/holiday-like
        return 0 if int(day_type) <= 5 else 1

    def _hvac_mode_bin(self, hvac_mode):
        return max(0, min(int(hvac_mode), self.config.hvac_mode_bins - 1))

    def _get_state(self, idx):
        row = self.data.iloc[idx]

        month_bin = self._digitize(row['month'], self.month_edges, self.config.month_bins)
        hour_bin = self._digitize(row['hour'], self.hour_edges, self.config.hour_bins)
        day_type_bin = self._day_type_bin(row['day_type'])
        temp_bin = self._digitize(row['outdoor_temp_pred_1'], self.temp_edges, self.config.temp_bins)
        humidity_bin = self._digitize(row['outdoor_humidity_pred_1'], self.humidity_edges, self.config.humidity_bins)
        solar_bin = self._digitize(row['solar_pred_1'], self.solar_edges, self.config.solar_bins)
        occupant_bin = self._digitize(row['occupant_count'], self.occupant_edges, self.config.occupant_bins)
        hvac_mode_bin = self._hvac_mode_bin(row['hvac_mode'])

        state_tuple = (
            month_bin,
            hour_bin,
            day_type_bin,
            temp_bin,
            humidity_bin,
            solar_bin,
            occupant_bin,
            hvac_mode_bin,
        )
        state = int(np.ravel_multi_index(state_tuple, self.state_shape))

        return state

    def step(self, action):
        row = self.data.iloc[self.t]

        base_load = row['non_shiftable_load'] + row['dhw_demand']
        hvac_load = row['cooling_demand'] * self.cooling_multipliers[action]

        # Net grid energy approximation (kWh for 1-hour step)
        net_grid = max(0.0, base_load + hvac_load - row['solar_generation'])

        # Comfort penalty keeps original logic but uses richer available signals.
        occupied_factor = float(np.clip(row['occupant_count'] / 4.0, 0.0, 1.0))
        too_hot_outdoor = 1.0 if row['outdoor_temp_pred_1'] >= self.hot_temp_threshold else 0.0
        too_hot_indoor = max(0.0, row['indoor_temp'] - row['cool_setpoint'])
        unmet = max(0.0, row['avg_unmet_cooling'])
        comfort_signal = too_hot_indoor + unmet + 0.3 * too_hot_outdoor
        comfort_penalty = occupied_factor * comfort_signal * self.comfort_action_weights[action]

        # Penalize unnecessary cooling when indoor temperature is already below setpoint.
        overcool_signal = max(
            0.0,
            (row['cool_setpoint'] - self.overcool_tolerance) - row['indoor_temp'],
        )
        unnecessary_cooling_penalty = occupied_factor * overcool_signal * self.cooling_multipliers[action]

        reward = -(
            net_grid
            + self.comfort_penalty_weight * comfort_penalty
            + self.overcool_penalty_weight * unnecessary_cooling_penalty
        )

        self.t += 1
        done = self.t >= self.n_steps
        next_state = None if done else self._get_state(self.t)

        info = {
            'net_grid': net_grid,
            'comfort_penalty': comfort_penalty,
            'unnecessary_cooling_penalty': unnecessary_cooling_penalty,
        }
        return next_state, reward, done, info


# -----------------------------
# 3) Q-learning agent
# -----------------------------
class QAgent:
    def __init__(self, n_states, n_actions, epsilon=0.2, alpha=0.1, gamma=0.98):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.action_counts = np.zeros(n_actions, dtype=np.int64)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = int(np.argmax(self.q[state]))

        self.action_counts[action] += 1
        return action

    def update(self, state, action, reward, next_state, done):
        next_q = 0.0 if done else np.max(self.q[next_state])
        target = reward + self.gamma * next_q
        self.q[state, action] += self.alpha * (target - self.q[state, action])


# -----------------------------
# 4) Train + evaluate
# -----------------------------
def train_q_learning(env, episodes=TRAIN_EPISODES, episode_length=TRAIN_EPISODE_LENGTH, eps_decay=EPSILON_DECAY):
    if AUTO_EPSILON_DECAY and episodes > 0 and EPSILON_START > EPSILON_MIN:
        # Choose decay so epsilon approaches EPSILON_MIN near the end of training.
        eps_decay = float((EPSILON_MIN / EPSILON_START) ** (1.0 / episodes))

    agent = QAgent(
        n_states=env.total_states,
        n_actions=env.total_actions,
        epsilon=EPSILON_START,
        alpha=ALPHA,
        gamma=GAMMA,
    )

    rewards_history = []
    energy_history = []
    comfort_history = []
    unnecessary_history = []
    epsilon_history = []

    max_start = max(1, env.n_steps - episode_length - 1)

    for _ in tqdm(range(episodes), desc='Training episodes'):
        start = np.random.randint(0, max_start)
        state = env.reset(start)

        total_reward = 0.0
        total_energy = 0.0
        total_comfort_penalty = 0.0
        total_unnecessary_cooling_penalty = 0.0

        for _step in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            total_energy += info['net_grid']
            total_comfort_penalty += info['comfort_penalty']
            total_unnecessary_cooling_penalty += info['unnecessary_cooling_penalty']

            if done:
                break

        rewards_history.append(total_reward)
        energy_history.append(total_energy)
        comfort_history.append(total_comfort_penalty)
        unnecessary_history.append(total_unnecessary_cooling_penalty)
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * eps_decay)
        epsilon_history.append(agent.epsilon)
    print(f'epsilon decay used: {eps_decay:.6f}')
    print(f'epsilon final: {agent.epsilon:.6f}')
    return (
        agent,
        np.array(rewards_history),
        np.array(energy_history),
        np.array(comfort_history),
        np.array(unnecessary_history),
        np.array(epsilon_history),
    )


def evaluate_policy(env, q_table, start=0, horizon=24 * 7):
    state = env.reset(start)

    total_reward = 0.0
    total_energy = 0.0
    comfort_penalty_sum = 0.0
    unnecessary_cooling_penalty_sum = 0.0

    for _ in range(horizon):
        action = int(np.argmax(q_table[state]))
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_energy += info['net_grid']
        comfort_penalty_sum += info['comfort_penalty']
        unnecessary_cooling_penalty_sum += info['unnecessary_cooling_penalty']

        if done:
            break
        state = next_state

    return {
        'reward': total_reward,
        'energy': total_energy,
        'comfort_penalty': comfort_penalty_sum,
        'unnecessary_cooling_penalty': unnecessary_cooling_penalty_sum,
    }


def evaluate_fixed_policy(env, fixed_action=1, start=0, horizon=24 * 7):
    state = env.reset(start)

    total_reward = 0.0
    total_energy = 0.0
    comfort_penalty_sum = 0.0
    unnecessary_cooling_penalty_sum = 0.0

    for _ in range(horizon):
        next_state, reward, done, info = env.step(int(fixed_action))

        total_reward += reward
        total_energy += info['net_grid']
        comfort_penalty_sum += info['comfort_penalty']
        unnecessary_cooling_penalty_sum += info['unnecessary_cooling_penalty']

        if done:
            break
        state = next_state

    return {
        'reward': total_reward,
        'energy': total_energy,
        'comfort_penalty': comfort_penalty_sum,
        'unnecessary_cooling_penalty': unnecessary_cooling_penalty_sum,
    }


def _moving_average(values, window):
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode='same')


def plot_training_curves(rewards, energies, comforts, unnecessary, epsilons):
    if not os.path.exists('./images'):
        os.makedirs('./images')

    n_episodes = len(rewards)
    smooth_window = max(5, n_episodes // 25)

    plt.figure(figsize=(11, 14))

    plt.subplot(4, 1, 1)
    plt.plot(rewards, color='tab:blue', alpha=0.28, label='Raw')
    plt.plot(_moving_average(rewards, smooth_window), color='tab:blue', linewidth=2.0, label=f'MA({smooth_window})')
    plt.title('Q-learning0604 training reward')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.legend(loc='best')

    plt.subplot(4, 1, 2)
    plt.plot(energies, color='tab:orange', alpha=0.28, label='Raw')
    plt.plot(_moving_average(energies, smooth_window), color='tab:orange', linewidth=2.0, label=f'MA({smooth_window})')
    plt.title('Episode net grid energy')
    plt.xlabel('Episode')
    plt.ylabel('Net grid energy')
    plt.legend(loc='best')

    plt.subplot(4, 1, 3)
    plt.plot(comforts, color='tab:red', alpha=0.28, label='Raw')
    plt.plot(_moving_average(comforts, smooth_window), color='tab:red', linewidth=2.0, label=f'MA({smooth_window})')
    plt.title('Episode comfort penalty')
    plt.xlabel('Episode')
    plt.ylabel('Comfort penalty')
    plt.legend(loc='best')

    plt.subplot(5, 1, 4)
    plt.plot(unnecessary, color='tab:purple', alpha=0.28, label='Raw')
    plt.plot(_moving_average(unnecessary, smooth_window), color='tab:purple', linewidth=2.0, label=f'MA({smooth_window})')
    plt.title('Episode unnecessary cooling penalty')
    plt.xlabel('Episode')
    plt.ylabel('Overcool penalty')
    plt.legend(loc='best')

    plt.subplot(5, 1, 5)
    plt.plot(epsilons, color='tab:green')
    plt.title('Epsilon decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.ylim(0.0, max(0.4, float(np.max(epsilons)) + 0.02))

    plt.tight_layout()
    plt.savefig('./images/q_learning0604_training.png', dpi=140)
    plt.close()


def _rollout_policy_series(env, q_table=None, fixed_action=None, start=0, horizon=24 * 7):
    state = env.reset(start)

    energies = []
    comfort_penalties = []
    actions = []

    for _ in range(horizon):
        if q_table is not None:
            action = int(np.argmax(q_table[state]))
        else:
            action = int(fixed_action)

        next_state, _reward, done, info = env.step(action)
        energies.append(info['net_grid'])
        comfort_penalties.append(info['comfort_penalty'])
        actions.append(action)

        if done:
            break
        state = next_state

    return {
        'energies': np.array(energies),
        'comfort_penalties': np.array(comfort_penalties),
        'actions': np.array(actions),
    }


def plot_policy_comparison(env, q_table, fixed_action=1, start=0, horizon=24 * 7):
    if not os.path.exists('./images'):
        os.makedirs('./images')

    learned = _rollout_policy_series(env, q_table=q_table, start=start, horizon=horizon)
    baseline = _rollout_policy_series(env, fixed_action=fixed_action, start=start, horizon=horizon)

    steps = np.arange(len(learned['energies']))
    learned_cum_energy = np.cumsum(learned['energies'])
    baseline_cum_energy = np.cumsum(baseline['energies'])

    plt.figure(figsize=(11, 12))

    plt.subplot(4, 1, 1)
    plt.plot(steps, learned['energies'], color='tab:blue', alpha=0.8, label='Learned policy')
    plt.plot(steps, baseline['energies'], color='tab:gray', alpha=0.7, label=f'Fixed action {fixed_action}')
    plt.title('Hourly net grid energy (evaluation horizon)')
    plt.xlabel('Hour')
    plt.ylabel('Net grid energy')
    plt.legend(loc='best')

    plt.subplot(4, 1, 2)
    plt.plot(steps, learned_cum_energy, color='tab:blue', linewidth=2.0, label='Learned cumulative')
    plt.plot(steps, baseline_cum_energy, color='tab:gray', linewidth=2.0, label='Baseline cumulative')
    plt.title('Cumulative net grid energy')
    plt.xlabel('Hour')
    plt.ylabel('Cumulative energy')
    plt.legend(loc='best')

    plt.subplot(4, 1, 3)
    plt.plot(steps, learned['comfort_penalties'], color='tab:red', label='Learned comfort penalty')
    plt.plot(steps, baseline['comfort_penalties'], color='tab:orange', label='Baseline comfort penalty')
    plt.title('Hourly comfort penalty')
    plt.xlabel('Hour')
    plt.ylabel('Comfort penalty')
    plt.legend(loc='best')

    learned_action_names = ['eco', 'normal', 'comfort']
    learned_counts = np.bincount(learned['actions'], minlength=3)
    baseline_counts = np.bincount(baseline['actions'], minlength=3)

    x = np.arange(3)
    width = 0.38
    plt.subplot(4, 1, 4)
    plt.bar(x - width / 2, learned_counts, width=width, color='tab:blue', alpha=0.8, label='Learned policy')
    plt.bar(x + width / 2, baseline_counts, width=width, color='tab:gray', alpha=0.8, label='Baseline')
    plt.xticks(x, learned_action_names)
    plt.title('Action usage count (evaluation horizon)')
    plt.xlabel('Action')
    plt.ylabel('Count')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig('./images/q_learning0604_policy_comparison.png', dpi=140)
    plt.close()


def main():
    np.random.seed(RANDOM_SEED)

    data = load_citylearn_data(DATASET_DIR)
    env = BuildingQEnv(data)

    agent, rewards, energies, comforts, unnecessary, epsilons = train_q_learning(env)

    metrics = evaluate_policy(env, agent.q, start=0)
    baseline_metrics = evaluate_fixed_policy(env, fixed_action=FIXED_BASELINE_ACTION, start=0)

    energy_saving_abs = baseline_metrics['energy'] - metrics['energy']
    energy_saving_pct = (
        100.0 * energy_saving_abs / baseline_metrics['energy']
        if baseline_metrics['energy'] > 0
        else 0.0
    )

    plot_training_curves(rewards, energies, comforts, unnecessary, epsilons)
    plot_policy_comparison(
        env,
        agent.q,
        fixed_action=FIXED_BASELINE_ACTION,
        start=0,
        horizon=24 * 7,
    )

    print('Training done.')
    print(f'Q-table shape: {agent.q.shape}')
    print(f'Action counts (training): eco={agent.action_counts[0]}, normal={agent.action_counts[1]}, comfort={agent.action_counts[2]}')
    print(f'Eval reward (1 week): {metrics["reward"]:.2f}')
    print(f'Eval net grid energy (1 week): {metrics["energy"]:.2f}')
    print(f'Eval comfort penalty (1 week): {metrics["comfort_penalty"]:.2f}')
    print(f'Eval unnecessary cooling penalty (1 week): {metrics["unnecessary_cooling_penalty"]:.2f}')
    print(f'--- Baseline: fixed strategy (action={FIXED_BASELINE_ACTION}) ---')
    print(f'Baseline reward (1 week): {baseline_metrics["reward"]:.2f}')
    print(f'Baseline net grid energy (1 week): {baseline_metrics["energy"]:.2f}')
    print(f'Baseline comfort penalty (1 week): {baseline_metrics["comfort_penalty"]:.2f}')
    print(f'Baseline unnecessary cooling penalty (1 week): {baseline_metrics["unnecessary_cooling_penalty"]:.2f}')
    print('--- Savings vs baseline ---')
    print(f'Absolute energy saving (1 week): {energy_saving_abs:.2f}')
    print(f'Percent energy saving (1 week): {energy_saving_pct:.2f}%')


if __name__ == '__main__':
    main()
