import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# 1) Data loading
# -----------------------------
def load_citylearn_data(dataset_dir):
    building_path = os.path.join(dataset_dir, 'Building_1.csv')
    weather_path = os.path.join(dataset_dir, 'weather.csv')

    building = pd.read_csv(building_path)
    weather = pd.read_csv(weather_path)

    # Use only columns needed for a simple and understandable baseline.
    data = pd.DataFrame({
        'hour': building['hour'].to_numpy(),
        'non_shiftable_load': building['non_shiftable_load'].to_numpy(),
        'cooling_demand': building['cooling_demand'].to_numpy(),
        'dhw_demand': building['dhw_demand'].to_numpy(),
        'solar_generation': building['solar_generation'].to_numpy(),
        'outdoor_temp_pred_1': weather['outdoor_dry_bulb_temperature_predicted_1'].to_numpy(),
    })

    # CityLearn 2020 zone 1 has no direct occupancy signal.
    # We create a simple occupancy proxy from non-shiftable load.
    occ_threshold = np.quantile(data['non_shiftable_load'], 0.65)
    data['occupancy_proxy'] = (data['non_shiftable_load'] >= occ_threshold).astype(int)

    return data


# -----------------------------
# 2) Simple environment
# -----------------------------
@dataclass
class EnvConfig:
    temp_bins: int = 6
    hour_bins: int = 6
    occ_bins: int = 2


class BuildingQEnv:
    """
    A tiny tabular environment:
    - State: (hour_bin, temp_bin, occupancy_proxy)
    - Action: 0=eco, 1=normal, 2=comfort
    - Reward: -energy_cost - comfort_penalty
    """

    def __init__(self, data, config=None):
        self.data = data.reset_index(drop=True)
        self.n_steps = len(self.data)
        self.config = config or EnvConfig()

        self.action_names = ['eco', 'normal', 'comfort']
        self.cooling_multipliers = np.array([0.5, 1.0, 1.50])

        self.temp_edges = np.linspace(
            self.data['outdoor_temp_pred_1'].min(),
            self.data['outdoor_temp_pred_1'].max(),
            self.config.temp_bins + 1,
        )

        self.hour_edges = np.linspace(1, 24, self.config.hour_bins + 1)

        self.total_states = self.config.hour_bins * self.config.temp_bins * self.config.occ_bins
        self.total_actions = len(self.action_names)

        self.t = 0

    def reset(self, start_index=0):
        self.t = int(start_index)
        return self._get_state(self.t)

    def _digitize(self, value, edges):
        idx = int(np.digitize(value, edges[1:-1], right=False))
        return max(0, min(idx, len(edges) - 2))

    def _get_state(self, idx):
        row = self.data.iloc[idx]
        hour_bin = self._digitize(row['hour'], self.hour_edges)
        temp_bin = self._digitize(row['outdoor_temp_pred_1'], self.temp_edges)
        occ_bin = int(row['occupancy_proxy'])

        # Flatten tuple -> integer state index
        state = (
            hour_bin * (self.config.temp_bins * self.config.occ_bins)
            + temp_bin * self.config.occ_bins
            + occ_bin
        )
        return state

    def step(self, action):
        row = self.data.iloc[self.t]

        base_load = row['non_shiftable_load'] + row['dhw_demand']
        hvac_load = row['cooling_demand'] * self.cooling_multipliers[action]

        # Net grid energy approximation (kWh for 1-hour step)
        net_grid = max(0.0, base_load + hvac_load - row['solar_generation'])

        # Comfort penalty only when likely occupied and hot.
        too_hot = 1.0 if row['outdoor_temp_pred_1'] >= 28.0 else 0.0
        occupied = float(row['occupancy_proxy'])
        comfort_penalty = occupied * too_hot * (1.0 if action == 0 else 0.0)

        reward = -(net_grid + 2.0 * comfort_penalty)

        self.t += 1
        done = self.t >= self.n_steps
        next_state = None if done else self._get_state(self.t)

        info = {
            'net_grid': net_grid,
            'comfort_penalty': comfort_penalty,
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
def train_q_learning(env, episodes=120, episode_length=24 * 14, eps_decay=0.985):
    agent = QAgent(
        n_states=env.total_states,
        n_actions=env.total_actions,
        epsilon=0.2,
        alpha=0.12,
        gamma=0.98,
    )

    rewards_history = []
    energy_history = []

    max_start = max(1, env.n_steps - episode_length - 1)

    for _ in tqdm(range(episodes), desc='Training episodes'):
        start = np.random.randint(0, max_start)
        state = env.reset(start)

        total_reward = 0.0
        total_energy = 0.0

        for _step in range(episode_length):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.update(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            total_energy += info['net_grid']

            if done:
                break

        rewards_history.append(total_reward)
        energy_history.append(total_energy)
        # print(f'reward: {total_reward}')

        # Gradual exploration decrease like in your bandit example
        agent.epsilon = max(0.03, agent.epsilon * eps_decay)

    return agent, np.array(rewards_history), np.array(energy_history)


def evaluate_policy(env, q_table, start=0, horizon=24 * 7):
    state = env.reset(start)

    total_reward = 0.0
    total_energy = 0.0
    comfort_penalty_sum = 0.0

    for _ in range(horizon):
        action = int(np.argmax(q_table[state]))
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_energy += info['net_grid']
        comfort_penalty_sum += info['comfort_penalty']

        if done:
            break
        state = next_state

    return {
        'reward': total_reward,
        'energy': total_energy,
        'comfort_penalty': comfort_penalty_sum,
    }


def evaluate_fixed_policy(env, fixed_action=1, start=0, horizon=24 * 7):
    state = env.reset(start)

    total_reward = 0.0
    total_energy = 0.0
    comfort_penalty_sum = 0.0

    for _ in range(horizon):
        action = int(fixed_action)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_energy += info['net_grid']
        comfort_penalty_sum += info['comfort_penalty']

        if done:
            break
        state = next_state

    return {
        'reward': total_reward,
        'energy': total_energy,
        'comfort_penalty': comfort_penalty_sum,
    }


def plot_training_curves(rewards, energies):
    if not os.path.exists('./images'):
        os.makedirs('./images')

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(rewards, color='tab:blue')
    plt.title('Q-learning training reward')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')

    plt.subplot(2, 1, 2)
    plt.plot(energies, color='tab:orange')
    plt.title('Q-learning episode energy')
    plt.xlabel('Episode')
    plt.ylabel('Net grid energy')

    plt.tight_layout()
    plt.savefig('./images/q_learning_training.png', dpi=140)
    plt.close()


def main():
    dataset_dir = (
        'CityLearn/data/datasets/'
        'citylearn_challenge_2020_climate_zone_1'
    )

    data = load_citylearn_data(dataset_dir)
    env = BuildingQEnv(data)

    agent, rewards, energies = train_q_learning(env)
    metrics = evaluate_policy(env, agent.q, start=0)
    baseline_metrics = evaluate_fixed_policy(env, fixed_action=1, start=0)

    energy_saving_abs = baseline_metrics['energy'] - metrics['energy']
    energy_saving_pct = 100.0 * energy_saving_abs / baseline_metrics['energy'] if baseline_metrics['energy'] > 0 else 0.0

    plot_training_curves(rewards, energies)

    print('Training done.')
    print(f'Q-table shape: {agent.q.shape}')
    print(f'Eval reward (1 week): {metrics["reward"]:.2f}')
    print(f'Eval net grid energy (1 week): {metrics["energy"]:.2f}')
    print(f'Eval comfort penalty (1 week): {metrics["comfort_penalty"]:.2f}')
    print('--- Baseline: fixed strategy (normal action) ---')
    print(f'Baseline reward (1 week): {baseline_metrics["reward"]:.2f}')
    print(f'Baseline net grid energy (1 week): {baseline_metrics["energy"]:.2f}')
    print(f'Baseline comfort penalty (1 week): {baseline_metrics["comfort_penalty"]:.2f}')
    print('--- Savings vs baseline ---')
    print(f'Absolute energy saving (1 week): {energy_saving_abs:.2f}')
    print(f'Percent energy saving (1 week): {energy_saving_pct:.2f}%')


if __name__ == '__main__':
    main()
