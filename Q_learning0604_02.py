import os
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------
# Tuning knobs
# -----------------------------
RANDOM_SEED = 42

DATASET_DIR = (
	'CityLearn/data/datasets/'
	'citylearn_challenge_2023_phase_3_3'
)

TRAIN_EPISODES = 1200
TRAIN_EPISODE_LENGTH = 24 * 28
EPSILON_START = 0.6
EPSILON_MIN = 0.02
AUTO_EPSILON_DECAY = True
EPSILON_DECAY = 0.997
ALPHA = 0.08
GAMMA = 0.0

FIXED_BASELINE_ACTION = 1  # 0=off, 1=cool_low, 2=cool_high

ENERGY_WEIGHT = 1.0

COOL_MULTIPLIERS = np.array([0.0, 0.5, 1.0], dtype=np.float64)


# -----------------------------
# Helpers
# -----------------------------
def _get_column(frame, column_name, default_value=0.0):
	if column_name in frame.columns:
		return frame[column_name].to_numpy()
	return np.full(len(frame), default_value, dtype=np.float64)


def _safe_edges(values, n_bins):
	q = np.linspace(0.0, 1.0, n_bins + 1)
	edges = np.unique(np.quantile(values, q))
	if len(edges) < 3:
		v_min = float(np.min(values))
		v_max = float(np.max(values))
		if v_min == v_max:
			v_max = v_min + 1e-6
		edges = np.linspace(v_min, v_max, n_bins + 1)
	return edges


def _bin_value(value, edges, n_bins):
	index = int(np.digitize(value, edges[1:-1], right=False))
	return max(0, min(index, n_bins - 1))


def _moving_average(values, window):
	if window <= 1:
		return values
	kernel = np.ones(window, dtype=np.float64) / float(window)
	return np.convolve(values, kernel, mode='same')


# -----------------------------
# 1) Data loading
# -----------------------------
def load_citylearn_data(dataset_dir):
	building_path = os.path.join(dataset_dir, 'Building_1.csv')
	weather_path = os.path.join(dataset_dir, 'weather.csv')

	building = pd.read_csv(building_path)
	weather = pd.read_csv(weather_path)

	temp_1 = _get_column(weather, 'outdoor_dry_bulb_temperature_predicted_1')
	temp_2 = _get_column(weather, 'outdoor_dry_bulb_temperature_predicted_2', temp_1)
	temp_3 = _get_column(weather, 'outdoor_dry_bulb_temperature_predicted_3', temp_2)

	hum_1 = _get_column(weather, 'outdoor_relative_humidity_predicted_1')
	hum_2 = _get_column(weather, 'outdoor_relative_humidity_predicted_2', hum_1)
	hum_3 = _get_column(weather, 'outdoor_relative_humidity_predicted_3', hum_2)

	diffuse_1 = _get_column(weather, 'diffuse_solar_irradiance_predicted_1')
	diffuse_2 = _get_column(weather, 'diffuse_solar_irradiance_predicted_2', diffuse_1)
	diffuse_3 = _get_column(weather, 'diffuse_solar_irradiance_predicted_3', diffuse_2)

	direct_1 = _get_column(weather, 'direct_solar_irradiance_predicted_1')
	direct_2 = _get_column(weather, 'direct_solar_irradiance_predicted_2', direct_1)
	direct_3 = _get_column(weather, 'direct_solar_irradiance_predicted_3', direct_2)

	solar_1 = diffuse_1 + direct_1
	solar_2 = diffuse_2 + direct_2
	solar_3 = diffuse_3 + direct_3

	data = pd.DataFrame({
		'occupant_count': _get_column(building, 'occupant_count'),
		'non_shiftable_load': _get_column(building, 'non_shiftable_load'),
		'dhw_demand': _get_column(building, 'dhw_demand'),
		'cooling_demand': _get_column(building, 'cooling_demand'),
		'solar_generation': _get_column(building, 'solar_generation'),
		'outdoor_temp_pred_1': temp_1,
		'outdoor_temp_pred_2': temp_2,
		'outdoor_temp_pred_3': temp_3,
		'solar_pred_1': solar_1,
		'solar_pred_2': solar_2,
		'solar_pred_3': solar_3,
		'outdoor_humidity_pred_1': hum_1,
		'outdoor_humidity_pred_2': hum_2,
		'outdoor_humidity_pred_3': hum_3,
		'diffuse_solar_pred_1': diffuse_1,
		'diffuse_solar_pred_2': diffuse_2,
		'diffuse_solar_pred_3': diffuse_3,
		'direct_solar_pred_1': direct_1,
		'direct_solar_pred_2': direct_2,
		'direct_solar_pred_3': direct_3,
	})

	data['forecast_temp_mean'] = data[['outdoor_temp_pred_1', 'outdoor_temp_pred_2', 'outdoor_temp_pred_3']].mean(axis=1)
	data['forecast_temp_trend'] = data['outdoor_temp_pred_3'] - data['outdoor_temp_pred_1']
	data['forecast_solar_mean'] = data[['solar_pred_1', 'solar_pred_2', 'solar_pred_3']].mean(axis=1)

	return data


# -----------------------------
# 2) Environment
# -----------------------------
@dataclass
class EnvConfig:
	occupancy_bins: int = 3
	temp_mean_bins: int = 4
	temp_trend_bins: int = 3
	solar_bins: int = 4


class WeatherOccupancyEnv:
	def __init__(self, data, config=None):
		self.data = data.reset_index(drop=True)
		self.n_steps = len(self.data)
		self.config = config or EnvConfig()

		self.action_names = ['off', 'cool_low', 'cool_high']
		self.cool_multipliers = COOL_MULTIPLIERS

		self.occupancy_edges = _safe_edges(self.data['occupant_count'].to_numpy(), self.config.occupancy_bins)
		self.temp_mean_edges = _safe_edges(self.data['forecast_temp_mean'].to_numpy(), self.config.temp_mean_bins)
		self.temp_trend_edges = np.array([-4.0, -0.6, 0.6, 4.0], dtype=np.float64)
		self.solar_edges = _safe_edges(self.data['forecast_solar_mean'].to_numpy(), self.config.solar_bins)

		self.state_shape = (
			self.config.occupancy_bins,
			self.config.temp_mean_bins,
			self.config.temp_trend_bins,
			self.config.solar_bins,
		)
		self.total_states = int(np.prod(self.state_shape))
		self.total_actions = len(self.action_names)
		self.t = 0

	def reset(self, start_index=0):
		self.t = int(start_index)
		return self._get_state(self.t)

	def _get_state(self, idx):
		row = self.data.iloc[idx]

		occupancy_bin = _bin_value(row['occupant_count'], self.occupancy_edges, self.config.occupancy_bins)
		temp_mean_bin = _bin_value(row['forecast_temp_mean'], self.temp_mean_edges, self.config.temp_mean_bins)
		temp_trend_bin = _bin_value(row['forecast_temp_trend'], self.temp_trend_edges, self.config.temp_trend_bins)
		solar_bin = _bin_value(row['forecast_solar_mean'], self.solar_edges, self.config.solar_bins)

		state_tuple = (
			occupancy_bin,
			temp_mean_bin,
			temp_trend_bin,
			solar_bin,
		)
		return int(np.ravel_multi_index(state_tuple, self.state_shape))

	def step(self, action):
		row = self.data.iloc[self.t]

		base_load = row['non_shiftable_load'] + row['dhw_demand']
		hvac_load = row['cooling_demand'] * self.cool_multipliers[action]
		net_grid = max(0.0, base_load + hvac_load - row['solar_generation'])

		reward = -ENERGY_WEIGHT * net_grid

		self.t += 1
		done = self.t >= self.n_steps
		next_state = None if done else self._get_state(self.t)

		info = {
			'net_grid': net_grid,
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
# 4) Training and evaluation
# -----------------------------
def train_q_learning(env, episodes=TRAIN_EPISODES, episode_length=TRAIN_EPISODE_LENGTH, eps_decay=EPSILON_DECAY):
	if AUTO_EPSILON_DECAY and episodes > 0 and EPSILON_START > EPSILON_MIN:
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
	epsilon_history = []

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
		agent.epsilon = max(EPSILON_MIN, agent.epsilon * eps_decay)
		epsilon_history.append(agent.epsilon)

	print(f'epsilon decay used: {eps_decay:.6f}')
	print(f'epsilon final: {agent.epsilon:.6f}')
	return agent, np.array(rewards_history), np.array(energy_history), np.array(epsilon_history)


def evaluate_policy(env, q_table, start=0, horizon=24 * 7):
	state = env.reset(start)

	total_reward = 0.0
	total_energy = 0.0

	for _ in range(horizon):
		action = int(np.argmax(q_table[state]))
		next_state, reward, done, info = env.step(action)

		total_reward += reward
		total_energy += info['net_grid']

		if done:
			break
		state = next_state

	return {
		'reward': total_reward,
		'energy': total_energy,
	}


def evaluate_fixed_policy(env, fixed_action=0, start=0, horizon=24 * 7):
	state = env.reset(start)

	total_reward = 0.0
	total_energy = 0.0

	for _ in range(horizon):
		next_state, reward, done, info = env.step(int(fixed_action))

		total_reward += reward
		total_energy += info['net_grid']

		if done:
			break
		state = next_state

	return {
		'reward': total_reward,
		'energy': total_energy,
	}


def plot_training_curves(rewards, energies, epsilons):
	if not os.path.exists('./images'):
		os.makedirs('./images')

	smooth_window = max(5, len(rewards) // 25)

	plt.figure(figsize=(11, 10))

	plt.subplot(3, 1, 1)
	plt.plot(rewards, color='tab:blue', alpha=0.28, label='Raw')
	plt.plot(_moving_average(rewards, smooth_window), color='tab:blue', linewidth=2.0, label=f'MA({smooth_window})')
	plt.title('Training reward')
	plt.xlabel('Episode')
	plt.ylabel('Total reward')
	plt.legend(loc='best')

	plt.subplot(3, 1, 2)
	plt.plot(energies, color='tab:orange', alpha=0.28, label='Raw')
	plt.plot(_moving_average(energies, smooth_window), color='tab:orange', linewidth=2.0, label=f'MA({smooth_window})')
	plt.title('Episode net grid energy')
	plt.xlabel('Episode')
	plt.ylabel('Net grid energy')
	plt.legend(loc='best')

	plt.subplot(3, 1, 3)
	plt.plot(epsilons, color='tab:green')
	plt.title('Epsilon decay')
	plt.xlabel('Episode')
	plt.ylabel('Epsilon')
	plt.ylim(0.0, max(0.4, float(np.max(epsilons)) + 0.02))

	plt.tight_layout()
	plt.savefig('./images/q_learning0604_02_training.png', dpi=140)
	plt.close()


def _rollout_series(env, q_table=None, fixed_action=None, start=0, horizon=24 * 7):
	state = env.reset(start)
	energies = []
	actions = []

	for _ in range(horizon):
		if q_table is not None:
			action = int(np.argmax(q_table[state]))
		else:
			action = int(fixed_action)

		next_state, _reward, done, info = env.step(action)
		energies.append(info['net_grid'])
		actions.append(action)

		if done:
			break
		state = next_state

	return {
		'energies': np.array(energies),
		'actions': np.array(actions),
	}


def plot_policy_comparison(env, q_table, fixed_action=0, start=0, horizon=24 * 7):
	if not os.path.exists('./images'):
		os.makedirs('./images')

	learned = _rollout_series(env, q_table=q_table, start=start, horizon=horizon)
	baseline = _rollout_series(env, fixed_action=fixed_action, start=start, horizon=horizon)

	steps = np.arange(len(learned['energies']))
	learned_cum_energy = np.cumsum(learned['energies'])
	baseline_cum_energy = np.cumsum(baseline['energies'])
	learned_action_counts = np.bincount(learned['actions'], minlength=len(env.action_names))
	baseline_action_counts = np.bincount(baseline['actions'], minlength=len(env.action_names))

	plt.figure(figsize=(12, 12))

	plt.subplot(4, 1, 1)
	plt.plot(steps, learned['energies'], color='tab:blue', alpha=0.8, label='Learned policy')
	plt.plot(steps, baseline['energies'], color='tab:gray', alpha=0.75, label=f'Baseline action {fixed_action}')
	plt.title('Hourly net grid energy')
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
	x = np.arange(len(env.action_names))
	width = 0.38
	plt.bar(x - width / 2, learned_action_counts, width=width, color='tab:blue', alpha=0.8, label='Learned policy')
	plt.bar(x + width / 2, baseline_action_counts, width=width, color='tab:gray', alpha=0.8, label='Baseline')
	plt.xticks(x, env.action_names)
	plt.title('Action usage count')
	plt.xlabel('Action')
	plt.ylabel('Count')
	plt.legend(loc='best')

	plt.subplot(4, 1, 4)
	plt.axis('off')
	plt.text(0.01, 0.8, 'The model uses only weather forecasts and occupancy in the state.', fontsize=11)
	plt.text(0.01, 0.55, 'Reward is only based on energy use.', fontsize=11)
	plt.text(0.01, 0.3, f'Baseline action: {env.action_names[fixed_action]}', fontsize=11)

	plt.tight_layout()
	plt.savefig('./images/q_learning0604_02_policy_comparison.png', dpi=140)
	plt.close()


def main():
	np.random.seed(RANDOM_SEED)

	data = load_citylearn_data(DATASET_DIR)
	env = WeatherOccupancyEnv(data)

	agent, rewards, energies, epsilons = train_q_learning(env)

	learned_metrics = evaluate_policy(env, agent.q, start=0)
	baseline_metrics = evaluate_fixed_policy(env, fixed_action=FIXED_BASELINE_ACTION, start=0)

	energy_saving_abs = baseline_metrics['energy'] - learned_metrics['energy']
	energy_saving_pct = (
		100.0 * energy_saving_abs / baseline_metrics['energy']
		if baseline_metrics['energy'] > 0
		else 0.0
	)

	plot_training_curves(rewards, energies, epsilons)
	plot_policy_comparison(env, agent.q, fixed_action=FIXED_BASELINE_ACTION, start=0, horizon=24 * 7)

	print('Training done.')
	print(f'Q-table shape: {agent.q.shape}')
	print(
		'Training action counts: '
		f"off={agent.action_counts[0]}, cool_low={agent.action_counts[1]}, cool_high={agent.action_counts[2]}"
	)
	print(f'Learned reward (1 week): {learned_metrics["reward"]:.2f}')
	print(f'Learned energy (1 week): {learned_metrics["energy"]:.2f}')
	print(f'Baseline reward (1 week): {baseline_metrics["reward"]:.2f}')
	print(f'Baseline energy (1 week): {baseline_metrics["energy"]:.2f}')
	print('--- Savings vs baseline ---')
	print(f'Absolute energy saving (1 week): {energy_saving_abs:.2f}')
	print(f'Percent energy saving (1 week): {energy_saving_pct:.2f}%')


if __name__ == '__main__':
	main()
