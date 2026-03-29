import random
from collections import defaultdict

import numpy as np
from citylearn.citylearn import CityLearnEnv


SCENARIO = "citylearn_challenge_2020_climate_zone_1"


def parse_reset(reset_out):
	# Supports old API (obs) and gymnasium API (obs, info).
	return reset_out[0] if isinstance(reset_out, tuple) else reset_out


def parse_step(step_out):
	# Supports old API (obs, reward, done, info) and new API (obs, reward, terminated, truncated, info).
	if len(step_out) == 4:
		obs, reward, done, info = step_out
	else:
		obs, reward, terminated, truncated, info = step_out
		done = bool(terminated or truncated)

	reward_value = float(np.sum(reward)) if isinstance(reward, (list, tuple, np.ndarray)) else float(reward)
	return obs, reward_value, done, info


def build_feature_map(env):
	names = env.observation_names[0]
	idx = {name: i for i, name in enumerate(names)}

	# Keep only a few features so tabular Q-learning stays simple.
	bins = {
		"hour": np.array([6, 12, 18]),
		"outdoor_dry_bulb_temperature": np.array([5, 15, 25, 35]),
		"electricity_pricing": np.array([0.1, 0.2, 0.3, 0.4]),
		"non_shiftable_load": np.array([1.0, 2.0, 3.5, 5.0]),
	}

	selected = [k for k in bins if k in idx]
	return idx, bins, selected


def make_state(obs_first_building, idx_map, bins_map, selected_features):
	if len(selected_features) == 0:
		# Fallback when expected names are missing.
		return tuple(int(np.clip(v * 10.0, -20, 20)) for v in obs_first_building[:3])

	state = []

	for feature in selected_features:
		raw_value = float(obs_first_building[idx_map[feature]])
		state.append(int(np.digitize(raw_value, bins_map[feature])))

	return tuple(state)


def action_list_from_index(env, action_idx, action_levels):
	actions = []

	for building_id, action_space in enumerate(env.action_space):
		a = np.zeros(action_space.shape, dtype=np.float32)

		# Control only first actuator of first building.
		if building_id == 0 and action_space.shape[0] > 0:
			low = float(action_space.low[0])
			high = float(action_space.high[0])
			a[0] = np.clip(action_levels[action_idx], low, high)

		actions.append(a)

	return actions


def zero_policy_actions(env):
	return [np.zeros(space.shape, dtype=np.float32) for space in env.action_space]


def evaluate(env, q_table, idx_map, bins_map, selected_features, action_levels, episodes=2, greedy=True, max_steps=60):
	rewards = []

	for _ in range(episodes):
		obs = parse_reset(env.reset())
		state = make_state(obs[0], idx_map, bins_map, selected_features)
		ep_reward = 0.0
		done = False

		for _ in range(max_steps):
			if greedy:
				action_idx = int(np.argmax(q_table[state]))
				actions = action_list_from_index(env, action_idx, action_levels)
			else:
				actions = zero_policy_actions(env)

			out = env.step(actions)
			obs2, reward, done, _ = parse_step(out)
			state = make_state(obs2[0], idx_map, bins_map, selected_features)
			ep_reward += reward

			if done:
				break

		rewards.append(ep_reward)

	return float(np.mean(rewards))


def main():
	np.random.seed(0)
	random.seed(0)

	env = CityLearnEnv(SCENARIO, central_agent=False)
	idx_map, bins_map, selected_features = build_feature_map(env)
	print("Pouzite features:", selected_features if selected_features else "fallback prve 3 hodnoty")

	action_levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
	q_table = defaultdict(lambda: np.zeros(len(action_levels), dtype=np.float32))

	alpha = 0.15
	gamma = 0.99
	epsilon = 1.0
	epsilon_min = 0.05
	epsilon_decay = 0.92
	episodes = 2
	max_steps = 20

	train_returns = []

	for ep in range(1, episodes + 1):
		obs = parse_reset(env.reset())
		state = make_state(obs[0], idx_map, bins_map, selected_features)
		ep_reward = 0.0
		done = False

		for _ in range(max_steps):
			if np.random.rand() < epsilon:
				action_idx = np.random.randint(len(action_levels))
			else:
				action_idx = int(np.argmax(q_table[state]))

			actions = action_list_from_index(env, action_idx, action_levels)
			out = env.step(actions)
			obs2, reward, done, _ = parse_step(out)
			next_state = make_state(obs2[0], idx_map, bins_map, selected_features)

			best_next = float(np.max(q_table[next_state]))
			old_q = q_table[state][action_idx]
			q_table[state][action_idx] = old_q + alpha * (reward + gamma * best_next - old_q)

			state = next_state
			ep_reward += reward

			if done:
				break

		epsilon = max(epsilon_min, epsilon * epsilon_decay)
		train_returns.append(ep_reward)
		print(f"Episode {ep:2d}: reward={ep_reward:.2f}, epsilon={epsilon:.3f}")

	learned_eval = evaluate(
		env,
		q_table,
		idx_map,
		bins_map,
		selected_features,
		action_levels,
		episodes=1,
		greedy=True,
		max_steps=max_steps,
	)
	baseline_eval = evaluate(
		env,
		q_table,
		idx_map,
		bins_map,
		selected_features,
		action_levels,
		episodes=1,
		greedy=False,
		max_steps=max_steps,
	)

	print("\n=== Jednoduche porovnanie ===")
	print(f"Priemer train reward (poslednych 5 epizod): {np.mean(train_returns[-5:]):.2f}")
	print(f"Learned policy eval reward: {learned_eval:.2f}")
	print(f"Zero policy eval reward:    {baseline_eval:.2f}")
	print("Hotovo: Q-learning zaklad bezi nad CityLearn.")


if __name__ == "__main__":
	main()
