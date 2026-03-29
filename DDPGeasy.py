import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from citylearn.citylearn import CityLearnEnv


SCENARIO = "citylearn_challenge_2020_climate_zone_1"


def unwrap_obs(obs):
	# CityLearn central agent observations are usually [obs_vector].
	if isinstance(obs, list) and len(obs) == 1:
		return np.array(obs[0], dtype=np.float32)
	return np.array(obs, dtype=np.float32)


def parse_reset(reset_out):
	# Supports both old API (obs) and gymnasium API (obs, info).
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


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, action_dim),
			nn.Tanh(),
		)

	def forward(self, x):
		return self.net(x)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim + action_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
		)

	def forward(self, s, a):
		return self.net(torch.cat([s, a], dim=1))


def main():
	np.random.seed(0)
	random.seed(0)
	torch.manual_seed(0)

	env = CityLearnEnv(SCENARIO, central_agent=True)

	first_obs = unwrap_obs(parse_reset(env.reset()))
	state_dim = first_obs.shape[0]
	action_dim = int(env.action_space[0].shape[0])
	action_low = env.action_space[0].low.astype(np.float32)
	action_high = env.action_space[0].high.astype(np.float32)

	actor = Actor(state_dim, action_dim)
	actor_t = Actor(state_dim, action_dim)
	actor_t.load_state_dict(actor.state_dict())

	critic = Critic(state_dim, action_dim)
	critic_t = Critic(state_dim, action_dim)
	critic_t.load_state_dict(critic.state_dict())

	opt_a = optim.Adam(actor.parameters(), lr=1e-4)
	opt_c = optim.Adam(critic.parameters(), lr=1e-3)

	gamma = 0.99
	tau = 0.005
	batch = 32
	episodes = 30  # fakt jednoduchy rychly test
	max_steps = 200
	noise_std = 0.1

	replay = deque(maxlen=5000)

	for ep in range(1, episodes + 1):
		s = unwrap_obs(parse_reset(env.reset()))
		ep_reward = 0.0

		for _ in range(max_steps):
			with torch.no_grad():
				a = actor(torch.tensor(s).unsqueeze(0)).squeeze(0).numpy()

			a = np.clip(a + np.random.normal(0, noise_std, size=action_dim), -1.0, 1.0)
			a_real = action_low + (a + 1.0) * 0.5 * (action_high - action_low)

			out = env.step([a_real.astype(np.float32)])
			next_obs, r, done, _ = parse_step(out)
			s2 = unwrap_obs(next_obs)

			replay.append((s, a.astype(np.float32), r, s2, float(done)))
			s = s2
			ep_reward += r

			if len(replay) >= batch:
				sample = random.sample(replay, batch)
				ss, aa, rr, s2s, dd = map(np.array, zip(*sample))

				ss = torch.tensor(ss, dtype=torch.float32)
				aa = torch.tensor(aa, dtype=torch.float32)
				rr = torch.tensor(rr, dtype=torch.float32).unsqueeze(1)
				s2s = torch.tensor(s2s, dtype=torch.float32)
				dd = torch.tensor(dd, dtype=torch.float32).unsqueeze(1)

				with torch.no_grad():
					y = rr + (1 - dd) * gamma * critic_t(s2s, actor_t(s2s))

				loss_c = nn.MSELoss()(critic(ss, aa), y)
				opt_c.zero_grad()
				loss_c.backward()
				opt_c.step()

				loss_a = -critic(ss, actor(ss)).mean()
				opt_a.zero_grad()
				loss_a.backward()
				opt_a.step()

				for p, tp in zip(actor.parameters(), actor_t.parameters()):
					tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
				for p, tp in zip(critic.parameters(), critic_t.parameters()):
					tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

			if done:
				break

		print(f"Episode {ep}: total_reward={ep_reward:.2f}")

	print("Hotovo: DDPG zaklad bezi nad CityLearn.")


if __name__ == "__main__":
	main()
