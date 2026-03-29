import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# 1) Very simple building environment
# -----------------------------
class SimpleBuildingEnv:
	"""
	Minimal toy environment for energy control.

	State (4 values):
	  - indoor_temp: current indoor temperature
	  - occupancy: room occupancy in [0, 1]
	  - outdoor_temp: simple weather signal
	  - price: simple electricity price signal

	Action (1 value in [-1, 1]):
	  - hvac_power: cooling/heating command
		negative -> cooling, positive -> heating
	"""

	def __init__(self, max_steps=48):
		self.max_steps = max_steps
		self.step_idx = 0
		self.indoor_temp = 22.0

		self.state_dim = 4
		self.action_dim = 1
		self.action_low = -1.0
		self.action_high = 1.0

	def _signal(self, t):
		# Daily-like cycles for occupancy/weather/price.
		# Kept very simple for a first runnable baseline.
		occupancy = 0.5 + 0.5 * np.sin(2 * np.pi * t / 24.0)
		outdoor_temp = 15.0 + 10.0 * np.sin(2 * np.pi * (t + 6) / 24.0)
		price = 0.2 + 0.1 * np.sin(2 * np.pi * (t + 3) / 24.0)
		return float(occupancy), float(outdoor_temp), float(price)

	def reset(self):
		self.step_idx = 0
		self.indoor_temp = 22.0 + np.random.uniform(-1.0, 1.0)

		occupancy, outdoor_temp, price = self._signal(self.step_idx)
		state = np.array([self.indoor_temp, occupancy, outdoor_temp, price], dtype=np.float32)
		return state

	def step(self, action):
		action = float(np.clip(action, self.action_low, self.action_high))
		occupancy, outdoor_temp, price = self._signal(self.step_idx)

		# Simple thermal dynamics:
		# - building drifts towards outdoor temperature
		# - hvac action pushes temperature up/down
		self.indoor_temp += 0.1 * (outdoor_temp - self.indoor_temp) + 0.8 * action

		# Comfort target: occupied -> 22C, unoccupied -> 19C (less strict)
		target_temp = 19.0 + 3.0 * occupancy
		comfort_penalty = abs(self.indoor_temp - target_temp)

		# Energy cost increases with action magnitude and price
		energy_penalty = price * (abs(action) ** 2)

		reward = -(0.7 * comfort_penalty + 2.0 * energy_penalty)

		self.step_idx += 1
		done = self.step_idx >= self.max_steps

		next_occupancy, next_outdoor_temp, next_price = self._signal(self.step_idx % self.max_steps)
		next_state = np.array(
			[self.indoor_temp, next_occupancy, next_outdoor_temp, next_price],
			dtype=np.float32,
		)

		return next_state, float(reward), done, {}


# -----------------------------
# 2) Replay buffer
# -----------------------------
class ReplayBuffer:
	def __init__(self, capacity=100_000):
		self.buffer = deque(maxlen=capacity)

	def push(self, state, action, reward, next_state, done):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
		return states, actions, rewards, next_states, dones

	def __len__(self):
		return len(self.buffer)


# -----------------------------
# 3) Neural networks
# -----------------------------
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, action_dim),
			nn.Tanh(),
		)
		self.max_action = max_action

	def forward(self, state):
		return self.max_action * self.net(state)


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(state_dim + action_dim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
		)

	def forward(self, state, action):
		x = torch.cat([state, action], dim=1)
		return self.net(x)


# -----------------------------
# 4) DDPG agent
# -----------------------------
class DDPGAgent:
	def __init__(self, state_dim, action_dim, max_action, device):
		self.device = device
		self.max_action = max_action

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = Critic(state_dim, action_dim).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())

		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

		self.gamma = 0.99
		self.tau = 0.005

	def select_action(self, state_np):
		state = torch.FloatTensor(state_np).unsqueeze(0).to(self.device)
		action = self.actor(state).detach().cpu().numpy()[0]
		return action

	def train_step(self, replay_buffer, batch_size=64):
		if len(replay_buffer) < batch_size:
			return None, None

		states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

		states = torch.FloatTensor(states).to(self.device)
		actions = torch.FloatTensor(actions).to(self.device)
		rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
		next_states = torch.FloatTensor(next_states).to(self.device)
		dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

		with torch.no_grad():
			target_actions = self.actor_target(next_states)
			target_q = self.critic_target(next_states, target_actions)
			y = rewards + (1.0 - dones) * self.gamma * target_q

		current_q = self.critic(states, actions)
		critic_loss = nn.MSELoss()(current_q, y)

		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		actor_loss = -self.critic(states, self.actor(states)).mean()

		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		self._soft_update(self.actor, self.actor_target)
		self._soft_update(self.critic, self.critic_target)

		return float(actor_loss.item()), float(critic_loss.item())

	def _soft_update(self, source, target):
		for src, tgt in zip(source.parameters(), target.parameters()):
			tgt.data.copy_(self.tau * src.data + (1.0 - self.tau) * tgt.data)


# -----------------------------
# 5) Training loop (minimal)
# -----------------------------
def train_ddpg(num_episodes=60):
	np.random.seed(42)
	random.seed(42)
	torch.manual_seed(42)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	env = SimpleBuildingEnv(max_steps=48)
	agent = DDPGAgent(
		state_dim=env.state_dim,
		action_dim=env.action_dim,
		max_action=env.action_high,
		device=device,
	)
	replay_buffer = ReplayBuffer(capacity=50_000)

	batch_size = 64
	noise_std = 0.2

	for episode in range(1, num_episodes + 1):
		state = env.reset()
		done = False
		episode_reward = 0.0
		last_actor_loss = None
		last_critic_loss = None

		while not done:
			action = agent.select_action(state)

			# Add simple Gaussian exploration noise
			noise = np.random.normal(0.0, noise_std, size=env.action_dim)
			action = np.clip(action + noise, env.action_low, env.action_high)

			next_state, reward, done, _ = env.step(action[0])
			replay_buffer.push(state, action, reward, next_state, float(done))

			state = next_state
			episode_reward += reward

			actor_loss, critic_loss = agent.train_step(replay_buffer, batch_size=batch_size)
			if actor_loss is not None:
				last_actor_loss, last_critic_loss = actor_loss, critic_loss

		if episode % 10 == 0 or episode == 1:
			print(
				f"Episode {episode:3d} | Reward: {episode_reward:8.2f} "
				f"| Actor loss: {last_actor_loss} | Critic loss: {last_critic_loss}"
			)

	print("\nTrening hotovy.")


if __name__ == "__main__":
	train_ddpg(num_episodes=60)
