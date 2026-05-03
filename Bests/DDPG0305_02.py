from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction

from Q_learning0305_02 import (
    ACTIVE_ACTIONS,
    ACTIVE_OBSERVATIONS,
    CombinedMultiObjectiveReward,
    ComfortAwareReward,
    GridImportOnlyReward,
    NightPrechargeReward,
    PeakShavingReward,
    PricingAwareReward,
    RampingPenaltyReward,
    REWARD_CONFIGS,
    SelfSufficiencyReward,
    SolarAlignmentReward,
    StorageManagementReward,
    TimeOfUseReward,
    WeatherOccupancyReward,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1' / 'schema.json'
DEFAULT_BUILDINGS = ['Building_1', 'Building_2', 'Building_3']
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_ddpg0305_02'
DEFAULT_EPISODES = 50
DEFAULT_BASELINE_COOLING = 0.5
DEFAULT_RANDOM_SEED = 7
DEFAULT_COMPARISON_HORIZON = 719


class CentralFixedPolicy:
    def __init__(self, action_dim: int, cooling_action: float = 0.25):
        self.action_dim = int(action_dim)
        self.cooling_action = float(cooling_action)
        self.base_pattern = np.array([0.0, 0.0, self.cooling_action], dtype=np.float32)

    def reset(self) -> None:
        pass

    def predict(self, observations: list[list[float]], deterministic: bool = True) -> list[list[float]]:
        tiled_actions = np.tile(self.base_pattern, self.action_dim // len(self.base_pattern))
        return [tiled_actions.astype(float).tolist()]


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.as_tensor(self.state_buffer[idx], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.action_buffer[idx], dtype=torch.float32, device=device)
        rewards = torch.as_tensor(self.reward_buffer[idx], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(self.next_state_buffer[idx], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.done_buffer[idx], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))


class RunningNormalizer:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        finite = np.isfinite(low) & np.isfinite(high) & (high > low)
        safe_low = np.where(finite, low, -1.0)
        safe_high = np.where(finite, high, 1.0)
        self.low = safe_low
        self.high = safe_high
        self.scale = np.maximum(self.high - self.low, 1e-6)

    def normalize(self, observation: Sequence[float]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        normalized = 2.0 * (obs - self.low) / self.scale - 1.0
        return np.clip(normalized, -5.0, 5.0).astype(np.float32)


class DDPGAgent:
    def __init__(
        self,
        env: CityLearnEnv,
        learning_rate_actor: float = 1e-4,
        learning_rate_critic: float = 1e-3,
        discount_factor: float = 0.99,
        tau: float = 5e-3,
        batch_size: int = 128,
        replay_size: int = 100_000,
        warmup_steps: int = 1_000,
        update_every: int = 1,
        policy_noise: float = 0.20,
        noise_decay: float = 0.995,
        min_noise: float = 0.03,
        hidden_dim: int = 256,
        random_seed: int = 7,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)

        self.state_dim = int(env.observation_space[0].shape[0])
        self.action_dim = int(env.action_space[0].shape[0])
        observation_low = np.asarray(env.observation_space[0].low, dtype=np.float32)
        observation_high = np.asarray(env.observation_space[0].high, dtype=np.float32)
        self.action_low = np.asarray(env.action_space[0].low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space[0].high, dtype=np.float32)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        self.normalizer = RunningNormalizer(observation_low, observation_high)
        self.action_low_tensor = torch.as_tensor(self.action_low, dtype=torch.float32, device=self.device)
        self.action_high_tensor = torch.as_tensor(self.action_high, dtype=torch.float32, device=self.device)
        self.action_scale_tensor = torch.as_tensor(self.action_scale, dtype=torch.float32, device=self.device)
        self.action_bias_tensor = torch.as_tensor(self.action_bias, dtype=torch.float32, device=self.device)

        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate_critic)

        self.discount_factor = float(discount_factor)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        self.update_every = int(update_every)
        self.noise_decay = float(noise_decay)
        self.min_noise = float(min_noise)
        self.current_noise = float(policy_noise)
        self.replay_buffer = ReplayBuffer(replay_size, self.state_dim, self.action_dim)
        self.total_steps = 0

    def reset(self) -> None:
        pass

    def _normalize_state(self, observation: Sequence[float]) -> np.ndarray:
        return self.normalizer.normalize(observation)

    def _scale_action(self, raw_action: np.ndarray) -> np.ndarray:
        return np.clip(self.action_bias + raw_action * self.action_scale, self.action_low, self.action_high)

    def predict(self, observations: list[list[float]], deterministic: bool = False) -> list[list[float]]:
        state = self._normalize_state(observations[0])
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw_action = self.actor(state_tensor).cpu().numpy()[0]
        if not deterministic:
            raw_action = raw_action + self.random_state.normal(0.0, self.current_noise, size=self.action_dim)
            raw_action = np.clip(raw_action, -1.0, 1.0)
        action = self._scale_action(raw_action.astype(np.float32))
        return [action.tolist()]

    def remember(
        self,
        observation: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_observation: Sequence[float],
        done: bool,
    ) -> None:
        state = self._normalize_state(observation)
        next_state = self._normalize_state(next_observation)
        self.replay_buffer.push(
            state=state,
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward),
            next_state=next_state,
            done=done,
        )
        self.total_steps += 1

    def update(self) -> tuple[float | None, float | None]:
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None, None
        if self.total_steps % self.update_every != 0:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_actions = self._scale_action_tensor(self.actor_target(next_states))
            target_q = self.critic_target(next_states, next_actions)
            q_target = rewards + (1.0 - dones) * self.discount_factor * target_q

        q_current = self.critic(states, actions)
        critic_loss = F.mse_loss(q_current, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_actions = self._scale_action_tensor(self.actor(states))
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return float(actor_loss.item()), float(critic_loss.item())

    def finish_episode(self) -> None:
        self.current_noise = max(self.min_noise, self.current_noise * self.noise_decay)

    def _scale_action_tensor(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            self.action_bias_tensor + raw_action * self.action_scale_tensor,
            min=self.action_low_tensor,
            max=self.action_high_tensor,
        )

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


@dataclass
class ExperimentResult:
    policy: str
    total_grid_import_kwh: float
    total_export_kwh: float
    total_net_consumption_kwh: float
    discomfort_proportion: float
    discomfort_cold_proportion: float
    discomfort_hot_proportion: float
    all_rewards: float
    cost_total_ratio: float | None = None
    carbon_emissions_total_ratio: float | None = None
    daily_peak_average_ratio: float | None = None
    ramping_average_ratio: float | None = None
    savings_vs_fixed_pct: float | None = None
    train_sec: float | None = None
    stability_episode: int | None = None
    last_10_episode_reward_mean: float | None = None


@dataclass
class PolicyRun:
    result: ExperimentResult
    trajectory: pd.DataFrame
    kpis: pd.DataFrame


@dataclass
class TrainingTrace:
    episode_rewards: list[float]
    actor_losses: list[float]
    critic_losses: list[float]
    exploration_noise: list[float]
    train_sec: float
    stability_episode: int | None


def make_env(schema_path: Path, building_names: list[str], random_seed: int, reward_function: type[RewardFunction] = WeatherOccupancyReward) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=True,
        buildings=building_names,
        active_observations=ACTIVE_OBSERVATIONS,
        active_actions=ACTIVE_ACTIONS,
        reward_function=reward_function,
        random_seed=random_seed,
    )


def estimate_stability_episode(rewards: list[float], window: int = 10, tolerance: float = 0.03) -> int | None:
    if len(rewards) < window * 2:
        return None

    arr = np.asarray(rewards, dtype=float)
    for index in range((window * 2) - 1, len(arr)):
        prev = arr[index - (2 * window) + 1:index - window + 1]
        curr = arr[index - window + 1:index + 1]
        prev_mean = float(np.mean(prev))
        curr_mean = float(np.mean(curr))
        scale = max(1.0, abs(prev_mean))
        stable_mean = abs(curr_mean - prev_mean) / scale <= tolerance
        stable_std = float(np.std(curr)) / max(1.0, abs(curr_mean)) <= tolerance * 2
        if stable_mean and stable_std:
            return index + 1

    return None


def train_ddpg(agent: DDPGAgent, env: CityLearnEnv, episodes: int) -> TrainingTrace:
    episode_rewards: list[float] = []
    actor_losses: list[float] = []
    critic_losses: list[float] = []
    exploration_noise: list[float] = []
    start_time = time.perf_counter()
    progress_every = 30

    for episode in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        terminated = False
        episode_reward = 0.0
        ep_actor_losses: list[float] = []
        ep_critic_losses: list[float] = []

        while not terminated:
            actions = agent.predict(observations, deterministic=False)
            next_observations, rewards, terminated, _, _ = env.step(actions)
            reward = float(np.sum(rewards))
            agent.remember(observations[0], actions[0], reward, next_observations[0], terminated)
            actor_loss, critic_loss = agent.update()
            if actor_loss is not None and critic_loss is not None:
                ep_actor_losses.append(actor_loss)
                ep_critic_losses.append(critic_loss)
            observations = next_observations
            episode_reward += reward

        agent.finish_episode()
        episode_rewards.append(episode_reward)
        actor_losses.append(float(np.mean(ep_actor_losses)) if ep_actor_losses else np.nan)
        critic_losses.append(float(np.mean(ep_critic_losses)) if ep_critic_losses else np.nan)
        exploration_noise.append(agent.current_noise)

        current_episode = episode + 1
        if current_episode == 1 or current_episode % progress_every == 0 or current_episode == episodes:
            rolling_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else episode_reward
            elapsed = time.perf_counter() - start_time
            print(
                f'    Episode {current_episode}/{episodes} | reward={episode_reward:.2f} '
                f'| rolling10={rolling_reward:.2f} | noise={agent.current_noise:.3f} | elapsed={elapsed:.1f}s',
                flush=True,
            )

    return TrainingTrace(
        episode_rewards=episode_rewards,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        exploration_noise=exploration_noise,
        train_sec=time.perf_counter() - start_time,
        stability_episode=estimate_stability_episode(episode_rewards),
    )


def eval_agent(env: CityLearnEnv, agent) -> PolicyRun:
    observations, _ = env.reset()
    agent.reset()
    terminated = False
    reward_trace: list[float] = []
    all_rewards = 0.0

    while not terminated:
        actions = agent.predict(observations, deterministic=True)
        observations, rewards, terminated, _, _ = env.step(actions)
        step_reward = float(np.sum(rewards))
        reward_trace.append(step_reward)
        all_rewards += step_reward

    base_env = env.unwrapped
    buildings = base_env.buildings
    kpis = base_env.evaluate()
    building_names = [building.name for building in buildings]
    discomfort_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_proportion')]
    discomfort_cold_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_cold_proportion')]
    discomfort_hot_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_hot_proportion')]
    discomfort = float(discomfort_rows['value'].mean()) if not discomfort_rows.empty else 0.0
    discomfort_cold = float(discomfort_cold_rows['value'].mean()) if not discomfort_cold_rows.empty else 0.0
    discomfort_hot = float(discomfort_hot_rows['value'].mean()) if not discomfort_hot_rows.empty else 0.0
    district_kpis = kpis[(kpis['name'] == 'District') & (kpis['level'] == 'district')].copy()
    district_metric_map = district_kpis.set_index('cost_function')['value'].to_dict()

    aggregate_net = np.zeros(len(reward_trace), dtype=float)
    trajectory_data: dict[str, np.ndarray | list[float]] = {
        'time_step': np.arange(len(reward_trace)),
        'reward': reward_trace,
        'all_rewards': np.cumsum(reward_trace),
    }
    for building in buildings:
        net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)[:len(reward_trace)]
        aggregate_net += net_consumption
        trajectory_data[f'grid_import_{building.name}_kwh'] = np.clip(net_consumption, 0.0, None)
        trajectory_data[f'export_{building.name}_kwh'] = np.clip(-net_consumption, 0.0, None)

    grid_import = np.clip(aggregate_net, 0.0, None)
    export = np.clip(-aggregate_net, 0.0, None)
    trajectory_data['grid_import_kwh'] = grid_import
    trajectory_data['export_kwh'] = export
    trajectory_data['cumulative_grid_import_kwh'] = np.cumsum(grid_import)
    trajectory_data['cumulative_export_kwh'] = np.cumsum(export)

    return PolicyRun(
        result=ExperimentResult(
            policy='',
            total_grid_import_kwh=float(np.sum(grid_import)),
            total_export_kwh=float(np.sum(export)),
            total_net_consumption_kwh=float(np.sum(aggregate_net)),
            discomfort_proportion=discomfort,
            discomfort_cold_proportion=discomfort_cold,
            discomfort_hot_proportion=discomfort_hot,
            all_rewards=all_rewards,
            cost_total_ratio=float(district_metric_map['cost_total']) if 'cost_total' in district_metric_map and pd.notna(district_metric_map['cost_total']) else None,
            carbon_emissions_total_ratio=float(district_metric_map['carbon_emissions_total']) if 'carbon_emissions_total' in district_metric_map and pd.notna(district_metric_map['carbon_emissions_total']) else None,
            daily_peak_average_ratio=float(district_metric_map['daily_peak_average']) if 'daily_peak_average' in district_metric_map and pd.notna(district_metric_map['daily_peak_average']) else None,
            ramping_average_ratio=float(district_metric_map['ramping_average']) if 'ramping_average' in district_metric_map and pd.notna(district_metric_map['ramping_average']) else None,
        ),
        trajectory=pd.DataFrame(trajectory_data),
        kpis=kpis,
    )


def build_results_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'policy': result.policy,
            'grid_import_kwh': round(result.total_grid_import_kwh, 3),
            'export_kwh': round(result.total_export_kwh, 3),
            'net_consumption_kwh': round(result.total_net_consumption_kwh, 3),
            'discomfort_proportion': round(result.discomfort_proportion, 4),
            'discomfort_cold_proportion': round(result.discomfort_cold_proportion, 4),
            'discomfort_hot_proportion': round(result.discomfort_hot_proportion, 4),
            'all_rewards': round(result.all_rewards, 3),
            'cost_total_ratio': None if result.cost_total_ratio is None else round(result.cost_total_ratio, 4),
            'carbon_emissions_total_ratio': None if result.carbon_emissions_total_ratio is None else round(result.carbon_emissions_total_ratio, 4),
            'daily_peak_average_ratio': None if result.daily_peak_average_ratio is None else round(result.daily_peak_average_ratio, 4),
            'ramping_average_ratio': None if result.ramping_average_ratio is None else round(result.ramping_average_ratio, 4),
            'savings_vs_fixed_pct': None if result.savings_vs_fixed_pct is None else round(result.savings_vs_fixed_pct, 3),
            'train_sec': None if result.train_sec is None else round(result.train_sec, 2),
            'stability_episode': result.stability_episode,
            'last_10_episode_reward_mean': None if result.last_10_episode_reward_mean is None else round(result.last_10_episode_reward_mean, 3),
        }
        for result in results
    ])


def get_policy_colors(count: int) -> list:
    if count <= 0:
        return []

    cmap = plt.get_cmap('tab20')
    colors = ['#9aa0a6']
    for index in range(max(0, count - 1)):
        colors.append(cmap(index % cmap.N))
    return colors[:count]


def save_policy_comparison_figure(results_frame: pd.DataFrame, policy_runs: list[PolicyRun], output_path: Path) -> None:
    labels = results_frame['policy'].tolist()
    x_axis = np.arange(len(labels))
    colors = get_policy_colors(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))

    axes[0, 0].bar(x_axis, results_frame['grid_import_kwh'], color=colors)
    axes[0, 0].set_title('Total grid import')
    axes[0, 0].set_ylabel('kWh')
    axes[0, 0].set_xticks(x_axis, labels, rotation=35, ha='right', fontsize=8)
    axes[0, 0].grid(axis='y', alpha=0.25)

    axes[0, 1].bar(x_axis, results_frame['savings_vs_fixed_pct'].fillna(0.0), color=colors)
    axes[0, 1].set_title('Savings vs fixed strategy')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].set_xticks(x_axis, labels, rotation=35, ha='right', fontsize=8)
    axes[0, 1].grid(axis='y', alpha=0.25)

    axes[1, 0].bar(x_axis, results_frame['discomfort_proportion'], color=colors)
    axes[1, 0].set_title('Discomfort proportion')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].set_xticks(x_axis, labels, rotation=35, ha='right', fontsize=8)
    axes[1, 0].grid(axis='y', alpha=0.25)

    profile_hours = min(14 * 24, min(len(policy_run.trajectory) for policy_run in policy_runs))
    profile_index = np.arange(profile_hours)
    for color, policy_run in zip(colors, policy_runs):
        profile = policy_run.trajectory.groupby(policy_run.trajectory['time_step'] % profile_hours)['grid_import_kwh'].mean()
        linestyle = '--' if policy_run.result.policy.startswith('Fixed') else '-'
        axes[1, 1].plot(
            profile_index,
            profile.reindex(profile_index, fill_value=np.nan).to_numpy(),
            linestyle=linestyle,
            linewidth=1.8,
            color=color,
            label=policy_run.result.policy,
        )
    axes[1, 1].set_title('Average grid import over 14-day profile')
    axes[1, 1].set_xlabel('Hour in 14-day cycle')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].set_xticks(range(0, profile_hours + 1, 24))
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc='best', fontsize=7, ncol=2)

    fig.suptitle(f'3 budovy: Fixed vs {len(labels) - 1} reward variantov DDPG')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_time_and_learning_comparison(fixed_run: PolicyRun, learned_run: PolicyRun, training_trace: TrainingTrace, output_path: Path, horizon: int) -> None:
    max_horizon = min(len(fixed_run.trajectory), len(learned_run.trajectory))
    horizon = max_horizon if horizon <= 0 else min(horizon, max_horizon)
    fixed_slice = fixed_run.trajectory.head(horizon)
    learned_slice = learned_run.trajectory.head(horizon)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(fixed_slice['time_step'], fixed_slice['reward'], '--', lw=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[0, 0].plot(learned_slice['time_step'], learned_slice['reward'], lw=1.9, color='#1d3557', label='DDPG')
    axes[0, 0].set_title('Step reward in time')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(fixed_slice['time_step'], fixed_slice['all_rewards'], '--', lw=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[0, 1].plot(learned_slice['time_step'], learned_slice['all_rewards'], lw=1.9, color='#1d3557', label='DDPG')
    axes[0, 1].set_title('Cumulative reward in time')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Cumulative reward')
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=9)

    episodes = np.arange(1, len(training_trace.episode_rewards) + 1)
    rolling = pd.Series(np.asarray(training_trace.episode_rewards, dtype=float)).rolling(
        min(10, len(training_trace.episode_rewards)), min_periods=1
    ).mean()
    axes[1, 0].plot(episodes, rolling, lw=2.0, color='#2a9d8f')
    axes[1, 0].set_title('Learning progress')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Rolling episode reward')
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(fixed_slice['time_step'], fixed_slice['cumulative_grid_import_kwh'], '--', lw=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[1, 1].plot(learned_slice['time_step'], learned_slice['cumulative_grid_import_kwh'], lw=1.9, color='#1d3557', label='DDPG')
    axes[1, 1].set_title('Cumulative grid import in time')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=9)

    fig.suptitle('3 akcie: DHW zasobnik + bateria + chladenie')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(
    schema_path: Path,
    building_names: list[str],
    episodes: int,
    baseline_cooling: float,
    random_seed: int,
    output_dir: Path,
    comparison_horizon: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    fixed_env = make_env(schema_path, building_names, random_seed)
    fixed_policy = CentralFixedPolicy(action_dim=int(fixed_env.action_space[0].shape[0]), cooling_action=baseline_cooling)
    fixed_run = eval_agent(fixed_env, fixed_policy)
    fixed_run.result.policy = f'Fixed(cool={baseline_cooling:.2f})'
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    learned_runs: list[tuple[str, str, PolicyRun, TrainingTrace, DDPGAgent]] = []
    for key, display_name, reward_cls in REWARD_CONFIGS:
        print(f'Running reward: {display_name}')
        train_env = make_env(schema_path, building_names, random_seed, reward_function=reward_cls)
        agent = DDPGAgent(train_env, random_seed=random_seed)
        trace = train_ddpg(agent, train_env, episodes)

        eval_env = make_env(schema_path, building_names, random_seed, reward_function=reward_cls)
        run = eval_agent(eval_env, agent)
        run.result.policy = f'DDPG ({display_name})'
        run.result.train_sec = trace.train_sec
        run.result.stability_episode = trace.stability_episode
        run.result.last_10_episode_reward_mean = float(np.mean(trace.episode_rewards[-10:])) if trace.episode_rewards else None
        if fixed_run.result.total_grid_import_kwh > 0.0:
            run.result.savings_vs_fixed_pct = 100.0 * (fixed_run.result.total_grid_import_kwh - run.result.total_grid_import_kwh) / fixed_run.result.total_grid_import_kwh

        results.append(run.result)
        learned_runs.append((key, display_name, run, trace, agent))

    results_frame = build_results_frame(results).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)

    all_policy_runs = [fixed_run] + [entry[2] for entry in learned_runs]
    save_policy_comparison_figure(results_frame, all_policy_runs, output_dir / 'policy_comparison.png')

    for key, _display_name, run, trace, agent in learned_runs:
        save_time_and_learning_comparison(
            fixed_run,
            run,
            trace,
            output_dir / f'reward_time_and_learning_comparison_{key}.png',
            comparison_horizon,
        )
        torch.save(agent.actor.state_dict(), output_dir / f'ddpg_actor_{key}.pt')
        torch.save(agent.critic.state_dict(), output_dir / f'ddpg_critic_{key}.pt')
        pd.DataFrame({
            'episode': np.arange(1, len(trace.episode_rewards) + 1),
            'episode_reward': trace.episode_rewards,
            'actor_loss': trace.actor_losses,
            'critic_loss': trace.critic_losses,
            'exploration_noise': trace.exploration_noise,
        }).to_csv(output_dir / f'learning_trace_{key}.csv', index=False)
        run.trajectory.to_csv(output_dir / f'trajectory_ddpg_{key}.csv', index=False)
        run.kpis.to_csv(output_dir / f'kpis_{key}.csv', index=False)

    fixed_run.trajectory.to_csv(output_dir / 'trajectory_fixed_strategy.csv', index=False)
    fixed_run.kpis.to_csv(output_dir / 'kpis_fixed.csv', index=False)

    return results_frame


def main() -> None:
    results = run_experiment(
        schema_path=DEFAULT_SCHEMA,
        building_names=DEFAULT_BUILDINGS,
        episodes=DEFAULT_EPISODES,
        baseline_cooling=DEFAULT_BASELINE_COOLING,
        random_seed=DEFAULT_RANDOM_SEED,
        output_dir=DEFAULT_OUTPUT_DIR,
        comparison_horizon=DEFAULT_COMPARISON_HORIZON,
    )
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()