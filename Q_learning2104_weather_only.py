from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import product as iterproduct
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_3_3' / 'schema.json'
DEFAULT_TRAIN_BUILDINGS = ['Building_1', 'Building_2']
DEFAULT_EVAL_BUILDINGS = ['Building_3']
DEFAULT_SEEDS = [7]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning2104_weather_only'

ACTIVE_OBSERVATIONS = [
    'outdoor_dry_bulb_temperature_predicted_1',
    'outdoor_dry_bulb_temperature_predicted_2',
    'diffuse_solar_irradiance_predicted_1',
    'diffuse_solar_irradiance_predicted_2',
    'direct_solar_irradiance_predicted_1',
    'direct_solar_irradiance_predicted_2',
    'occupant_count',
]

OBSERVATION_BIN_SIZES = {
    'temp_pred_1': 3,
    'temp_trend_12': 3,
    'solar_pred_1_total': 3,
    'solar_trend_12_total': 3,
    'occupancy_present': 2,
}
ENGINEERED_FEATURES = list(OBSERVATION_BIN_SIZES.keys())

ACTIVE_ACTIONS = ['cooling_device']
ACTION_BIN_COUNTS = [3]

POLICY_COLORS = [
    '#9aa0a6',
    '#1d3557',
]


class WeatherOccupancyReward(RewardFunction):
    """Penalize grid import more when the building is occupied and hot weather is forecast."""

    def __init__(self, env_metadata, occupancy_weight: float = 1.5, hot_weather_weight: float = 0.05, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.occupancy_weight = float(occupancy_weight)
        self.hot_weather_weight = float(hot_weather_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            occ = max(obs.get('occupant_count', 0.0), 0.0)
            occ_present = 1.0 if occ > 0.0 else 0.0
            mean_forecast = np.mean([
                obs.get('outdoor_dry_bulb_temperature_predicted_1', 0.0),
                obs.get('outdoor_dry_bulb_temperature_predicted_2', 0.0),
                obs.get('outdoor_dry_bulb_temperature_predicted_3', 0.0),
            ])
            occ_factor = 1.0 + self.occupancy_weight * occ_present
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            reward_list.append(-(grid_import * occ_factor * weather_factor))
        return [sum(reward_list)] if self.central_agent else reward_list


class FixedPolicy:
    def __init__(self, cooling_action: float = 0.5):
        self.actions = [float(cooling_action)]

    def reset(self) -> None:
        pass

    def predict(self, observations: list[list[float]], deterministic: bool = None) -> list[list[float]]:
        return [list(self.actions) for _ in observations]


class ObservationDiscretizer:
    def __init__(self, env: CityLearnEnv, bin_counts: dict[str, int]):
        self.observation_names = env.observation_names[0]
        self.index_by_name = {name: i for i, name in enumerate(self.observation_names)}
        self.feature_names = [
            'temp_pred_1', 'temp_trend_12',
            'solar_pred_1_total', 'solar_trend_12_total',
            'occupancy_present',
        ]
        self.bin_counts = [int(bin_counts[name]) for name in self.feature_names]
        self.state_shape = tuple(self.bin_counts)
        self.state_count = int(np.prod(self.state_shape))

        low_by_name = {n: float(v) for n, v in zip(self.observation_names, env.observation_space[0].low)}
        high_by_name = {n: float(v) for n, v in zip(self.observation_names, env.observation_space[0].high)}

        t1_lo, t1_hi = low_by_name['outdoor_dry_bulb_temperature_predicted_1'], high_by_name['outdoor_dry_bulb_temperature_predicted_1']
        t2_lo, t2_hi = low_by_name['outdoor_dry_bulb_temperature_predicted_2'], high_by_name['outdoor_dry_bulb_temperature_predicted_2']
        d1_lo, d1_hi = low_by_name['diffuse_solar_irradiance_predicted_1'], high_by_name['diffuse_solar_irradiance_predicted_1']
        d2_lo, d2_hi = low_by_name['diffuse_solar_irradiance_predicted_2'], high_by_name['diffuse_solar_irradiance_predicted_2']
        r1_lo, r1_hi = low_by_name['direct_solar_irradiance_predicted_1'], high_by_name['direct_solar_irradiance_predicted_1']
        r2_lo, r2_hi = low_by_name['direct_solar_irradiance_predicted_2'], high_by_name['direct_solar_irradiance_predicted_2']
        s1_lo, s1_hi = d1_lo + r1_lo, d1_hi + r1_hi
        s2_lo, s2_hi = d2_lo + r2_lo, d2_hi + r2_hi

        feature_lows = [t1_lo, t2_lo - t1_hi, s1_lo, s2_lo - s1_hi, 0.0]
        feature_highs = [t1_hi, t2_hi - t1_lo, s1_hi, s2_hi - s1_lo, 1.0]
        self.edges = [
            np.linspace(float(low), float(high), count + 1)[1:-1]
            for low, high, count in zip(feature_lows, feature_highs, self.bin_counts)
        ]

    def encode(self, observation: list[float]) -> int:
        idx = self.index_by_name
        t1 = float(observation[idx['outdoor_dry_bulb_temperature_predicted_1']])
        t2 = float(observation[idx['outdoor_dry_bulb_temperature_predicted_2']])
        d1 = float(observation[idx['diffuse_solar_irradiance_predicted_1']])
        d2 = float(observation[idx['diffuse_solar_irradiance_predicted_2']])
        r1 = float(observation[idx['direct_solar_irradiance_predicted_1']])
        r2 = float(observation[idx['direct_solar_irradiance_predicted_2']])
        occ = 1.0 if float(observation[idx['occupant_count']]) > 0.0 else 0.0
        values = [t1, t2 - t1, d1 + r1, (d2 + r2) - (d1 + r1), occ]
        digits = [int(np.digitize(float(value), edge, right=False)) for value, edge in zip(values, self.edges)]
        return int(np.ravel_multi_index(tuple(digits), self.state_shape))


class MultiActionDiscretizer:
    def __init__(self, env: CityLearnEnv, bin_counts_per_action: list[int]):
        n_dims = env.action_space[0].shape[0]
        if len(bin_counts_per_action) != n_dims:
            raise ValueError(f'Expected {n_dims} bin counts, got {len(bin_counts_per_action)}.')
        lows = env.action_space[0].low.tolist()
        highs = env.action_space[0].high.tolist()
        self.value_grids = [
            np.linspace(float(low), float(high), int(count), dtype=float)
            for low, high, count in zip(lows, highs, bin_counts_per_action)
        ]
        self.joint_actions: list[tuple[float, ...]] = list(iterproduct(*self.value_grids))

    @property
    def action_count(self) -> int:
        return len(self.joint_actions)

    def decode_one(self, action_index: int) -> list[float]:
        return list(self.joint_actions[action_index])


class OwnAdaptiveTabularQLearning:
    def __init__(self, env: CityLearnEnv, observation_bin_sizes: dict[str, int],
                 action_bin_counts: list[int], learning_rate: float = 0.15,
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 minimum_epsilon: float = 0.05, epsilon_decay: float = 0.03,
                 adaptive_patience: int = 8, adaptive_epsilon_boost: float = 0.08,
                 adaptive_min_improvement: float = 0.01, random_seed: int = 7):
        self.observation_discretizer = ObservationDiscretizer(env, observation_bin_sizes)
        self.action_discretizer = MultiActionDiscretizer(env, action_bin_counts)
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.epsilon_init = float(epsilon)
        self.minimum_epsilon = float(minimum_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.adaptive_patience = int(adaptive_patience)
        self.adaptive_epsilon_boost = float(adaptive_epsilon_boost)
        self.adaptive_min_improvement = float(adaptive_min_improvement)
        self.random_state = np.random.RandomState(random_seed)
        self.q_table = np.zeros(
            (self.observation_discretizer.state_count, self.action_discretizer.action_count), dtype=np.float32)
        self.episode_index = 0
        self.last_state_indices: list[int] = []
        self.last_action_indices: list[int] = []
        self.best_rolling_reward = -np.inf
        self.episodes_since_improvement = 0
        state_count = self.observation_discretizer.state_count
        action_count = self.action_discretizer.action_count
        print(
            f'  [Q-table] states={state_count}, joint_actions={action_count}, '
            f'cells={state_count * action_count:,} ({state_count * action_count * 4 / 1024 / 1024:.2f} MB)',
            flush=True,
        )

    def reset(self) -> None:
        self.last_state_indices = []
        self.last_action_indices = []

    def predict(self, observations: list[list[float]], deterministic: bool = False) -> list[list[float]]:
        self.last_state_indices = []
        self.last_action_indices = []
        actions: list[list[float]] = []
        for observation in observations:
            state_index = self.observation_discretizer.encode(observation)
            if deterministic or self.random_state.rand() > self.epsilon:
                action_index = int(np.argmax(self.q_table[state_index]))
            else:
                action_index = int(self.random_state.randint(self.action_discretizer.action_count))
            self.last_state_indices.append(state_index)
            self.last_action_indices.append(action_index)
            actions.append(self.action_discretizer.decode_one(action_index))
        return actions

    def update(self, rewards: list[float], next_observations: list[list[float]], terminated: bool) -> None:
        for state_index, action_index, reward, next_observation in zip(
            self.last_state_indices, self.last_action_indices, rewards, next_observations
        ):
            next_state = self.observation_discretizer.encode(next_observation)
            best_next = 0.0 if terminated else float(np.max(self.q_table[next_state]))
            td_target = float(reward) + self.discount_factor * best_next
            self.q_table[state_index, action_index] += self.learning_rate * (
                td_target - float(self.q_table[state_index, action_index])
            )

    def finish_episode(self, reward_history: Sequence[float]) -> None:
        self.episode_index += 1
        self.epsilon = max(self.minimum_epsilon, self.epsilon_init * np.exp(-self.epsilon_decay * self.episode_index))
        if len(reward_history) >= 5:
            rolling = float(np.mean(reward_history[-5:]))
            if rolling > self.best_rolling_reward + self.adaptive_min_improvement:
                self.best_rolling_reward = rolling
                self.episodes_since_improvement = 0
            else:
                self.episodes_since_improvement += 1
            if self.episodes_since_improvement >= self.adaptive_patience:
                self.epsilon = min(0.35, self.epsilon + self.adaptive_epsilon_boost)
                self.episodes_since_improvement = 0


@dataclass
class ExperimentResult:
    policy: str
    seed: int | None
    total_grid_import_kwh: float
    total_net_consumption_kwh: float
    discomfort_proportion: float
    cumulative_reward: float
    savings_vs_fixed_pct: float | None = None
    training_seconds: float | None = None
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
    epsilons: list[float]
    training_seconds: float
    stability_episode: int | None


def make_env(schema_path: Path, building_names: list[str], random_seed: int,
             reward_function=WeatherOccupancyReward) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=False,
        buildings=building_names,
        active_observations=ACTIVE_OBSERVATIONS,
        active_actions=ACTIVE_ACTIONS,
        reward_function=reward_function,
        random_seed=random_seed,
    )


def train_q_learning(agent: OwnAdaptiveTabularQLearning, env: CityLearnEnv,
                     episodes: int, progress_every: int) -> TrainingTrace:
    episode_rewards: list[float] = []
    epsilons: list[float] = []
    start_time = time.perf_counter()
    for episode in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        terminated = False
        cumulative_reward = 0.0
        while not terminated:
            actions = agent.predict(observations, deterministic=False)
            next_obs, rewards, terminated, _, _ = env.step(actions)
            agent.update(rewards, next_obs, terminated)
            observations = next_obs
            cumulative_reward += float(np.sum(rewards))
        episode_rewards.append(cumulative_reward)
        agent.finish_episode(episode_rewards)
        epsilons.append(agent.epsilon)
        if progress_every > 0 and ((episode + 1) % progress_every == 0 or episode == 0 or episode + 1 == episodes):
            elapsed = time.perf_counter() - start_time
            rolling = float(np.mean(episode_rewards[-5:])) if len(episode_rewards) >= 5 else cumulative_reward
            print(
                f'  Episode {episode + 1}/{episodes} | reward={cumulative_reward:.2f} | '
                f'rolling5={rolling:.2f} | epsilon={agent.epsilon:.3f} | elapsed={elapsed:.1f}s',
                flush=True,
            )
    return TrainingTrace(
        episode_rewards=episode_rewards,
        epsilons=epsilons,
        training_seconds=time.perf_counter() - start_time,
        stability_episode=estimate_stability_episode(episode_rewards),
    )


def run_policy(agent, env: CityLearnEnv, deterministic: bool = True) -> PolicyRun:
    observations, _ = env.reset()
    agent.reset()
    terminated = False
    cumulative_reward = 0.0
    reward_trace: list[float] = []
    while not terminated:
        actions = agent.predict(observations, deterministic=deterministic)
        observations, rewards, terminated, _, _ = env.step(actions)
        step_reward = float(np.sum(rewards))
        reward_trace.append(step_reward)
        cumulative_reward += step_reward

    base_env = env.unwrapped
    buildings = base_env.buildings
    kpis = base_env.evaluate()
    building_names = [building.name for building in buildings]
    discomfort_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_proportion')]
    discomfort = float(discomfort_rows['value'].mean()) if not discomfort_rows.empty else 0.0
    aggregate_net = np.zeros(len(reward_trace), dtype=float)
    trajectory_data: dict = {
        'time_step': np.arange(len(reward_trace)),
        'reward': reward_trace,
        'cumulative_reward': np.cumsum(reward_trace),
    }
    for building in buildings:
        net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)[:len(reward_trace)]
        aggregate_net += net_consumption
        trajectory_data[f'grid_import_{building.name}_kwh'] = np.clip(net_consumption, 0.0, None)
    trajectory_data['grid_import_kwh'] = np.clip(aggregate_net, 0.0, None)
    trajectory_data['cumulative_grid_import_kwh'] = np.cumsum(np.clip(aggregate_net, 0.0, None))
    return PolicyRun(
        result=ExperimentResult(
            policy=agent.__class__.__name__,
            seed=None,
            total_grid_import_kwh=float(np.sum(np.clip(aggregate_net, 0.0, None))),
            total_net_consumption_kwh=float(np.sum(aggregate_net)),
            discomfort_proportion=discomfort,
            cumulative_reward=cumulative_reward,
        ),
        trajectory=pd.DataFrame(trajectory_data),
        kpis=kpis,
    )


def estimate_stability_episode(rewards: Sequence[float], window: int = 10, tolerance: float = 0.03) -> int | None:
    if len(rewards) < window * 2:
        return None
    arr = np.asarray(rewards, dtype=float)
    for index in range((window * 2) - 1, len(arr)):
        previous = arr[index - (2 * window) + 1:index - window + 1]
        current = arr[index - window + 1:index + 1]
        previous_mean = float(np.mean(previous))
        current_mean = float(np.mean(current))
        scale = max(1.0, abs(previous_mean))
        if abs(current_mean - previous_mean) / scale <= tolerance and float(np.std(current)) / max(1.0, abs(current_mean)) <= tolerance * 1.5:
            return index + 1
    return None


def build_results_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'policy': result.policy,
            'seed': result.seed,
            'grid_import_kwh': round(result.total_grid_import_kwh, 3),
            'net_consumption_kwh': round(result.total_net_consumption_kwh, 3),
            'discomfort_proportion': round(result.discomfort_proportion, 4),
            'cumulative_reward': round(result.cumulative_reward, 3),
            'savings_vs_fixed_pct': None if result.savings_vs_fixed_pct is None else round(result.savings_vs_fixed_pct, 3),
            'training_seconds': None if result.training_seconds is None else round(result.training_seconds, 2),
            'stability_episode': result.stability_episode,
            'last_10_episode_reward_mean': None if result.last_10_episode_reward_mean is None else round(result.last_10_episode_reward_mean, 3),
        }
        for result in results
    ])


def save_policy_comparison_figure(results_frame: pd.DataFrame, policy_runs: list[PolicyRun], output_path: Path) -> None:
    labels = results_frame['policy'].tolist()
    x_axis = np.arange(len(labels))
    colors = POLICY_COLORS[:len(labels)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].bar(x_axis, results_frame['grid_import_kwh'], color=colors)
    axes[0, 0].set_title('Total grid import')
    axes[0, 0].set_ylabel('kWh')
    axes[0, 0].set_xticks(x_axis, labels, rotation=25, ha='right', fontsize=8)
    axes[0, 0].grid(axis='y', alpha=0.25)

    axes[0, 1].bar(x_axis, results_frame['savings_vs_fixed_pct'].fillna(0.0), color=colors)
    axes[0, 1].set_title('Savings vs fixed strategy')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].set_xticks(x_axis, labels, rotation=25, ha='right', fontsize=8)
    axes[0, 1].grid(axis='y', alpha=0.25)

    axes[1, 0].bar(x_axis, results_frame['discomfort_proportion'], color=colors)
    axes[1, 0].set_title('Discomfort proportion')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].set_xticks(x_axis, labels, rotation=25, ha='right', fontsize=8)
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
    axes[1, 1].legend(loc='best', fontsize=8)

    fig.suptitle('Weather+occupancy transfer: train on Buildings 1-2, eval on Building 3')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_time_and_learning_comparison(fixed_run: PolicyRun, learned_run: PolicyRun,
                                      training_trace: TrainingTrace, output_path: Path,
                                      horizon: int) -> None:
    max_horizon = min(len(fixed_run.trajectory), len(learned_run.trajectory))
    horizon = max_horizon if horizon <= 0 else min(horizon, max_horizon)
    fixed_slice = fixed_run.trajectory.head(horizon)
    learned_slice = learned_run.trajectory.head(horizon)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(fixed_slice['time_step'], fixed_slice['reward'], '--', lw=1.6, color='#9aa0a6', label='Fixed')
    axes[0, 0].plot(learned_slice['time_step'], learned_slice['reward'], lw=1.9, color='#1d3557', label='Q-learning')
    axes[0, 0].set_title('Step reward over time')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(fixed_slice['time_step'], fixed_slice['cumulative_reward'], '--', lw=1.6, color='#9aa0a6', label='Fixed')
    axes[0, 1].plot(learned_slice['time_step'], learned_slice['cumulative_reward'], lw=1.9, color='#1d3557', label='Q-learning')
    axes[0, 1].set_title('Cumulative reward over time')
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

    axes[1, 1].plot(fixed_slice['time_step'], fixed_slice['cumulative_grid_import_kwh'], '--', lw=1.6, color='#9aa0a6', label='Fixed')
    axes[1, 1].plot(learned_slice['time_step'], learned_slice['cumulative_grid_import_kwh'], lw=1.9, color='#1d3557', label='Q-learning')
    axes[1, 1].set_title('Cumulative grid import over time')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=9)

    fig.suptitle('Weather+occupancy only state, cooling action only')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(schema_path: Path, train_buildings: list[str], eval_buildings: list[str], episodes: int,
                   baseline_cooling: float, random_seeds: list[int], output_dir: Path,
                   comparison_horizon: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    print('Evaluating fixed cooling strategy on evaluation buildings...', flush=True)
    fixed_env = make_env(schema_path, eval_buildings, random_seeds[0])
    fixed_policy = FixedPolicy(cooling_action=baseline_cooling)
    fixed_run = run_policy(fixed_policy, fixed_env, deterministic=True)
    fixed_run.result.policy = f'Fixed(cool={baseline_cooling:.2f})'
    fixed_run.result.seed = None
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    learned_runs: list[tuple[str, str, PolicyRun, TrainingTrace, OwnAdaptiveTabularQLearning]] = []

    for seed in random_seeds:
        print(f'\nTraining Q-learning with Weather reward, seed={seed}...', flush=True)
        train_env = make_env(schema_path, train_buildings, seed, reward_function=WeatherOccupancyReward)
        agent = OwnAdaptiveTabularQLearning(
            train_env,
            observation_bin_sizes=OBSERVATION_BIN_SIZES,
            action_bin_counts=ACTION_BIN_COUNTS,
            epsilon=1.0,
            minimum_epsilon=0.05,
            epsilon_decay=0.03,
            learning_rate=0.15,
            discount_factor=0.95,
            adaptive_patience=8,
            adaptive_epsilon_boost=0.08,
            adaptive_min_improvement=0.01,
            random_seed=seed,
        )
        trace = train_q_learning(agent, train_env, episodes=episodes, progress_every=max(1, episodes // 20))
        eval_env = make_env(schema_path, eval_buildings, seed, reward_function=WeatherOccupancyReward)
        run = run_policy(agent, eval_env, deterministic=True)
        run.result.policy = f'Q-learning (Weather reward, seed={seed})'
        run.result.seed = seed
        run.result.training_seconds = trace.training_seconds
        run.result.stability_episode = trace.stability_episode
        run.result.last_10_episode_reward_mean = float(np.mean(trace.episode_rewards[-10:])) if trace.episode_rewards else None
        if fixed_run.result.total_grid_import_kwh > 0.0:
            run.result.savings_vs_fixed_pct = 100.0 * (
                fixed_run.result.total_grid_import_kwh - run.result.total_grid_import_kwh
            ) / fixed_run.result.total_grid_import_kwh
        results.append(run.result)
        learned_runs.append((f'weather_seed_{seed}', f'Weather reward (seed={seed})', run, trace, agent))

    results_frame = build_results_frame(results).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)

    all_policy_runs = [fixed_run] + [entry[2] for entry in learned_runs]
    save_policy_comparison_figure(results_frame, all_policy_runs, output_dir / 'policy_comparison.png')

    for key, _display_name, run, trace, agent in learned_runs:
        save_time_and_learning_comparison(
            fixed_run,
            run,
            trace,
            output_dir / f'time_and_learning_comparison_{key}.png',
            comparison_horizon,
        )
        np.save(output_dir / f'q_table_{key}.npy', agent.q_table)
        pd.DataFrame({
            'episode': np.arange(1, len(trace.episode_rewards) + 1),
            'episode_reward': trace.episode_rewards,
            'epsilon': trace.epsilons,
        }).to_csv(output_dir / f'learning_trace_{key}.csv', index=False)
        run.trajectory.to_csv(output_dir / f'trajectory_q_learning_{key}.csv', index=False)
        run.kpis.to_csv(output_dir / f'kpis_{key}.csv', index=False)

    fixed_run.trajectory.to_csv(output_dir / 'trajectory_fixed_strategy.csv', index=False)
    fixed_run.kpis.to_csv(output_dir / 'kpis_fixed.csv', index=False)

    return results_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Weather+occupancy only Q-learning example with training on Buildings 1-2 and evaluation on Building 3.'
    )
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument('--train-buildings', nargs='+', default=DEFAULT_TRAIN_BUILDINGS)
    parser.add_argument('--eval-buildings', nargs='+', default=DEFAULT_EVAL_BUILDINGS)
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--baseline-cooling', type=float, default=0.5)
    parser.add_argument('--random-seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--comparison-horizon', type=int, default=2208)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print(f'Schema: {args.schema}', flush=True)
    print(f'Train buildings: {", ".join(args.train_buildings)}', flush=True)
    print(f'Eval buildings: {", ".join(args.eval_buildings)}', flush=True)
    print(f'Engineered features: {", ".join(ENGINEERED_FEATURES)}', flush=True)
    print(f'Feature bins: {OBSERVATION_BIN_SIZES} -> {int(np.prod(list(OBSERVATION_BIN_SIZES.values())))} states', flush=True)
    print(f'Actions: {", ".join(ACTIVE_ACTIONS)}', flush=True)
    print(f'Action bins: {ACTION_BIN_COUNTS} -> {int(np.prod(ACTION_BIN_COUNTS))} joint actions', flush=True)
    print(f'Reward function: WeatherOccupancyReward', flush=True)
    print(f'Seeds: {args.random_seeds}', flush=True)
    print(f'Episodes: {args.episodes}', flush=True)
    print(f'Baseline cooling: {args.baseline_cooling}', flush=True)
    print(f'Output dir: {args.output_dir}', flush=True)

    results = run_experiment(
        schema_path=args.schema,
        train_buildings=args.train_buildings,
        eval_buildings=args.eval_buildings,
        episodes=args.episodes,
        baseline_cooling=args.baseline_cooling,
        random_seeds=args.random_seeds,
        output_dir=args.output_dir,
        comparison_horizon=args.comparison_horizon,
    )

    print('\nResults:', flush=True)
    print(results.to_string(index=False))
    print(f'\nAll files saved in: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
