from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
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
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1' / 'schema.json'
DEFAULT_BUILDING = 'Building_1'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning1604_01'
ACTIVE_OBSERVATIONS = [
    'occupant_count',
    'outdoor_dry_bulb_temperature_predicted_1',
    'outdoor_dry_bulb_temperature_predicted_2',
    'outdoor_dry_bulb_temperature_predicted_3',
]
ACTIVE_ACTIONS = ['cooling_device']
OBSERVATION_BIN_SIZES = {
    'occupant_count': 4,
    'outdoor_dry_bulb_temperature_predicted_1': 6,
    'outdoor_dry_bulb_temperature_predicted_2': 6,
    'outdoor_dry_bulb_temperature_predicted_3': 6,
}
ACTION_BIN_COUNT = 9


class WeatherOccupancyReward(RewardFunction):
    def __init__(self, env_metadata, occupancy_weight: float = 1.5, hot_weather_weight: float = 0.05, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.occupancy_weight = float(occupancy_weight)
        self.hot_weather_weight = float(hot_weather_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []

        for observation in observations:
            grid_import = max(observation['net_electricity_consumption'], 0.0)
            occupant_count = max(observation.get('occupant_count', 0.0), 0.0)
            mean_forecast = np.mean([
                observation.get('outdoor_dry_bulb_temperature_predicted_1', 0.0),
                observation.get('outdoor_dry_bulb_temperature_predicted_2', 0.0),
                observation.get('outdoor_dry_bulb_temperature_predicted_3', 0.0),
            ])
            occupancy_factor = 1.0 + self.occupancy_weight * min(occupant_count / 5.0, 1.0)
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            reward_list.append(-(grid_import * occupancy_factor * weather_factor))

        return [sum(reward_list)] if self.central_agent else reward_list


class FixedCoolingPolicy:
    def __init__(self, action_value: float):
        self.action_value = float(action_value)

    def reset(self) -> None:
        pass

    def predict(self, observations: list[list[float]], deterministic: bool = None) -> list[list[float]]:
        return [[self.action_value] for _ in observations]


class ObservationDiscretizer:
    def __init__(self, env: CityLearnEnv, bin_counts: dict[str, int]):
        self.observation_names = env.observation_names[0]
        self.bin_counts = [int(bin_counts[name]) for name in self.observation_names]
        self.state_shape = tuple(self.bin_counts)
        self.state_count = int(np.prod(self.state_shape))
        self.edges = []

        for low, high, count in zip(env.observation_space[0].low, env.observation_space[0].high, self.bin_counts):
            self.edges.append(np.linspace(float(low), float(high), count + 1)[1:-1])

    def encode(self, observation: list[float]) -> int:
        digits = []

        for value, edges in zip(observation, self.edges):
            digits.append(int(np.digitize(float(value), edges, right=False)))

        return int(np.ravel_multi_index(tuple(digits), self.state_shape))


class ActionDiscretizer:
    def __init__(self, env: CityLearnEnv, action_bin_count: int):
        if env.action_space[0].shape[0] != 1:
            raise ValueError('This script expects exactly one controllable action dimension.')

        low = float(env.action_space[0].low[0])
        high = float(env.action_space[0].high[0])
        self.values = np.linspace(low, high, int(action_bin_count), dtype=float)

    @property
    def action_count(self) -> int:
        return len(self.values)

    def decode(self, action_index: int) -> list[list[float]]:
        return [[float(self.values[action_index])]]


class OwnAdaptiveTabularQLearning:
    def __init__(
        self,
        env: CityLearnEnv,
        observation_bin_sizes: dict[str, int],
        action_bin_count: int,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        minimum_epsilon: float = 0.05,
        epsilon_decay: float = 0.03,
        adaptive_patience: int = 8,
        adaptive_epsilon_boost: float = 0.08,
        adaptive_min_improvement: float = 0.01,
        random_seed: int = 7,
    ):
        self.observation_discretizer = ObservationDiscretizer(env, observation_bin_sizes)
        self.action_discretizer = ActionDiscretizer(env, action_bin_count)
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
            (self.observation_discretizer.state_count, self.action_discretizer.action_count),
            dtype=np.float32,
        )
        self.episode_index = 0
        self.last_state_index: int | None = None
        self.last_action_index: int | None = None
        self.best_rolling_reward = -np.inf
        self.episodes_since_improvement = 0

    def reset(self) -> None:
        self.last_state_index = None
        self.last_action_index = None

    def predict(self, observations: list[list[float]], deterministic: bool = False) -> list[list[float]]:
        state_index = self.observation_discretizer.encode(observations[0])

        if deterministic or self.random_state.rand() > self.epsilon:
            action_index = int(np.argmax(self.q_table[state_index]))
        else:
            action_index = int(self.random_state.randint(self.action_discretizer.action_count))

        self.last_state_index = state_index
        self.last_action_index = action_index
        return self.action_discretizer.decode(action_index)

    def update(self, reward: float, next_observations: list[list[float]], terminated: bool) -> None:
        if self.last_state_index is None or self.last_action_index is None:
            raise RuntimeError('Cannot update Q-table before selecting an action.')

        next_state_index = self.observation_discretizer.encode(next_observations[0])
        best_next_value = 0.0 if terminated else float(np.max(self.q_table[next_state_index]))
        td_target = float(reward) + self.discount_factor * best_next_value
        td_error = td_target - float(self.q_table[self.last_state_index, self.last_action_index])
        self.q_table[self.last_state_index, self.last_action_index] += self.learning_rate * td_error

    def finish_episode(self, reward_history: Sequence[float]) -> None:
        self.episode_index += 1
        self.epsilon = max(self.minimum_epsilon, self.epsilon_init * np.exp(-self.epsilon_decay * self.episode_index))

        if len(reward_history) >= 5:
            rolling_reward = float(np.mean(reward_history[-5:]))

            if rolling_reward > self.best_rolling_reward + self.adaptive_min_improvement:
                self.best_rolling_reward = rolling_reward
                self.episodes_since_improvement = 0
            else:
                self.episodes_since_improvement += 1

            if self.episodes_since_improvement >= self.adaptive_patience:
                self.epsilon = min(0.35, self.epsilon + self.adaptive_epsilon_boost)
                self.episodes_since_improvement = 0


@dataclass
class ExperimentResult:
    policy: str
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


def make_env(schema_path: Path, building_name: str, random_seed: int) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=False,
        buildings=[building_name],
        active_observations=ACTIVE_OBSERVATIONS,
        active_actions=ACTIVE_ACTIONS,
        reward_function=WeatherOccupancyReward,
        random_seed=random_seed,
    )


def train_q_learning(agent: OwnAdaptiveTabularQLearning, env: CityLearnEnv, episodes: int, progress_every: int) -> TrainingTrace:
    episode_rewards = []
    epsilons = []
    training_start = time.perf_counter()

    for episode in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        terminated = False
        cumulative_reward = 0.0

        while not terminated:
            actions = agent.predict(observations, deterministic=False)
            next_observations, rewards, terminated, _, _ = env.step(actions)
            agent.update(float(rewards[0]), next_observations, terminated)
            observations = next_observations
            cumulative_reward += float(np.sum(rewards))

        episode_rewards.append(cumulative_reward)
        agent.finish_episode(episode_rewards)
        epsilons.append(agent.epsilon)

        if progress_every > 0 and ((episode + 1) % progress_every == 0 or episode == 0 or episode + 1 == episodes):
            elapsed = time.perf_counter() - training_start
            rolling = float(np.mean(episode_rewards[-5:])) if episode_rewards else cumulative_reward
            print(
                f'  Episode {episode + 1}/{episodes} | reward={cumulative_reward:.2f} | rolling5={rolling:.2f} | '
                f'epsilon={agent.epsilon:.3f} | elapsed={elapsed:.1f}s',
                flush=True,
            )

    training_seconds = time.perf_counter() - training_start
    return TrainingTrace(
        episode_rewards=episode_rewards,
        epsilons=epsilons,
        training_seconds=training_seconds,
        stability_episode=estimate_stability_episode(episode_rewards),
    )


def run_policy(agent, env: CityLearnEnv, deterministic: bool = True) -> PolicyRun:
    observations, _ = env.reset()
    agent.reset()
    terminated = False
    cumulative_reward = 0.0
    reward_trace = []

    while not terminated:
        action = agent.predict(observations, deterministic=deterministic)
        observations, rewards, terminated, _, _ = env.step(action)
        step_reward = float(np.sum(rewards))
        reward_trace.append(step_reward)
        cumulative_reward += step_reward

    base_env = env.unwrapped
    building = base_env.buildings[0]
    net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)
    kpis = base_env.evaluate()
    discomfort_proportion = float(
        kpis[(kpis['name'] == building.name) & (kpis['cost_function'] == 'discomfort_proportion')]['value'].iloc[0]
    )
    trajectory = pd.DataFrame({
        'time_step': np.arange(len(reward_trace)),
        'reward': reward_trace,
        'cumulative_reward': np.cumsum(reward_trace),
        'grid_import_kwh': np.clip(net_consumption[:len(reward_trace)], 0.0, None),
        'cumulative_grid_import_kwh': np.cumsum(np.clip(net_consumption[:len(reward_trace)], 0.0, None)),
    })

    result = ExperimentResult(
        policy=agent.__class__.__name__,
        total_grid_import_kwh=float(np.sum(np.clip(net_consumption, 0.0, None))),
        total_net_consumption_kwh=float(np.sum(net_consumption)),
        discomfort_proportion=discomfort_proportion,
        cumulative_reward=cumulative_reward,
    )

    return PolicyRun(result=result, trajectory=trajectory, kpis=kpis)


def estimate_stability_episode(rewards: Sequence[float], window: int = 10, tolerance: float = 0.03) -> int | None:
    if len(rewards) < window * 2:
        return None

    reward_array = np.asarray(rewards, dtype=float)

    for index in range((window * 2) - 1, len(reward_array)):
        previous_window = reward_array[index - (2 * window) + 1:index - window + 1]
        current_window = reward_array[index - window + 1:index + 1]
        previous_mean = float(np.mean(previous_window))
        current_mean = float(np.mean(current_window))
        scale = max(1.0, abs(previous_mean))
        relative_change = abs(current_mean - previous_mean) / scale
        coefficient_of_variation = float(np.std(current_window)) / max(1.0, abs(current_mean))

        if relative_change <= tolerance and coefficient_of_variation <= tolerance * 1.5:
            return index + 1

    return None


def build_results_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'policy': result.policy,
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


def save_single_comparison_figure(results_frame: pd.DataFrame, output_path: Path) -> None:
    labels = results_frame['policy'].tolist()
    x = np.arange(len(labels))
    colors = ['#9aa0a6' if policy.startswith('FixedCooling') else '#1d3557' for policy in results_frame['policy']]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].bar(x, results_frame['grid_import_kwh'], color=colors)
    axes[0].set_title('Total grid import')
    axes[0].set_ylabel('kWh')
    axes[0].set_xticks(x, labels, rotation=20, ha='right')
    axes[0].grid(axis='y', alpha=0.25)

    axes[1].bar(x, results_frame['savings_vs_fixed_pct'].fillna(0.0), color=colors)
    axes[1].set_title('Savings vs fixed strategy')
    axes[1].set_ylabel('%')
    axes[1].set_xticks(x, labels, rotation=20, ha='right')
    axes[1].grid(axis='y', alpha=0.25)

    stability_values = results_frame['stability_episode'].fillna(0.0)
    axes[2].bar(x, stability_values, color=colors)
    axes[2].set_title('Episode of estimated stability')
    axes[2].set_ylabel('Episode')
    axes[2].set_xticks(x, labels, rotation=20, ha='right')
    axes[2].grid(axis='y', alpha=0.25)

    fig.suptitle('Weather+occupancy reward: fixed strategy vs Q-learning', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_time_and_learning_comparison(
    fixed_run: PolicyRun,
    learned_run: PolicyRun,
    training_trace: TrainingTrace,
    output_path: Path,
    horizon: int,
) -> None:
    max_horizon = min(len(fixed_run.trajectory), len(learned_run.trajectory))
    if horizon <= 0:
        horizon = max_horizon
    else:
        horizon = min(horizon, max_horizon)

    fixed_slice = fixed_run.trajectory.head(horizon)
    learned_slice = learned_run.trajectory.head(horizon)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(fixed_slice['time_step'], fixed_slice['reward'], linestyle='--', linewidth=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[0, 0].plot(learned_slice['time_step'], learned_slice['reward'], linewidth=1.9, color='#1d3557', label='Q-learning')

    axes[0, 1].plot(fixed_slice['time_step'], fixed_slice['cumulative_reward'], linestyle='--', linewidth=1.6, color='#9aa0a6')
    axes[0, 1].plot(learned_slice['time_step'], learned_slice['cumulative_reward'], linewidth=1.9, color='#1d3557')

    episodes = np.arange(1, len(training_trace.episode_rewards) + 1)
    rolling = pd.Series(np.asarray(training_trace.episode_rewards, dtype=float)).rolling(
        min(10, len(training_trace.episode_rewards)), min_periods=1
    ).mean()
    axes[1, 0].plot(episodes, rolling, linewidth=2.0, color='#2a9d8f', label='Q-learning rolling reward')

    axes[1, 1].plot(
        fixed_slice['time_step'],
        fixed_slice['cumulative_grid_import_kwh'],
        linestyle='--',
        linewidth=1.6,
        color='#9aa0a6',
    )
    axes[1, 1].plot(
        learned_slice['time_step'],
        learned_slice['cumulative_grid_import_kwh'],
        linewidth=1.9,
        color='#1d3557',
    )

    axes[0, 0].set_title('Step reward in time')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc='best', fontsize=8)

    axes[0, 1].set_title('Cumulative reward in time')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Cumulative reward')
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].set_title('Learning progress (rolling episode reward)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Rolling episode reward')
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(loc='best', fontsize=8)

    axes[1, 1].set_title('Cumulative grid import in time')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle('Weather+occupancy reward: time behavior and learning comparison', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(
    schema_path: Path,
    building_name: str,
    episodes: int,
    baseline_action: float,
    random_seed: int,
    output_dir: Path,
    comparison_horizon: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    print('Evaluating fixed strategy...', flush=True)
    fixed_env = make_env(schema_path, building_name, random_seed)
    fixed_run = run_policy(FixedCoolingPolicy(action_value=baseline_action), fixed_env, deterministic=True)
    fixed_run.result.policy = f'FixedCooling({baseline_action:.2f})'
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    print('Training Q-learning...', flush=True)
    training_env = make_env(schema_path, building_name, random_seed)
    agent = OwnAdaptiveTabularQLearning(
        training_env,
        observation_bin_sizes=OBSERVATION_BIN_SIZES,
        action_bin_count=ACTION_BIN_COUNT,
        epsilon=1.0,
        minimum_epsilon=0.05,
        epsilon_decay=0.03,
        learning_rate=0.15,
        discount_factor=0.95,
        adaptive_patience=8,
        adaptive_epsilon_boost=0.08,
        adaptive_min_improvement=0.01,
        random_seed=random_seed,
    )
    training_trace = train_q_learning(agent, training_env, episodes=episodes, progress_every=max(1, episodes // 20))
    learned_run = run_policy(agent, training_env, deterministic=True)
    learned_run.result.policy = 'OwnAdaptiveTabularQLearning'
    learned_run.result.training_seconds = training_trace.training_seconds
    learned_run.result.stability_episode = training_trace.stability_episode
    learned_run.result.last_10_episode_reward_mean = (
        float(np.mean(training_trace.episode_rewards[-10:])) if training_trace.episode_rewards else None
    )

    if fixed_run.result.total_grid_import_kwh > 0.0:
        learned_run.result.savings_vs_fixed_pct = 100.0 * (
            fixed_run.result.total_grid_import_kwh - learned_run.result.total_grid_import_kwh
        ) / fixed_run.result.total_grid_import_kwh

    results.append(learned_run.result)

    results_frame = build_results_frame(results).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)
    save_single_comparison_figure(results_frame, output_dir / 'reward_vs_fixed_comparison.png')
    save_time_and_learning_comparison(
        fixed_run,
        learned_run,
        training_trace,
        output_dir / 'reward_time_and_learning_comparison.png',
        comparison_horizon,
    )

    np.save(output_dir / 'q_table.npy', agent.q_table)
    pd.DataFrame(
        {
            'episode': np.arange(1, len(training_trace.episode_rewards) + 1),
            'episode_reward': training_trace.episode_rewards,
            'epsilon': training_trace.epsilons,
        }
    ).to_csv(output_dir / 'learning_trace.csv', index=False)

    fixed_run.trajectory.to_csv(output_dir / 'trajectory_fixed_strategy.csv', index=False)
    learned_run.trajectory.to_csv(output_dir / 'trajectory_q_learning.csv', index=False)

    return results_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Q-learning with weather+occupancy reward, compared to a fixed cooling strategy.'
    )
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA, help='Path to the CityLearn schema JSON file.')
    parser.add_argument('--building', type=str, default=DEFAULT_BUILDING, help='Building name to control.')
    parser.add_argument('--episodes', type=int, default=400, help='Number of training episodes.')
    parser.add_argument('--baseline-action', type=float, default=0.5, help='Fixed cooling action used as the reference strategy.')
    parser.add_argument('--random-seed', type=int, default=7, help='Random seed for reproducibility.')
    parser.add_argument(
        '--comparison-horizon',
        type=int,
        default=8760,
        help='Number of initial time steps shown in the time comparison figure. Use 0 for full trajectory.',
    )
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory where PNG and CSV outputs will be saved.')
    args = parser.parse_args()

    print(f'Loading schema: {args.schema}', flush=True)
    print(f'Building: {args.building}', flush=True)
    print(f'State variables: {", ".join(ACTIVE_OBSERVATIONS)}', flush=True)
    print(f'Fixed strategy action: {args.baseline_action}', flush=True)
    print('Reward function: weather_occupancy (combined weather + occupancy weighting)', flush=True)
    print(f'Training episodes: {args.episodes}', flush=True)
    print(f'Comparison horizon: {args.comparison_horizon}', flush=True)
    print(f'Output directory: {args.output_dir}', flush=True)

    results = run_experiment(
        schema_path=args.schema,
        building_name=args.building,
        episodes=args.episodes,
        baseline_action=args.baseline_action,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        comparison_horizon=args.comparison_horizon,
    )

    print(results.to_string(index=False))
    print('Saved files:', flush=True)
    print(f'  {args.output_dir / "summary_results.csv"}', flush=True)
    print(f'  {args.output_dir / "learning_trace.csv"}', flush=True)
    print(f'  {args.output_dir / "q_table.npy"}', flush=True)
    print(f'  {args.output_dir / "trajectory_fixed_strategy.csv"}', flush=True)
    print(f'  {args.output_dir / "trajectory_q_learning.csv"}', flush=True)
    print(f'  {args.output_dir / "reward_vs_fixed_comparison.png"}', flush=True)
    print(f'  {args.output_dir / "reward_time_and_learning_comparison.png"}', flush=True)


if __name__ == '__main__':
    main()
