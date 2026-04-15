from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Type

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning1504_04'
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


class EnergyOnlyReward(RewardFunction):
    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = [-max(o['net_electricity_consumption'], 0.0) for o in observations]
        return [sum(reward_list)] if self.central_agent else reward_list


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
            hot_forecast = np.mean([
                observation.get('outdoor_dry_bulb_temperature_predicted_1', 0.0),
                observation.get('outdoor_dry_bulb_temperature_predicted_2', 0.0),
                observation.get('outdoor_dry_bulb_temperature_predicted_3', 0.0),
            ])
            occupancy_factor = 1.0 + self.occupancy_weight * min(occupant_count / 5.0, 1.0)
            weather_factor = 1.0 + self.hot_weather_weight * max(hot_forecast - 24.0, 0.0)
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
        self.env = env
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
    reward_name: str
    total_grid_import_kwh: float
    total_net_consumption_kwh: float
    discomfort_proportion: float
    cumulative_reward: float
    savings_vs_best_fixed_pct: float | None = None
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


def make_env(schema_path: Path, building_name: str, reward_function: Type[RewardFunction], random_seed: int) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=False,
        buildings=[building_name],
        active_observations=ACTIVE_OBSERVATIONS,
        active_actions=ACTIVE_ACTIONS,
        reward_function=reward_function,
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
    action_trace = []
    reward_trace = []
    occupancy_trace = []
    weather_1_trace = []
    weather_2_trace = []
    weather_3_trace = []

    while not terminated:
        action = agent.predict(observations, deterministic=deterministic)
        action_trace.append(float(action[0][0]))
        occupancy_trace.append(float(observations[0][ACTIVE_OBSERVATIONS.index('occupant_count')]))
        weather_1_trace.append(float(observations[0][ACTIVE_OBSERVATIONS.index('outdoor_dry_bulb_temperature_predicted_1')]))
        weather_2_trace.append(float(observations[0][ACTIVE_OBSERVATIONS.index('outdoor_dry_bulb_temperature_predicted_2')]))
        weather_3_trace.append(float(observations[0][ACTIVE_OBSERVATIONS.index('outdoor_dry_bulb_temperature_predicted_3')]))
        observations, rewards, terminated, _, _ = env.step(action)
        reward_trace.append(float(np.sum(rewards)))
        cumulative_reward += float(np.sum(rewards))

    base_env = env.unwrapped
    building = base_env.buildings[0]
    net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)
    kpis = base_env.evaluate()
    discomfort_proportion = float(
        kpis[(kpis['name'] == building.name) & (kpis['cost_function'] == 'discomfort_proportion')]['value'].iloc[0]
    )

    length = len(action_trace)
    trajectory = pd.DataFrame({
        'time_step': np.arange(length),
        'occupant_count': occupancy_trace,
        'weather_predicted_1': weather_1_trace,
        'weather_predicted_2': weather_2_trace,
        'weather_predicted_3': weather_3_trace,
        'action_value': action_trace,
        'reward': reward_trace,
        'net_consumption_kwh': net_consumption[:length],
        'grid_import_kwh': np.clip(net_consumption[:length], 0.0, None),
        'cumulative_grid_import_kwh': np.cumsum(np.clip(net_consumption[:length], 0.0, None)),
    })

    result = ExperimentResult(
        policy=agent.__class__.__name__,
        reward_name=base_env.reward_function.__class__.__name__,
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


def build_results_frame(results: Iterable[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'policy': result.policy,
            'reward': result.reward_name,
            'grid_import_kwh': round(result.total_grid_import_kwh, 3),
            'net_consumption_kwh': round(result.total_net_consumption_kwh, 3),
            'discomfort_proportion': round(result.discomfort_proportion, 4),
            'cumulative_reward': round(result.cumulative_reward, 3),
            'savings_vs_best_fixed_pct': None if result.savings_vs_best_fixed_pct is None else round(result.savings_vs_best_fixed_pct, 3),
            'training_seconds': None if result.training_seconds is None else round(result.training_seconds, 2),
            'stability_episode': result.stability_episode,
            'last_10_episode_reward_mean': None if result.last_10_episode_reward_mean is None else round(result.last_10_episode_reward_mean, 3),
        }
        for result in results
    ])


def save_learning_dashboard(reward_name: str, training_trace: TrainingTrace, output_path: Path) -> None:
    rewards = np.asarray(training_trace.episode_rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)
    rolling_window = min(10, len(rewards))
    rolling_mean = pd.Series(rewards).rolling(rolling_window, min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(episodes, rewards, color='#8d99ae', alpha=0.45, linewidth=1.2, label='Episode reward')
    axes[0].plot(episodes, rolling_mean, color='#1d3557', linewidth=2.2, label=f'Rolling mean ({rolling_window})')
    if training_trace.stability_episode is not None:
        axes[0].axvline(training_trace.stability_episode, color='#2a9d8f', linestyle='--', linewidth=1.5, label='Estimated stability')
    axes[0].set_title(f'Learning curve: {reward_name}')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Cumulative reward')
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc='best')

    axes[1].plot(episodes, training_trace.epsilons, color='#e63946', linewidth=2.0, label='Adaptive epsilon')
    axes[1].set_title('Exploration schedule')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Epsilon')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_strategy_dashboard(reward_name: str, fixed_runs: Sequence[PolicyRun], learned_run: PolicyRun, output_path: Path, horizon: int) -> None:
    learned_slice = learned_run.trajectory.head(horizon)
    steps = learned_slice['time_step'].to_numpy()
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)

    for fixed_run in fixed_runs:
        fixed_slice = fixed_run.trajectory.head(horizon)
        axes[0].plot(steps, fixed_slice['action_value'], linewidth=1.1, alpha=0.7, label=fixed_run.result.policy)
    axes[0].plot(steps, learned_slice['action_value'], color='#1d3557', linewidth=2.1, label='Q-learning')
    axes[0].set_ylabel('Cooling action')
    axes[0].set_title(f'Weather + occupancy strategy dashboard: {reward_name}')
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc='best', ncol=2)

    axes[1].plot(steps, learned_slice['weather_predicted_1'], color='#457b9d', linewidth=2.0, label='Weather +1h')
    axes[1].plot(steps, learned_slice['weather_predicted_2'], color='#1d3557', linewidth=1.6, label='Weather +2h')
    axes[1].plot(steps, learned_slice['weather_predicted_3'], color='#6d597a', linewidth=1.4, label='Weather +3h')
    axes[1].set_ylabel('Forecast temp [C]')
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc='best')

    axes[2].fill_between(steps, 0.0, learned_slice['occupant_count'], color='#f4a261', alpha=0.35, label='Occupancy count')
    axes[2].set_ylabel('Occupancy')
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc='best')

    for fixed_run in fixed_runs:
        fixed_slice = fixed_run.trajectory.head(horizon)
        axes[3].plot(steps, fixed_slice['cumulative_grid_import_kwh'], linewidth=1.1, alpha=0.7, label=fixed_run.result.policy)
    axes[3].plot(steps, learned_slice['cumulative_grid_import_kwh'], color='#264653', linewidth=2.3, label='Q-learning')
    axes[3].set_ylabel('Cumulative import')
    axes[3].set_xlabel('Time step')
    axes[3].grid(alpha=0.25)
    axes[3].legend(loc='best', ncol=2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(
    schema_path: Path,
    building_name: str,
    episodes: int,
    baseline_actions: Sequence[float],
    random_seed: int,
    reward_classes: Sequence[Type[RewardFunction]],
    output_dir: Path,
    comparison_horizon: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []
    baseline_runs: list[PolicyRun] = []

    print('Evaluating fixed strategies...', flush=True)
    for baseline_action in baseline_actions:
        baseline_env = make_env(schema_path, building_name, reward_classes[0], random_seed)
        baseline_run = run_policy(FixedCoolingPolicy(action_value=baseline_action), baseline_env, deterministic=True)
        baseline_run.result.policy = f'FixedCooling({baseline_action:.2f})'
        baseline_run.result.reward_name = 'ReferencePolicy'
        baseline_run.result.savings_vs_best_fixed_pct = 0.0
        baseline_runs.append(baseline_run)
        results.append(baseline_run.result)
        baseline_run.trajectory.to_csv(output_dir / f'trajectory_fixed_{baseline_action:.2f}.csv', index=False)

    best_fixed_run = min(baseline_runs, key=lambda item: item.result.total_grid_import_kwh)

    for reward_class in reward_classes:
        print(f'Training weather-occupancy Q-learning with {reward_class.__name__}...', flush=True)
        training_env = make_env(schema_path, building_name, reward_class, random_seed)
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
        learned_run.result.last_10_episode_reward_mean = float(np.mean(training_trace.episode_rewards[-10:])) if training_trace.episode_rewards else None

        if best_fixed_run.result.total_grid_import_kwh > 0.0:
            learned_run.result.savings_vs_best_fixed_pct = 100.0 * (
                best_fixed_run.result.total_grid_import_kwh - learned_run.result.total_grid_import_kwh
            ) / best_fixed_run.result.total_grid_import_kwh

        results.append(learned_run.result)
        reward_slug = reward_class.__name__.replace('Reward', '').lower()
        pd.DataFrame({
            'episode': np.arange(1, len(training_trace.episode_rewards) + 1),
            'episode_reward': training_trace.episode_rewards,
            'epsilon': training_trace.epsilons,
        }).to_csv(output_dir / f'learning_trace_{reward_slug}.csv', index=False)
        learned_run.trajectory.to_csv(output_dir / f'trajectory_q_learning_{reward_slug}.csv', index=False)
        save_learning_dashboard(reward_class.__name__, training_trace, output_dir / f'learning_dashboard_{reward_slug}.png')
        save_strategy_dashboard(reward_class.__name__, baseline_runs, learned_run, output_dir / f'strategy_dashboard_{reward_slug}.png', comparison_horizon)

    results_frame = build_results_frame(results).sort_values(['policy', 'reward']).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)
    return results_frame


def parse_reward_names(reward_names: Sequence[str]) -> list[Type[RewardFunction]]:
    reward_map: dict[str, Type[RewardFunction]] = {
        'energy_only': EnergyOnlyReward,
        'weather_occupancy': WeatherOccupancyReward,
    }
    selected = []

    for reward_name in reward_names:
        key = reward_name.strip().lower()
        if key not in reward_map:
            raise ValueError(f'Unknown reward configuration: {reward_name}')
        selected.append(reward_map[key])

    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Adaptive Q-learning that uses only occupancy and weather forecasts as state variables.'
    )
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA, help='Path to the CityLearn schema JSON file.')
    parser.add_argument('--building', type=str, default=DEFAULT_BUILDING, help='Building name to control.')
    parser.add_argument('--episodes', type=int, default=80, help='Number of training episodes.')
    parser.add_argument('--baseline-actions', nargs='+', type=float, default=[0.5, 0.7, 0.9], help='Fixed baseline cooling actions to compare against.')
    parser.add_argument('--random-seed', type=int, default=7, help='Random seed for reproducibility.')
    parser.add_argument('--comparison-horizon', type=int, default=168, help='Number of initial time steps shown in the strategy dashboard.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory where PNG and CSV outputs will be saved.')
    parser.add_argument(
        '--reward-configs',
        nargs='+',
        default=['energy_only', 'weather_occupancy'],
        help='Reward configurations to evaluate. Supported: energy_only weather_occupancy.',
    )
    args = parser.parse_args()

    reward_classes = parse_reward_names(args.reward_configs)
    print(f'Loading schema: {args.schema}', flush=True)
    print(f'Building: {args.building}', flush=True)
    print(f'State variables: {", ".join(ACTIVE_OBSERVATIONS)}', flush=True)
    print(f'Reward configs: {", ".join(args.reward_configs)}', flush=True)
    print(f'Training episodes: {args.episodes}', flush=True)
    print(f'Fixed baselines: {args.baseline_actions}', flush=True)
    print(f'Output directory: {args.output_dir}', flush=True)

    results = run_experiment(
        schema_path=args.schema,
        building_name=args.building,
        episodes=args.episodes,
        baseline_actions=args.baseline_actions,
        random_seed=args.random_seed,
        reward_classes=reward_classes,
        output_dir=args.output_dir,
        comparison_horizon=args.comparison_horizon,
    )

    print('This version uses only occupancy and weather forecasts in the state.')
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()