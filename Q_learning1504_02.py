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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning1504_02'
ACTIVE_OBSERVATIONS = [
    'hour',
    'outdoor_dry_bulb_temperature_predicted_1',
    'occupant_count',
    'indoor_dry_bulb_temperature_cooling_delta',
]
ACTIVE_ACTIONS = ['cooling_device']
OBSERVATION_BIN_SIZES = {
    'hour': 24,
    'outdoor_dry_bulb_temperature_predicted_1': 6,
    'occupant_count': 4,
    'indoor_dry_bulb_temperature_cooling_delta': 8,
}
ACTION_BIN_COUNT = 9


class EnergyOnlyReward(RewardFunction):
    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = [-max(o['net_electricity_consumption'], 0.0) for o in observations]
        return [sum(reward_list)] if self.central_agent else reward_list


class OccupancyAwareReward(RewardFunction):
    def __init__(self, env_metadata, comfort_coefficient: float = 6.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.comfort_coefficient = float(comfort_coefficient)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []

        for observation in observations:
            net_import = max(observation['net_electricity_consumption'], 0.0)
            occupied = 1.0 if observation.get('occupant_count', 0.0) > 0.0 else 0.0
            overheating = max(observation.get('indoor_dry_bulb_temperature_cooling_delta', 0.0), 0.0)
            reward = -(net_import + self.comfort_coefficient * occupied * (overheating ** 2))
            reward_list.append(reward)

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

        for name, low, high, count in zip(
            self.observation_names,
            env.observation_space[0].low,
            env.observation_space[0].high,
            self.bin_counts,
        ):
            low_value = float(low)
            high_value = float(high)

            if name == 'hour':
                low_value, high_value = 1.0, 24.0

            self.edges.append(np.linspace(low_value, high_value, count + 1)[1:-1])

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


class OwnTabularQLearning:
    def __init__(
        self,
        env: CityLearnEnv,
        observation_bin_sizes: dict[str, int],
        action_bin_count: int,
        learning_rate: float = 0.15,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        minimum_epsilon: float = 0.05,
        epsilon_decay: float = 0.08,
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
        self.random_state = np.random.RandomState(random_seed)
        self.q_table = np.zeros(
            (self.observation_discretizer.state_count, self.action_discretizer.action_count),
            dtype=np.float32,
        )
        self.episode_index = 0
        self.last_state_index: int | None = None
        self.last_action_index: int | None = None

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

    def finish_episode(self) -> None:
        self.episode_index += 1
        self.epsilon = max(
            self.minimum_epsilon,
            self.epsilon_init * np.exp(-self.epsilon_decay * self.episode_index),
        )


@dataclass
class ExperimentResult:
    policy: str
    reward_name: str
    total_grid_import_kwh: float
    total_net_consumption_kwh: float
    discomfort_proportion: float
    occupied_overheat_mean_degC: float
    occupied_overheat_hours: float
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


def make_env(
    schema_path: Path,
    building_name: str,
    reward_function: Type[RewardFunction],
    random_seed: int,
) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=False,
        buildings=[building_name],
        active_observations=ACTIVE_OBSERVATIONS,
        active_actions=ACTIVE_ACTIONS,
        reward_function=reward_function,
        random_seed=random_seed,
    )


def train_q_learning(
    agent: OwnTabularQLearning,
    env: CityLearnEnv,
    episodes: int,
    progress_every: int,
) -> TrainingTrace:
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

        agent.finish_episode()
        episode_rewards.append(cumulative_reward)
        epsilons.append(agent.epsilon)

        should_report = progress_every > 0 and ((episode + 1) % progress_every == 0 or episode == 0 or episode + 1 == episodes)

        if should_report:
            elapsed = time.perf_counter() - training_start
            print(
                f'  Episode {episode + 1}/{episodes} | reward={cumulative_reward:.2f} | '
                f'epsilon={agent.epsilon:.3f} | elapsed={elapsed:.1f}s',
                flush=True,
            )

    training_seconds = time.perf_counter() - training_start
    stability_episode = estimate_stability_episode(episode_rewards)
    return TrainingTrace(
        episode_rewards=episode_rewards,
        epsilons=epsilons,
        training_seconds=training_seconds,
        stability_episode=stability_episode,
    )


def run_policy(agent, env: CityLearnEnv, deterministic: bool = True) -> PolicyRun:
    observations, _ = env.reset()
    agent.reset()
    terminated = False
    cumulative_reward = 0.0
    action_trace = []
    reward_trace = []
    hour_trace = []
    occupant_trace = []
    cooling_delta_trace = []

    while not terminated:
        action = agent.predict(observations, deterministic=deterministic)
        action_trace.append(float(action[0][0]))
        hour_trace.append(float(observations[0][ACTIVE_OBSERVATIONS.index('hour')]))
        occupant_trace.append(float(observations[0][ACTIVE_OBSERVATIONS.index('occupant_count')]))
        cooling_delta_trace.append(float(observations[0][ACTIVE_OBSERVATIONS.index('indoor_dry_bulb_temperature_cooling_delta')]))
        observations, rewards, terminated, _, _ = env.step(action)
        reward_trace.append(float(np.sum(rewards)))
        cumulative_reward += float(np.sum(rewards))

    base_env = env.unwrapped
    building = base_env.buildings[0]
    net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)
    indoor_temperature = np.asarray(building.indoor_dry_bulb_temperature, dtype=float)
    cooling_setpoint = np.asarray(building.indoor_dry_bulb_temperature_cooling_set_point, dtype=float)
    occupant_count = np.asarray(building.occupant_count, dtype=float)
    occupied_mask = occupant_count > 0.0
    overheating = np.clip(indoor_temperature - cooling_setpoint, 0.0, None)
    occupied_overheat = overheating[occupied_mask]
    kpis = base_env.evaluate()
    discomfort_proportion = float(
        kpis[
            (kpis['name'] == building.name)
            & (kpis['cost_function'] == 'discomfort_proportion')
        ]['value'].iloc[0]
    )

    length = len(action_trace)
    trajectory = pd.DataFrame({
        'time_step': np.arange(length),
        'hour': hour_trace,
        'occupant_count': occupant_trace,
        'cooling_delta_before_action': cooling_delta_trace,
        'action_value': action_trace,
        'reward': reward_trace,
        'net_consumption_kwh': net_consumption[:length],
        'grid_import_kwh': np.clip(net_consumption[:length], 0.0, None),
        'indoor_temperature': indoor_temperature[:length],
        'cooling_setpoint': cooling_setpoint[:length],
        'cooling_delta_after_step': overheating[:length],
    })

    result = ExperimentResult(
        policy=agent.__class__.__name__,
        reward_name=base_env.reward_function.__class__.__name__,
        total_grid_import_kwh=float(np.sum(np.clip(net_consumption, 0.0, None))),
        total_net_consumption_kwh=float(np.sum(net_consumption)),
        discomfort_proportion=discomfort_proportion,
        occupied_overheat_mean_degC=float(occupied_overheat.mean()) if occupied_overheat.size else 0.0,
        occupied_overheat_hours=float(np.sum(occupied_overheat > 0.0)),
        cumulative_reward=cumulative_reward,
    )

    return PolicyRun(result=result, trajectory=trajectory, kpis=kpis)


def estimate_stability_episode(rewards: Sequence[float], window: int = 8, tolerance: float = 0.02) -> int | None:
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
            'occupied_overheat_mean_degC': round(result.occupied_overheat_mean_degC, 4),
            'occupied_overheat_hours': round(result.occupied_overheat_hours, 3),
            'cumulative_reward': round(result.cumulative_reward, 3),
            'savings_vs_fixed_pct': None if result.savings_vs_fixed_pct is None else round(result.savings_vs_fixed_pct, 3),
            'training_seconds': None if result.training_seconds is None else round(result.training_seconds, 2),
            'stability_episode': result.stability_episode,
            'last_10_episode_reward_mean': None if result.last_10_episode_reward_mean is None else round(result.last_10_episode_reward_mean, 3),
        }
        for result in results
    ])


def save_learning_curve_plot(
    reward_name: str,
    training_trace: TrainingTrace,
    output_path: Path,
) -> None:
    rewards = np.asarray(training_trace.episode_rewards, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)
    rolling_window = min(5, len(rewards))
    rolling_mean = pd.Series(rewards).rolling(rolling_window, min_periods=1).mean()

    fig, ax_reward = plt.subplots(figsize=(11, 5.5))
    ax_reward.plot(episodes, rewards, color='#9d4edd', alpha=0.45, linewidth=1.5, label='Episode reward')
    ax_reward.plot(episodes, rolling_mean, color='#1d3557', linewidth=2.5, label=f'Rolling mean ({rolling_window})')
    ax_reward.set_xlabel('Episode')
    ax_reward.set_ylabel('Cumulative reward')
    ax_reward.set_title(f'Learning curve: {reward_name}')
    ax_reward.grid(alpha=0.25)

    if training_trace.stability_episode is not None:
        ax_reward.axvline(
            training_trace.stability_episode,
            color='#2a9d8f',
            linestyle='--',
            linewidth=1.5,
            label=f'Stability episode {training_trace.stability_episode}',
        )

    ax_epsilon = ax_reward.twinx()
    ax_epsilon.plot(episodes, training_trace.epsilons, color='#e76f51', linewidth=1.8, label='Epsilon')
    ax_epsilon.set_ylabel('Exploration rate')
    ax_epsilon.set_ylim(0.0, 1.05)

    lines_left, labels_left = ax_reward.get_legend_handles_labels()
    lines_right, labels_right = ax_epsilon.get_legend_handles_labels()
    ax_reward.legend(lines_left + lines_right, labels_left + labels_right, loc='best')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_strategy_comparison_plot(
    reward_name: str,
    fixed_run: PolicyRun,
    learned_run: PolicyRun,
    output_path: Path,
    horizon: int,
) -> None:
    fixed_slice = fixed_run.trajectory.head(horizon)
    learned_slice = learned_run.trajectory.head(horizon)
    steps = learned_slice['time_step'].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(steps, fixed_slice['action_value'], color='#6c757d', linewidth=2.0, label='Fixed action')
    axes[0].plot(steps, learned_slice['action_value'], color='#1d3557', linewidth=2.0, label='Q-learning action')
    axes[0].set_ylabel('Cooling action')
    axes[0].set_title(f'Strategy comparison: {reward_name} (first {len(learned_slice)} steps)')
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc='best')

    axes[1].plot(steps, fixed_slice['grid_import_kwh'], color='#6c757d', linewidth=1.8, label='Fixed grid import')
    axes[1].plot(steps, learned_slice['grid_import_kwh'], color='#e63946', linewidth=1.8, label='Q-learning grid import')
    axes[1].set_ylabel('Grid import [kWh]')
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc='best')

    axes[2].plot(steps, fixed_slice['cooling_delta_after_step'], color='#6c757d', linewidth=1.8, label='Fixed overheating')
    axes[2].plot(steps, learned_slice['cooling_delta_after_step'], color='#2a9d8f', linewidth=1.8, label='Q-learning overheating')
    axes[2].fill_between(
        steps,
        0.0,
        learned_slice['occupant_count'].to_numpy(),
        color='#ffb703',
        alpha=0.15,
        label='Occupancy level',
    )
    axes[2].set_ylabel('Cooling delta / occupancy')
    axes[2].set_xlabel('Time step')
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc='best')

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_experiment(
    schema_path: Path,
    building_name: str,
    episodes: int,
    baseline_action: float,
    random_seed: int,
    reward_classes: Sequence[Type[RewardFunction]],
    output_dir: Path,
    comparison_horizon: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    print('Evaluating fixed strategy...', flush=True)
    baseline_env = make_env(
        schema_path=schema_path,
        building_name=building_name,
        reward_function=reward_classes[0],
        random_seed=random_seed,
    )
    baseline_policy = FixedCoolingPolicy(action_value=baseline_action)
    baseline_run = run_policy(baseline_policy, baseline_env, deterministic=True)
    baseline_run.result.policy = f'FixedCooling({baseline_action:.2f})'
    baseline_run.result.reward_name = 'ReferencePolicy'
    baseline_run.result.savings_vs_fixed_pct = 0.0
    results.append(baseline_run.result)
    baseline_run.trajectory.to_csv(output_dir / 'trajectory_fixed_strategy.csv', index=False)

    for reward_class in reward_classes:
        print(f'Training own Q-learning with {reward_class.__name__}...', flush=True)
        training_env = make_env(
            schema_path=schema_path,
            building_name=building_name,
            reward_function=reward_class,
            random_seed=random_seed,
        )
        agent = OwnTabularQLearning(
            training_env,
            observation_bin_sizes=OBSERVATION_BIN_SIZES,
            action_bin_count=ACTION_BIN_COUNT,
            epsilon=1.0,
            minimum_epsilon=0.05,
            epsilon_decay=0.08,
            learning_rate=0.15,
            discount_factor=0.95,
            random_seed=random_seed,
        )

        training_trace = train_q_learning(agent, training_env, episodes=episodes, progress_every=1)
        learned_run = run_policy(agent, training_env, deterministic=True)
        learned_run.result.policy = 'OwnTabularQLearning'
        learned_run.result.training_seconds = training_trace.training_seconds
        learned_run.result.stability_episode = training_trace.stability_episode
        learned_run.result.last_10_episode_reward_mean = float(np.mean(training_trace.episode_rewards[-10:])) if training_trace.episode_rewards else None

        if baseline_run.result.total_grid_import_kwh > 0.0:
            learned_run.result.savings_vs_fixed_pct = 100.0 * (
                baseline_run.result.total_grid_import_kwh - learned_run.result.total_grid_import_kwh
            ) / baseline_run.result.total_grid_import_kwh
        else:
            learned_run.result.savings_vs_fixed_pct = None

        results.append(learned_run.result)

        reward_slug = reward_class.__name__.replace('Reward', '').lower()
        pd.DataFrame({
            'episode': np.arange(1, len(training_trace.episode_rewards) + 1),
            'episode_reward': training_trace.episode_rewards,
            'epsilon': training_trace.epsilons,
        }).to_csv(output_dir / f'learning_trace_{reward_slug}.csv', index=False)
        learned_run.trajectory.to_csv(output_dir / f'trajectory_q_learning_{reward_slug}.csv', index=False)
        save_learning_curve_plot(
            reward_name=reward_class.__name__,
            training_trace=training_trace,
            output_path=output_dir / f'learning_curve_{reward_slug}.png',
        )
        save_strategy_comparison_plot(
            reward_name=reward_class.__name__,
            fixed_run=baseline_run,
            learned_run=learned_run,
            output_path=output_dir / f'strategy_comparison_{reward_slug}.png',
            horizon=comparison_horizon,
        )

    results_frame = build_results_frame(results).sort_values(['policy', 'reward']).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)
    return results_frame


def parse_reward_names(reward_names: Sequence[str]) -> list[Type[RewardFunction]]:
    reward_map: dict[str, Type[RewardFunction]] = {
        'energy_only': EnergyOnlyReward,
        'occupancy_aware': OccupancyAwareReward,
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
        description='Own tabular Q-learning experiment with saved learning and strategy plots.'
    )
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA, help='Path to the CityLearn schema JSON file.')
    parser.add_argument('--building', type=str, default=DEFAULT_BUILDING, help='Building name to control.')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes.')
    parser.add_argument('--baseline-action', type=float, default=0.70, help='Fixed cooling action used in the reference strategy.')
    parser.add_argument('--random-seed', type=int, default=7, help='Random seed for reproducibility.')
    parser.add_argument('--comparison-horizon', type=int, default=168, help='Number of initial time steps shown in the strategy comparison plot.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='Directory where PNG and CSV outputs will be saved.')
    parser.add_argument(
        '--reward-configs',
        nargs='+',
        default=['energy_only'],
        help='Reward configurations to evaluate. Supported: energy_only occupancy_aware.',
    )
    args = parser.parse_args()

    reward_classes = parse_reward_names(args.reward_configs)
    print(f'Loading schema: {args.schema}', flush=True)
    print(f'Building: {args.building}', flush=True)
    print(f'Reward configs: {", ".join(args.reward_configs)}', flush=True)
    print(f'Training episodes: {args.episodes}', flush=True)
    print(f'Output directory: {args.output_dir}', flush=True)

    results = run_experiment(
        schema_path=args.schema,
        building_name=args.building,
        episodes=args.episodes,
        baseline_action=args.baseline_action,
        random_seed=args.random_seed,
        reward_classes=reward_classes,
        output_dir=args.output_dir,
        comparison_horizon=args.comparison_horizon,
    )

    print('Fixed strategy: constant cooling_device action independent of weather and occupancy.')
    print('Q-learning state:', ', '.join(ACTIVE_OBSERVATIONS))
    print(results.to_string(index=False))
    print('Saved files:', flush=True)
    print(f'  {args.output_dir / "summary_results.csv"}', flush=True)

    for reward_name in args.reward_configs:
        reward_slug = reward_name.replace('_', '').lower().replace('energyonly', 'energyonly')
        if reward_name == 'energy_only':
            reward_slug = 'energyonly'
        elif reward_name == 'occupancy_aware':
            reward_slug = 'occupancyaware'
        print(f'  {args.output_dir / f"learning_curve_{reward_slug}.png"}', flush=True)
        print(f'  {args.output_dir / f"strategy_comparison_{reward_slug}.png"}', flush=True)


if __name__ == '__main__':
    main()