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
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1' / 'schema.json'
DEFAULT_BUILDING = 'Building_1'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning1604_03'

# ── Observations ─────────────────────────────────────────────────────────────
# SOC (state-of-charge) observácie sú kľúčové: agent musí vedieť koľko je
# nabitá batéria a zásobník teplej vody pred tým ako sa rozhodne akciu.
ACTIVE_OBSERVATIONS = [
    'outdoor_dry_bulb_temperature_predicted_1',
    'outdoor_dry_bulb_temperature_predicted_2',
    'outdoor_dry_bulb_temperature_predicted_3',
    'dhw_storage_soc',
    'electrical_storage_soc',
    'occupant_count',
]
OBSERVATION_BIN_SIZES = {
    'outdoor_dry_bulb_temperature_predicted_1': 5,
    'outdoor_dry_bulb_temperature_predicted_2': 4,
    'outdoor_dry_bulb_temperature_predicted_3': 4,
    'dhw_storage_soc': 3,     # nízky / stredný / vysoký
    'electrical_storage_soc': 3,
    'occupant_count': 4,
}
# Veľkosť stavového priestoru: 5×4×4×3×3×4 = 2880 stavov

# ── Akcie ────────────────────────────────────────────────────────────────────
# Poradie musí zodpovedať env.action_space: [dhw_storage, electrical_storage, cooling_device]
ACTIVE_ACTIONS = ['dhw_storage', 'electrical_storage', 'cooling_device']

# Počet binov pre každú akciu (v rovnakom poradí ako ACTIVE_ACTIONS).
# Zásobníky majú 3 biny: -1 (vybíjaj) / 0 (nič nerob) / +1 (nabíjaj)
# Chladenie má 5 binov: 0.0 / 0.25 / 0.5 / 0.75 / 1.0
# Celkový počet joint akcií: 3 × 3 × 5 = 45
ACTION_BIN_COUNTS = [3, 3, 5]

HOURS_PER_MONTH = [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744]
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ── Reward funkcia ────────────────────────────────────────────────────────────
class WeatherOccupancyReward(RewardFunction):
    """Penalizuje dovoz energie zo siete, pričom berie do úvahy obsadenosť
    budovy a predpoveď vonkajšej teploty."""

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


# ── Fixná referenčná politika ─────────────────────────────────────────────────
class FixedPolicy:
    """Vždy vracia rovnaké akcie pre všetky 3 dimenzie.
    Default: zásobníky nechávame na 0 (neutral), chladenie na fixnej hodnote."""

    def __init__(self, dhw_action: float = 0.0, electrical_action: float = 0.0, cooling_action: float = 0.5):
        self.actions = [float(dhw_action), float(electrical_action), float(cooling_action)]

    def reset(self) -> None:
        pass

    def predict(self, observations: list[list[float]], deterministic: bool = None) -> list[list[float]]:
        return [list(self.actions) for _ in observations]


# ── Diskretizácia stavového priestoru ─────────────────────────────────────────
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
        digits = [int(np.digitize(float(v), e, right=False)) for v, e in zip(observation, self.edges)]
        return int(np.ravel_multi_index(tuple(digits), self.state_shape))


# ── Diskretizácia akčného priestoru (multi-dimenzionálny) ─────────────────────
class MultiActionDiscretizer:
    """Vytvára mriežku všetkých kombinácií akcií (kartézsky súčin).

    Pre ACTION_BIN_COUNTS=[3,3,5] to dáva 3×3×5 = 45 joint akcií.
    Každá joint akcia je jeden riadok Q-tabuľky.
    """

    def __init__(self, env: CityLearnEnv, bin_counts_per_action: list[int]):
        n_dims = env.action_space[0].shape[0]
        if len(bin_counts_per_action) != n_dims:
            raise ValueError(f'Očakávalo sa {n_dims} bin counts, dostalo sa {len(bin_counts_per_action)}.')

        lows = env.action_space[0].low.tolist()
        highs = env.action_space[0].high.tolist()

        # Hodnoty pre každú dimenziu zvlášť
        self.value_grids = [
            np.linspace(float(lo), float(hi), int(n), dtype=float)
            for lo, hi, n in zip(lows, highs, bin_counts_per_action)
        ]

        # Všetky kombinácie (joint akcie)
        self.joint_actions: list[tuple[float, ...]] = list(iterproduct(*self.value_grids))

    @property
    def action_count(self) -> int:
        return len(self.joint_actions)

    def decode(self, action_index: int) -> list[list[float]]:
        """Vracia [[dhw_val, elec_val, cooling_val]] pre jednu budovu."""
        return [list(self.joint_actions[action_index])]


# ── Q-learning agent ──────────────────────────────────────────────────────────
class OwnAdaptiveTabularQLearning:
    def __init__(
        self,
        env: CityLearnEnv,
        observation_bin_sizes: dict[str, int],
        action_bin_counts: list[int],
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
            (self.observation_discretizer.state_count, self.action_discretizer.action_count),
            dtype=np.float32,
        )
        self.episode_index = 0
        self.last_state_index: int | None = None
        self.last_action_index: int | None = None
        self.best_rolling_reward = -np.inf
        self.episodes_since_improvement = 0

        n_states = self.observation_discretizer.state_count
        n_actions = self.action_discretizer.action_count
        print(
            f'  [Q-tabuľka] stavy={n_states}, joint_akcie={n_actions}, '
            f'veľkosť={n_states * n_actions:,} buniek '
            f'({n_states * n_actions * 4 / 1024 / 1024:.1f} MB)',
            flush=True,
        )

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


# ── Dátové triedy ──────────────────────────────────────────────────────────────
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


# ── Pomocné funkcie ───────────────────────────────────────────────────────────
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


def train_q_learning(
    agent: OwnAdaptiveTabularQLearning,
    env: CityLearnEnv,
    episodes: int,
    progress_every: int,
) -> TrainingTrace:
    episode_rewards: list[float] = []
    epsilons: list[float] = []
    training_start = time.perf_counter()

    for episode in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        terminated = False
        cumulative_reward = 0.0
        i = 0
        while not terminated:
            actions = agent.predict(observations, deterministic=False)
            next_observations, rewards, terminated, _, _ = env.step(actions)
            agent.update(float(rewards[0]), next_observations, terminated)
            observations = next_observations
            cumulative_reward += float(np.sum(rewards))
            i+=1

        print(f'Steps were : {i}', flush=True)
        episode_rewards.append(cumulative_reward)
        agent.finish_episode(episode_rewards)
        epsilons.append(agent.epsilon)

        if progress_every > 0 and ((episode + 1) % progress_every == 0 or episode == 0 or episode + 1 == episodes):
            elapsed = time.perf_counter() - training_start
            rolling = float(np.mean(episode_rewards[-5:])) if len(episode_rewards) >= 5 else cumulative_reward
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
    reward_trace: list[float] = []

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
        prev = reward_array[index - (2 * window) + 1:index - window + 1]
        curr = reward_array[index - window + 1:index + 1]
        prev_mean = float(np.mean(prev))
        curr_mean = float(np.mean(curr))
        scale = max(1.0, abs(prev_mean))
        if abs(curr_mean - prev_mean) / scale <= tolerance and float(np.std(curr)) / max(1.0, abs(curr_mean)) <= tolerance * 1.5:
            return index + 1
    return None


def build_results_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'policy': r.policy,
            'grid_import_kwh': round(r.total_grid_import_kwh, 3),
            'net_consumption_kwh': round(r.total_net_consumption_kwh, 3),
            'discomfort_proportion': round(r.discomfort_proportion, 4),
            'cumulative_reward': round(r.cumulative_reward, 3),
            'savings_vs_fixed_pct': None if r.savings_vs_fixed_pct is None else round(r.savings_vs_fixed_pct, 3),
            'training_seconds': None if r.training_seconds is None else round(r.training_seconds, 2),
            'stability_episode': r.stability_episode,
            'last_10_episode_reward_mean': None if r.last_10_episode_reward_mean is None else round(r.last_10_episode_reward_mean, 3),
        }
        for r in results
    ])


# ── Grafy ──────────────────────────────────────────────────────────────────────
def save_single_comparison_figure(results_frame: pd.DataFrame, output_path: Path) -> None:
    labels = results_frame['policy'].tolist()
    x = np.arange(len(labels))
    colors = ['#9aa0a6' if p.startswith('Fixed') else '#1d3557' for p in results_frame['policy']]

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

    axes[2].bar(x, results_frame['stability_episode'].fillna(0.0), color=colors)
    axes[2].set_title('Episode of estimated stability')
    axes[2].set_ylabel('Episode')
    axes[2].set_xticks(x, labels, rotation=20, ha='right')
    axes[2].grid(axis='y', alpha=0.25)

    fig.suptitle('3 akcie (DHW + batéria + chladenie): fixná vs Q-learning', fontsize=13)
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
    max_h = min(len(fixed_run.trajectory), len(learned_run.trajectory))
    horizon = max_h if horizon <= 0 else min(horizon, max_h)
    fixed_slice = fixed_run.trajectory.head(horizon)
    learned_slice = learned_run.trajectory.head(horizon)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(fixed_slice['time_step'], fixed_slice['reward'], linestyle='--', linewidth=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[0, 0].plot(learned_slice['time_step'], learned_slice['reward'], linewidth=1.9, color='#1d3557', label='Q-learning')
    axes[0, 0].set_title('Step reward in time')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(loc='best', fontsize=9)

    axes[0, 1].plot(fixed_slice['time_step'], fixed_slice['cumulative_reward'], linestyle='--', linewidth=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[0, 1].plot(learned_slice['time_step'], learned_slice['cumulative_reward'], linewidth=1.9, color='#1d3557', label='Q-learning')
    axes[0, 1].set_title('Cumulative reward in time')
    axes[0, 1].set_xlabel('Time step')
    axes[0, 1].set_ylabel('Cumulative reward')
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(loc='best', fontsize=9)

    episodes = np.arange(1, len(training_trace.episode_rewards) + 1)
    rolling = pd.Series(np.asarray(training_trace.episode_rewards, dtype=float)).rolling(
        min(10, len(training_trace.episode_rewards)), min_periods=1
    ).mean()
    axes[1, 0].plot(episodes, rolling, linewidth=2.0, color='#2a9d8f')
    axes[1, 0].set_title('Learning progress (rolling episode reward)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Rolling episode reward')
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(fixed_slice['time_step'], fixed_slice['cumulative_grid_import_kwh'], linestyle='--', linewidth=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[1, 1].plot(learned_slice['time_step'], learned_slice['cumulative_grid_import_kwh'], linewidth=1.9, color='#1d3557', label='Q-learning')
    axes[1, 1].set_title('Cumulative grid import in time')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc='best', fontsize=9)

    fig.suptitle('3 akcie: DHW zásobník + batéria + chladenie', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def split_month_ranges(horizon: int) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    start = 0
    for i, h in enumerate(HOURS_PER_MONTH):
        if start >= horizon:
            break
        end = min(start + h, horizon)
        ranges.append((i, start, end))
        start = end
    return ranges


def save_monthly_comparisons(fixed_run: PolicyRun, learned_run: PolicyRun, output_dir: Path) -> pd.DataFrame:
    max_h = min(len(fixed_run.trajectory), len(learned_run.trajectory))
    monthly_dir = output_dir / 'monthly_comparisons'
    monthly_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for mi, start, end in split_month_ranges(max_h):
        fixed_sl = fixed_run.trajectory.iloc[start:end]
        learned_sl = learned_run.trajectory.iloc[start:end]
        local_t = np.arange(end - start)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        axes[0].plot(local_t, np.cumsum(fixed_sl['reward'].to_numpy()), linestyle='--', linewidth=1.8, color='#9aa0a6', label='Fixed')
        axes[0].plot(local_t, np.cumsum(learned_sl['reward'].to_numpy()), linewidth=1.9, color='#1d3557', label='Q-learning')
        axes[0].set_title(f'{MONTH_NAMES[mi]}: cumulative reward')
        axes[0].set_xlabel('Hour in month')
        axes[0].set_ylabel('Cumulative reward')
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc='best', fontsize=8)

        axes[1].plot(local_t, np.cumsum(fixed_sl['grid_import_kwh'].to_numpy()), linestyle='--', linewidth=1.8, color='#9aa0a6', label='Fixed')
        axes[1].plot(local_t, np.cumsum(learned_sl['grid_import_kwh'].to_numpy()), linewidth=1.9, color='#1d3557', label='Q-learning')
        axes[1].set_title(f'{MONTH_NAMES[mi]}: cumulative grid import')
        axes[1].set_xlabel('Hour in month')
        axes[1].set_ylabel('kWh')
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc='best', fontsize=8)

        fig.suptitle(f'Monthly comparison - {MONTH_NAMES[mi]}', fontsize=12)
        fig.tight_layout()
        fig.savefig(monthly_dir / f'{mi + 1:02d}_{MONTH_NAMES[mi].lower()}_comparison.png', dpi=160)
        plt.close(fig)

        fixed_grid = float(np.sum(fixed_sl['grid_import_kwh'].to_numpy()))
        learned_grid = float(np.sum(learned_sl['grid_import_kwh'].to_numpy()))
        savings = 100.0 * (fixed_grid - learned_grid) / fixed_grid if fixed_grid > 0.0 else None
        rows.append({
            'month_index': mi + 1,
            'month_name': MONTH_NAMES[mi],
            'hours': end - start,
            'fixed_cumulative_reward': float(np.sum(fixed_sl['reward'].to_numpy())),
            'q_learning_cumulative_reward': float(np.sum(learned_sl['reward'].to_numpy())),
            'fixed_grid_import_kwh': fixed_grid,
            'q_learning_grid_import_kwh': learned_grid,
            'q_learning_savings_vs_fixed_pct': savings,
        })

    monthly_frame = pd.DataFrame(rows)
    monthly_frame.to_csv(output_dir / 'monthly_summary.csv', index=False)
    return monthly_frame


# ── Hlavná experimentálna slučka ───────────────────────────────────────────────
def run_experiment(
    schema_path: Path,
    building_name: str,
    episodes: int,
    baseline_cooling: float,
    random_seed: int,
    output_dir: Path,
    comparison_horizon: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    # ── Fixná referencia ──
    print('Evaluating fixed strategy (DHW=0, battery=0, cooling=fixed)...', flush=True)
    fixed_env = make_env(schema_path, building_name, random_seed)
    fixed_policy = FixedPolicy(dhw_action=0.0, electrical_action=0.0, cooling_action=baseline_cooling)
    fixed_run = run_policy(fixed_policy, fixed_env, deterministic=True)
    fixed_run.result.policy = f'Fixed(cool={baseline_cooling:.2f})'
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    # ── Q-learning ──
    print('Training Q-learning with 3 actions...', flush=True)
    training_env = make_env(schema_path, building_name, random_seed)
    agent = OwnAdaptiveTabularQLearning(
        training_env,
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
        random_seed=random_seed,
    )
    training_trace = train_q_learning(
        agent, training_env, episodes=episodes, progress_every=max(1, episodes // 20)
    )
    learned_run = run_policy(agent, training_env, deterministic=True)
    learned_run.result.policy = 'Q-learning (3 actions)'
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

    # ── Ukladanie výsledkov ──
    results_frame = build_results_frame(results).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)
    save_single_comparison_figure(results_frame, output_dir / 'reward_vs_fixed_comparison.png')
    save_time_and_learning_comparison(
        fixed_run, learned_run, training_trace,
        output_dir / 'reward_time_and_learning_comparison.png',
        comparison_horizon,
    )
    save_monthly_comparisons(fixed_run, learned_run, output_dir)
    np.save(output_dir / 'q_table.npy', agent.q_table)
    pd.DataFrame({
        'episode': np.arange(1, len(training_trace.episode_rewards) + 1),
        'episode_reward': training_trace.episode_rewards,
        'epsilon': training_trace.epsilons,
    }).to_csv(output_dir / 'learning_trace.csv', index=False)
    fixed_run.trajectory.to_csv(output_dir / 'trajectory_fixed_strategy.csv', index=False)
    learned_run.trajectory.to_csv(output_dir / 'trajectory_q_learning.csv', index=False)

    return results_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Q-learning so všetkými 3 akciami (DHW zásobník, batéria, chladenie) na jednej budove.'
    )
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument('--building', type=str, default=DEFAULT_BUILDING)
    parser.add_argument('--episodes', type=int, default=700,
                        help='Počet trénovacích epizód. Odporúčané min. 700 kvôli väčšiemu priestoru stavov.')
    parser.add_argument('--baseline-cooling', type=float, default=0.5,
                        help='Fixná hodnota chladenia pre referenčnú stratégiu (0.0–1.0).')
    parser.add_argument('--random-seed', type=int, default=7)
    parser.add_argument('--comparison-horizon', type=int, default=8760,
                        help='Počet krokov v grafe porovnania. 0 = celý rok.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print(f'Schema: {args.schema}', flush=True)
    print(f'Building: {args.building}', flush=True)
    print(f'Observations: {", ".join(ACTIVE_OBSERVATIONS)}', flush=True)
    print(f'Actions: {", ".join(ACTIVE_ACTIONS)}', flush=True)
    print(f'Action bins per dim: {ACTION_BIN_COUNTS}  →  {int(np.prod(ACTION_BIN_COUNTS))} joint akcií', flush=True)
    print(f'Reward function: WeatherOccupancyReward', flush=True)
    print(f'Episodes: {args.episodes}', flush=True)
    print(f'Baseline cooling action: {args.baseline_cooling}', flush=True)
    print(f'Output dir: {args.output_dir}', flush=True)

    results = run_experiment(
        schema_path=args.schema,
        building_name=args.building,
        episodes=args.episodes,
        baseline_cooling=args.baseline_cooling,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        comparison_horizon=args.comparison_horizon,
    )

    print('\nVýsledky:', flush=True)
    print(results.to_string(index=False))
    print('\nUložené súbory:', flush=True)
    for fname in [
        'summary_results.csv', 'learning_trace.csv', 'q_table.npy',
        'trajectory_fixed_strategy.csv', 'trajectory_q_learning.csv',
        'reward_vs_fixed_comparison.png', 'reward_time_and_learning_comparison.png',
        'monthly_summary.csv', 'monthly_comparisons/',
    ]:
        print(f'  {args.output_dir / fname}', flush=True)


if __name__ == '__main__':
    main()
