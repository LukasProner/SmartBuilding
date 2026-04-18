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
DEFAULT_BUILDINGS = ['Building_1', 'Building_2', 'Building_3']
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning1804_04'

# ── Observations ─────────────────────────────────────────────────────────────
# Kompaktný stavový priestor: používame iba predikcie + trendy, nie všetky
# jednotlivé predikcie zvlášť. Tým znížime počet stavov bez straty smerovej
# informácie (rast/pokles teploty a solárneho vstupu).
ACTIVE_OBSERVATIONS = [
    'outdoor_dry_bulb_temperature_predicted_1',
    'outdoor_dry_bulb_temperature_predicted_2',
    'diffuse_solar_irradiance_predicted_1',
    'diffuse_solar_irradiance_predicted_2',
    'direct_solar_irradiance_predicted_1',
    'direct_solar_irradiance_predicted_2',
    'electricity_pricing',
    'electricity_pricing_predicted_1',
    'dhw_storage_soc',
    'electrical_storage_soc',
    'occupant_count',
]

# Feature-space po transformácii observation vektora:
# - temp_pred_1
# - temp_trend_12 = temp_pred_2 - temp_pred_1
# - solar_pred_1_total = diffuse_1 + direct_1
# - solar_trend_12_total = (diffuse_2 + direct_2) - (diffuse_1 + direct_1)
# - price_now
# - price_trend_01 = price_pred_1 - price_now
# - dhw_storage_soc
# - electrical_storage_soc
# - occupancy_present
OBSERVATION_BIN_SIZES = {
    'temp_pred_1': 3,
    'temp_trend_12': 3,
    'solar_pred_1_total': 3,
    'solar_trend_12_total': 3,
    'price_now': 3,
    'price_trend_01': 3,
    'dhw_storage_soc': 3,
    'electrical_storage_soc': 3,
    'occupancy_present': 2,
}
# Veľkosť stavového priestoru: 3×3×3×3×3×3×3×3×2 = 13,122 stavov
ENGINEERED_FEATURES = list(OBSERVATION_BIN_SIZES.keys())

# ── Akcie ────────────────────────────────────────────────────────────────────
# Poradie musí zodpovedať env.action_space: [dhw_storage, electrical_storage, cooling_device]
ACTIVE_ACTIONS = ['dhw_storage', 'electrical_storage', 'cooling_device']

# Počet binov pre každú akciu (v rovnakom poradí ako ACTIVE_ACTIONS).
# Zásobníky majú 3 biny: -1 (vybíjaj) / 0 (nič nerob) / +1 (nabíjaj)
# Chladenie má 5 binov: 0.0 / 0.25 / 0.5 / 0.75 / 1.0
# Celkový počet joint akcií: 3 × 3 × 5 = 45
ACTION_BIN_COUNTS = [3, 3, 5]


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
            occupancy_present = 1.0 if occupant_count > 0.0 else 0.0
            mean_forecast = np.mean([
                observation.get('outdoor_dry_bulb_temperature_predicted_1', 0.0),
                observation.get('outdoor_dry_bulb_temperature_predicted_2', 0.0),
                observation.get('outdoor_dry_bulb_temperature_predicted_3', 0.0),
            ])
            occupancy_factor = 1.0 + self.occupancy_weight * occupancy_present
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            reward_list.append(-(grid_import * occupancy_factor * weather_factor))
        return [sum(reward_list)] if self.central_agent else reward_list


class GridImportOnlyReward(RewardFunction):
    """Jednoduchá reward funkcia: penalizuje iba dovoz energie zo siete."""

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for observation in observations:
            grid_import = max(observation['net_electricity_consumption'], 0.0)
            reward_list.append(-grid_import)
        return [sum(reward_list)] if self.central_agent else reward_list


class PricingAwareReward(RewardFunction):
    """Penalizuje grid import viac v drahých hodinách."""

    def __init__(self, env_metadata, price_weight: float = 2.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.price_weight = float(price_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for observation in observations:
            grid_import = max(observation['net_electricity_consumption'], 0.0)
            price_now = max(observation.get('electricity_pricing', 0.0), 0.0)
            reward_list.append(-(grid_import * (1.0 + self.price_weight * price_now)))
        return [sum(reward_list)] if self.central_agent else reward_list


class ComfortAwareReward(RewardFunction):
    """Penalizuje grid import a zároveň prehrievanie nad cooling setpoint pri obsadení."""

    def __init__(
        self,
        env_metadata,
        occupancy_weight: float = 1.0,
        hot_weather_weight: float = 0.05,
        discomfort_weight: float = 6.0,
        occupied_discomfort_multiplier: float = 2.0,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.occupancy_weight = float(occupancy_weight)
        self.hot_weather_weight = float(hot_weather_weight)
        self.discomfort_weight = float(discomfort_weight)
        self.occupied_discomfort_multiplier = float(occupied_discomfort_multiplier)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for observation in observations:
            grid_import = max(observation['net_electricity_consumption'], 0.0)
            occupant_count = max(observation.get('occupant_count', 0.0), 0.0)
            occupancy_present = 1.0 if occupant_count > 0.0 else 0.0
            mean_forecast = np.mean([
                observation.get('outdoor_dry_bulb_temperature_predicted_1', 0.0),
                observation.get('outdoor_dry_bulb_temperature_predicted_2', 0.0),
                observation.get('outdoor_dry_bulb_temperature_predicted_3', 0.0),
            ])
            cooling_delta = max(observation.get('indoor_dry_bulb_temperature_cooling_delta', 0.0), 0.0)

            occupancy_factor = 1.0 + self.occupancy_weight * occupancy_present
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            discomfort_factor = 1.0 + self.occupied_discomfort_multiplier * occupancy_present
            reward_list.append(
                -(
                    grid_import * occupancy_factor * weather_factor
                    + self.discomfort_weight * cooling_delta * discomfort_factor
                )
            )

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
        self.index_by_name = {name: i for i, name in enumerate(self.observation_names)}

        self.feature_names = [
            'temp_pred_1',
            'temp_trend_12',
            'solar_pred_1_total',
            'solar_trend_12_total',
            'price_now',
            'price_trend_01',
            'dhw_storage_soc',
            'electrical_storage_soc',
            'occupancy_present',
        ]
        self.bin_counts = [int(bin_counts[name]) for name in self.feature_names]
        self.state_shape = tuple(self.bin_counts)
        self.state_count = int(np.prod(self.state_shape))

        low_by_name = {name: float(v) for name, v in zip(self.observation_names, env.observation_space[0].low)}
        high_by_name = {name: float(v) for name, v in zip(self.observation_names, env.observation_space[0].high)}

        t1_lo = low_by_name['outdoor_dry_bulb_temperature_predicted_1']
        t1_hi = high_by_name['outdoor_dry_bulb_temperature_predicted_1']
        t2_lo = low_by_name['outdoor_dry_bulb_temperature_predicted_2']
        t2_hi = high_by_name['outdoor_dry_bulb_temperature_predicted_2']

        d1_lo = low_by_name['diffuse_solar_irradiance_predicted_1']
        d1_hi = high_by_name['diffuse_solar_irradiance_predicted_1']
        d2_lo = low_by_name['diffuse_solar_irradiance_predicted_2']
        d2_hi = high_by_name['diffuse_solar_irradiance_predicted_2']
        r1_lo = low_by_name['direct_solar_irradiance_predicted_1']
        r1_hi = high_by_name['direct_solar_irradiance_predicted_1']
        r2_lo = low_by_name['direct_solar_irradiance_predicted_2']
        r2_hi = high_by_name['direct_solar_irradiance_predicted_2']
        p0_lo = low_by_name['electricity_pricing']
        p0_hi = high_by_name['electricity_pricing']
        p1_lo = low_by_name['electricity_pricing_predicted_1']
        p1_hi = high_by_name['electricity_pricing_predicted_1']

        solar1_lo = d1_lo + r1_lo
        solar1_hi = d1_hi + r1_hi
        solar2_lo = d2_lo + r2_lo
        solar2_hi = d2_hi + r2_hi

        feature_lows = [
            t1_lo,
            t2_lo - t1_hi,
            solar1_lo,
            solar2_lo - solar1_hi,
            p0_lo,
            p1_lo - p0_hi,
            low_by_name['dhw_storage_soc'],
            low_by_name['electrical_storage_soc'],
            0.0,
        ]
        feature_highs = [
            t1_hi,
            t2_hi - t1_lo,
            solar1_hi,
            solar2_hi - solar1_lo,
            p0_hi,
            p1_hi - p0_lo,
            high_by_name['dhw_storage_soc'],
            high_by_name['electrical_storage_soc'],
            1.0,
        ]

        self.edges = []
        for low, high, count in zip(feature_lows, feature_highs, self.bin_counts):
            self.edges.append(np.linspace(float(low), float(high), count + 1)[1:-1])

    def encode(self, observation: list[float]) -> int:
        t1 = float(observation[self.index_by_name['outdoor_dry_bulb_temperature_predicted_1']])
        t2 = float(observation[self.index_by_name['outdoor_dry_bulb_temperature_predicted_2']])
        d1 = float(observation[self.index_by_name['diffuse_solar_irradiance_predicted_1']])
        d2 = float(observation[self.index_by_name['diffuse_solar_irradiance_predicted_2']])
        r1 = float(observation[self.index_by_name['direct_solar_irradiance_predicted_1']])
        r2 = float(observation[self.index_by_name['direct_solar_irradiance_predicted_2']])
        price_now = float(observation[self.index_by_name['electricity_pricing']])
        price_1 = float(observation[self.index_by_name['electricity_pricing_predicted_1']])
        dhw_soc = float(observation[self.index_by_name['dhw_storage_soc']])
        elec_soc = float(observation[self.index_by_name['electrical_storage_soc']])
        occ = float(observation[self.index_by_name['occupant_count']])
        occupancy_present = 1.0 if occ > 0.0 else 0.0

        feature_values = [
            t1,
            t2 - t1,
            d1 + r1,
            (d2 + r2) - (d1 + r1),
            price_now,
            price_1 - price_now,
            dhw_soc,
            elec_soc,
            occupancy_present,
        ]
        digits = [int(np.digitize(float(v), e, right=False)) for v, e in zip(feature_values, self.edges)]
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

    def decode_one(self, action_index: int) -> list[float]:
        return list(self.joint_actions[action_index])


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
        self.last_state_indices: list[int] = []
        self.last_action_indices: list[int] = []
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
        if not self.last_state_indices or not self.last_action_indices:
            raise RuntimeError('Cannot update Q-table before selecting an action.')

        for state_index, action_index, reward, next_observation in zip(
            self.last_state_indices,
            self.last_action_indices,
            rewards,
            next_observations,
        ):
            next_state_index = self.observation_discretizer.encode(next_observation)
            best_next_value = 0.0 if terminated else float(np.max(self.q_table[next_state_index]))
            td_target = float(reward) + self.discount_factor * best_next_value
            td_error = td_target - float(self.q_table[state_index, action_index])
            self.q_table[state_index, action_index] += self.learning_rate * td_error

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
def make_env(
    schema_path: Path,
    building_names: list[str],
    random_seed: int,
    reward_function=WeatherOccupancyReward,
) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=False,
        buildings=building_names,
        active_observations=ACTIVE_OBSERVATIONS,
        active_actions=ACTIVE_ACTIONS,
        reward_function=reward_function,
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
        while not terminated:
            actions = agent.predict(observations, deterministic=False)
            next_observations, rewards, terminated, _, _ = env.step(actions)
            agent.update(rewards, next_observations, terminated)
            observations = next_observations
            cumulative_reward += float(np.sum(rewards))
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
    buildings = base_env.buildings
    kpis = base_env.evaluate()
    building_names = [building.name for building in buildings]
    discomfort_rows = kpis[
        (kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_proportion')
    ]
    discomfort_proportion = float(discomfort_rows['value'].mean()) if not discomfort_rows.empty else 0.0

    aggregate_net_consumption = np.zeros(len(reward_trace), dtype=float)
    trajectory_data = {
        'time_step': np.arange(len(reward_trace)),
        'reward': reward_trace,
        'cumulative_reward': np.cumsum(reward_trace),
    }

    for building in buildings:
        net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)[:len(reward_trace)]
        aggregate_net_consumption += net_consumption
        trajectory_data[f'grid_import_{building.name}_kwh'] = np.clip(net_consumption, 0.0, None)

    trajectory_data['grid_import_kwh'] = np.clip(aggregate_net_consumption, 0.0, None)
    trajectory_data['cumulative_grid_import_kwh'] = np.cumsum(np.clip(aggregate_net_consumption, 0.0, None))
    trajectory = pd.DataFrame(trajectory_data)
    result = ExperimentResult(
        policy=agent.__class__.__name__,
        total_grid_import_kwh=float(np.sum(np.clip(aggregate_net_consumption, 0.0, None))),
        total_net_consumption_kwh=float(np.sum(aggregate_net_consumption)),
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
def save_policy_comparison_figure(
    results_frame: pd.DataFrame,
    policy_runs: list[PolicyRun],
    output_path: Path,
) -> None:
    labels = results_frame['policy'].tolist()
    x = np.arange(len(labels))
    colors = ['#9aa0a6', '#1d3557', '#2a9d8f', '#e76f51', '#6d597a']

    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    axes[0, 0].bar(x, results_frame['grid_import_kwh'], color=colors)
    axes[0, 0].set_title('Total grid import')
    axes[0, 0].set_ylabel('kWh')
    axes[0, 0].set_xticks(x, labels, rotation=20, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.25)

    axes[0, 1].bar(x, results_frame['savings_vs_fixed_pct'].fillna(0.0), color=colors)
    axes[0, 1].set_title('Savings vs fixed strategy')
    axes[0, 1].set_ylabel('%')
    axes[0, 1].set_xticks(x, labels, rotation=20, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.25)

    axes[1, 0].bar(x, results_frame['discomfort_proportion'], color=colors)
    axes[1, 0].set_title('Discomfort proportion')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].set_xticks(x, labels, rotation=20, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.25)

    profile_hours = min(14 * 24, min(len(policy_run.trajectory) for policy_run in policy_runs))
    profile_index = np.arange(profile_hours)

    for color, policy_run in zip(colors, policy_runs):
        profile = policy_run.trajectory.groupby(
            policy_run.trajectory['time_step'] % profile_hours
        )['grid_import_kwh'].mean()
        line_style = '--' if policy_run.result.policy.startswith('Fixed') else '-'
        axes[1, 1].plot(
            profile_index,
            profile.reindex(profile_index, fill_value=np.nan).to_numpy(),
            linestyle=line_style,
            linewidth=2.0,
            color=color,
            label=policy_run.result.policy,
        )
    axes[1, 1].set_title('Average grid import over 14-day profile')
    axes[1, 1].set_xlabel('Hour in 14-day cycle')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].set_xticks(range(0, profile_hours + 1, 24))
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc='best', fontsize=8)

    fig.suptitle('3 budovy: Fixed vs 4 reward varianty Q-learningu', fontsize=13)
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


# ── Hlavná experimentálna slučka ───────────────────────────────────────────────
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

    # ── Fixná referencia ──
    print('Evaluating fixed strategy (DHW=0, battery=0, cooling=fixed)...', flush=True)
    fixed_env = make_env(schema_path, building_names, random_seed)
    fixed_policy = FixedPolicy(dhw_action=0.0, electrical_action=0.0, cooling_action=baseline_cooling)
    fixed_run = run_policy(fixed_policy, fixed_env, deterministic=True)
    fixed_run.result.policy = f'Fixed(cool={baseline_cooling:.2f})'
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    # ── Q-learning (WeatherOccupancyReward) ──
    print('Training Q-learning with WeatherOccupancyReward...', flush=True)
    weather_env = make_env(schema_path, building_names, random_seed, reward_function=WeatherOccupancyReward)
    weather_agent = OwnAdaptiveTabularQLearning(
        weather_env,
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
    weather_trace = train_q_learning(
        weather_agent, weather_env, episodes=episodes, progress_every=max(1, episodes // 20)
    )
    learned_weather_run = run_policy(weather_agent, weather_env, deterministic=True)
    learned_weather_run.result.policy = 'Q-learning (Weather reward)'
    learned_weather_run.result.training_seconds = weather_trace.training_seconds
    learned_weather_run.result.stability_episode = weather_trace.stability_episode
    learned_weather_run.result.last_10_episode_reward_mean = (
        float(np.mean(weather_trace.episode_rewards[-10:])) if weather_trace.episode_rewards else None
    )
    if fixed_run.result.total_grid_import_kwh > 0.0:
        learned_weather_run.result.savings_vs_fixed_pct = 100.0 * (
            fixed_run.result.total_grid_import_kwh - learned_weather_run.result.total_grid_import_kwh
        ) / fixed_run.result.total_grid_import_kwh
    results.append(learned_weather_run.result)

    # ── Q-learning (GridImportOnlyReward) ──
    print('Training Q-learning with GridImportOnlyReward...', flush=True)
    energy_env = make_env(schema_path, building_names, random_seed, reward_function=GridImportOnlyReward)
    energy_agent = OwnAdaptiveTabularQLearning(
        energy_env,
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
    energy_trace = train_q_learning(
        energy_agent, energy_env, episodes=episodes, progress_every=max(1, episodes // 20)
    )
    learned_energy_run = run_policy(energy_agent, energy_env, deterministic=True)
    learned_energy_run.result.policy = 'Q-learning (Energy-only reward)'
    learned_energy_run.result.training_seconds = energy_trace.training_seconds
    learned_energy_run.result.stability_episode = energy_trace.stability_episode
    learned_energy_run.result.last_10_episode_reward_mean = (
        float(np.mean(energy_trace.episode_rewards[-10:])) if energy_trace.episode_rewards else None
    )
    if fixed_run.result.total_grid_import_kwh > 0.0:
        learned_energy_run.result.savings_vs_fixed_pct = 100.0 * (
            fixed_run.result.total_grid_import_kwh - learned_energy_run.result.total_grid_import_kwh
        ) / fixed_run.result.total_grid_import_kwh
    results.append(learned_energy_run.result)

    # ── Q-learning (PricingAwareReward) ──
    print('Training Q-learning with PricingAwareReward...', flush=True)
    price_env = make_env(schema_path, building_names, random_seed, reward_function=PricingAwareReward)
    price_agent = OwnAdaptiveTabularQLearning(
        price_env,
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
    price_trace = train_q_learning(
        price_agent, price_env, episodes=episodes, progress_every=max(1, episodes // 20)
    )
    learned_price_run = run_policy(price_agent, price_env, deterministic=True)
    learned_price_run.result.policy = 'Q-learning (Pricing reward)'
    learned_price_run.result.training_seconds = price_trace.training_seconds
    learned_price_run.result.stability_episode = price_trace.stability_episode
    learned_price_run.result.last_10_episode_reward_mean = (
        float(np.mean(price_trace.episode_rewards[-10:])) if price_trace.episode_rewards else None
    )
    if fixed_run.result.total_grid_import_kwh > 0.0:
        learned_price_run.result.savings_vs_fixed_pct = 100.0 * (
            fixed_run.result.total_grid_import_kwh - learned_price_run.result.total_grid_import_kwh
        ) / fixed_run.result.total_grid_import_kwh
    results.append(learned_price_run.result)

    # ── Q-learning (ComfortAwareReward) ──
    print('Training Q-learning with ComfortAwareReward...', flush=True)
    comfort_env = make_env(schema_path, building_names, random_seed, reward_function=ComfortAwareReward)
    comfort_agent = OwnAdaptiveTabularQLearning(
        comfort_env,
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
    comfort_trace = train_q_learning(
        comfort_agent, comfort_env, episodes=episodes, progress_every=max(1, episodes // 20)
    )
    learned_comfort_run = run_policy(comfort_agent, comfort_env, deterministic=True)
    learned_comfort_run.result.policy = 'Q-learning (Comfort reward)'
    learned_comfort_run.result.training_seconds = comfort_trace.training_seconds
    learned_comfort_run.result.stability_episode = comfort_trace.stability_episode
    learned_comfort_run.result.last_10_episode_reward_mean = (
        float(np.mean(comfort_trace.episode_rewards[-10:])) if comfort_trace.episode_rewards else None
    )
    if fixed_run.result.total_grid_import_kwh > 0.0:
        learned_comfort_run.result.savings_vs_fixed_pct = 100.0 * (
            fixed_run.result.total_grid_import_kwh - learned_comfort_run.result.total_grid_import_kwh
        ) / fixed_run.result.total_grid_import_kwh
    results.append(learned_comfort_run.result)

    # ── Ukladanie výsledkov ──
    results_frame = build_results_frame(results).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)
    save_policy_comparison_figure(
        results_frame,
        [fixed_run, learned_weather_run, learned_energy_run, learned_price_run, learned_comfort_run],
        output_dir / 'policy_comparison.png',
    )
    save_time_and_learning_comparison(
        fixed_run, learned_weather_run, weather_trace,
        output_dir / 'reward_time_and_learning_comparison_weather.png',
        comparison_horizon,
    )
    save_time_and_learning_comparison(
        fixed_run, learned_energy_run, energy_trace,
        output_dir / 'reward_time_and_learning_comparison_energy.png',
        comparison_horizon,
    )
    save_time_and_learning_comparison(
        fixed_run, learned_price_run, price_trace,
        output_dir / 'reward_time_and_learning_comparison_pricing.png',
        comparison_horizon,
    )
    save_time_and_learning_comparison(
        fixed_run, learned_comfort_run, comfort_trace,
        output_dir / 'reward_time_and_learning_comparison_comfort.png',
        comparison_horizon,
    )
    np.save(output_dir / 'q_table_weather.npy', weather_agent.q_table)
    np.save(output_dir / 'q_table_energy.npy', energy_agent.q_table)
    np.save(output_dir / 'q_table_pricing.npy', price_agent.q_table)
    np.save(output_dir / 'q_table_comfort.npy', comfort_agent.q_table)
    pd.DataFrame({
        'episode': np.arange(1, len(weather_trace.episode_rewards) + 1),
        'episode_reward': weather_trace.episode_rewards,
        'epsilon': weather_trace.epsilons,
    }).to_csv(output_dir / 'learning_trace_weather.csv', index=False)
    pd.DataFrame({
        'episode': np.arange(1, len(energy_trace.episode_rewards) + 1),
        'episode_reward': energy_trace.episode_rewards,
        'epsilon': energy_trace.epsilons,
    }).to_csv(output_dir / 'learning_trace_energy.csv', index=False)
    pd.DataFrame({
        'episode': np.arange(1, len(price_trace.episode_rewards) + 1),
        'episode_reward': price_trace.episode_rewards,
        'epsilon': price_trace.epsilons,
    }).to_csv(output_dir / 'learning_trace_pricing.csv', index=False)
    pd.DataFrame({
        'episode': np.arange(1, len(comfort_trace.episode_rewards) + 1),
        'episode_reward': comfort_trace.episode_rewards,
        'epsilon': comfort_trace.epsilons,
    }).to_csv(output_dir / 'learning_trace_comfort.csv', index=False)
    fixed_run.trajectory.to_csv(output_dir / 'trajectory_fixed_strategy.csv', index=False)
    learned_weather_run.trajectory.to_csv(output_dir / 'trajectory_q_learning_weather.csv', index=False)
    learned_energy_run.trajectory.to_csv(output_dir / 'trajectory_q_learning_energy.csv', index=False)
    learned_price_run.trajectory.to_csv(output_dir / 'trajectory_q_learning_pricing.csv', index=False)
    learned_comfort_run.trajectory.to_csv(output_dir / 'trajectory_q_learning_comfort.csv', index=False)
    fixed_run.kpis.to_csv(output_dir / 'kpis_fixed.csv', index=False)
    learned_weather_run.kpis.to_csv(output_dir / 'kpis_weather.csv', index=False)
    learned_energy_run.kpis.to_csv(output_dir / 'kpis_energy.csv', index=False)
    learned_price_run.kpis.to_csv(output_dir / 'kpis_pricing.csv', index=False)
    learned_comfort_run.kpis.to_csv(output_dir / 'kpis_comfort.csv', index=False)

    return results_frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Q-learning (compact-state, 3 buildings): teplota/solar/pricing, porovnanie 4 reward funkcií vrátane discomfort-aware variantu.'
    )
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument('--buildings', nargs='+', default=DEFAULT_BUILDINGS)
    parser.add_argument('--episodes', type=int, default=200,
                        help='Počet trénovacích epizód. Odporúčané min. 700 kvôli väčšiemu priestoru stavov.')
    parser.add_argument('--baseline-cooling', type=float, default=0.5,
                        help='Fixná hodnota chladenia pre referenčnú stratégiu (0.0–1.0).')
    parser.add_argument('--random-seed', type=int, default=7)
    parser.add_argument('--comparison-horizon', type=int, default=719,
                        help='Počet krokov v grafe porovnania. 0 = celý priebeh datasetu.')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print(f'Schema: {args.schema}', flush=True)
    print(f'Buildings: {", ".join(args.buildings)}', flush=True)
    print(f'Raw observations: {", ".join(ACTIVE_OBSERVATIONS)}', flush=True)
    print(f'Engineered features: {", ".join(ENGINEERED_FEATURES)}', flush=True)
    print(f'Feature bins: {OBSERVATION_BIN_SIZES}  ->  {int(np.prod(list(OBSERVATION_BIN_SIZES.values())))} states', flush=True)
    print(f'Actions: {", ".join(ACTIVE_ACTIONS)}', flush=True)
    print(f'Action bins per dim: {ACTION_BIN_COUNTS}  →  {int(np.prod(ACTION_BIN_COUNTS))} joint akcií', flush=True)
    print('Reward functions: WeatherOccupancyReward, GridImportOnlyReward, PricingAwareReward, ComfortAwareReward', flush=True)
    print(f'Episodes: {args.episodes}', flush=True)
    print(f'Baseline cooling action: {args.baseline_cooling}', flush=True)
    print(f'Output dir: {args.output_dir}', flush=True)

    results = run_experiment(
        schema_path=args.schema,
        building_names=args.buildings,
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
        'summary_results.csv',
        'learning_trace_weather.csv', 'learning_trace_energy.csv', 'learning_trace_pricing.csv', 'learning_trace_comfort.csv',
        'q_table_weather.npy', 'q_table_energy.npy', 'q_table_pricing.npy', 'q_table_comfort.npy',
        'trajectory_fixed_strategy.csv',
        'trajectory_q_learning_weather.csv', 'trajectory_q_learning_energy.csv', 'trajectory_q_learning_pricing.csv', 'trajectory_q_learning_comfort.csv',
        'kpis_fixed.csv', 'kpis_weather.csv', 'kpis_energy.csv', 'kpis_pricing.csv', 'kpis_comfort.csv',
        'policy_comparison.png',
        'reward_time_and_learning_comparison_weather.png',
        'reward_time_and_learning_comparison_energy.png',
        'reward_time_and_learning_comparison_pricing.png',
        'reward_time_and_learning_comparison_comfort.png',
    ]:
        print(f'  {args.output_dir / fname}', flush=True)


if __name__ == '__main__':
    main()
