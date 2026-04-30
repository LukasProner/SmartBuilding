from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from itertools import product as iterproduct
from pathlib import Path

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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning2404_02_01'

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

OBS_BINS = {
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

ACTIVE_ACTIONS = ['dhw_storage', 'electrical_storage', 'cooling_device']
ACTION_BINS = [3, 3, 5]


class WeatherOccupancyReward(RewardFunction):
    def __init__(self, env_metadata, occupancy_weight: float = 1.5, hot_weather_weight: float = 0.05, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.occupancy_weight = float(occupancy_weight)
        self.hot_weather_weight = float(hot_weather_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            occ_present = 1.0 if max(obs.get('occupant_count', 0.0), 0.0) > 0.0 else 0.0
            mean_forecast = float(np.mean([
                obs.get('outdoor_dry_bulb_temperature_predicted_1', 0.0),
                obs.get('outdoor_dry_bulb_temperature_predicted_2', 0.0),
                obs.get('outdoor_dry_bulb_temperature_predicted_3', 0.0),
            ]))
            occ_factor = 1.0 + self.occupancy_weight * occ_present
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            rewards.append(-(grid_import * occ_factor * weather_factor))
        return [sum(rewards)] if self.central_agent else rewards


class GridImportOnlyReward(RewardFunction):
    def calculate(self, observations: list[dict]) -> list[float]:
        rewards = [-max(obs['net_electricity_consumption'], 0.0) for obs in observations]
        return [sum(rewards)] if self.central_agent else rewards


class PricingAwareReward(RewardFunction):
    def __init__(self, env_metadata, price_weight: float = 2.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.price_weight = float(price_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            price = max(obs.get('electricity_pricing', 0.0), 0.0)
            rewards.append(-(grid_import * (1.0 + self.price_weight * price)))
        return [sum(rewards)] if self.central_agent else rewards


class ComfortAwareReward(RewardFunction):
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
        rewards: list[float] = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            occ_present = 1.0 if max(obs.get('occupant_count', 0.0), 0.0) > 0.0 else 0.0
            mean_forecast = float(np.mean([
                obs.get('outdoor_dry_bulb_temperature_predicted_1', 0.0),
                obs.get('outdoor_dry_bulb_temperature_predicted_2', 0.0),
                obs.get('outdoor_dry_bulb_temperature_predicted_3', 0.0),
            ]))
            cooling_delta = max(obs.get('indoor_dry_bulb_temperature_cooling_delta', 0.0), 0.0)
            occ_factor = 1.0 + self.occupancy_weight * occ_present
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            discomfort_factor = 1.0 + self.occupied_discomfort_multiplier * occ_present
            rewards.append(-(grid_import * occ_factor * weather_factor + self.discomfort_weight * cooling_delta * discomfort_factor))
        return [sum(rewards)] if self.central_agent else rewards


class PeakShavingReward(RewardFunction):
    def calculate(self, observations: list[dict]) -> list[float]:
        rewards = [-(max(obs['net_electricity_consumption'], 0.0) ** 2) for obs in observations]
        return [sum(rewards)] if self.central_agent else rewards


class SolarAlignmentReward(RewardFunction):
    def __init__(self, env_metadata, solar_weight: float = 0.3, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.solar_weight = float(solar_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            diffuse = max(obs.get('diffuse_solar_irradiance_predicted_1', 0.0), 0.0)
            direct = max(obs.get('direct_solar_irradiance_predicted_1', 0.0), 0.0)
            solar_factor = 1.0 - self.solar_weight * min((diffuse + direct) / 800.0, 1.0)
            rewards.append(-(grid_import * solar_factor))
        return [sum(rewards)] if self.central_agent else rewards


class StorageManagementReward(RewardFunction):
    def __init__(self, env_metadata, soc_penalty_weight: float = 2.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.soc_penalty_weight = float(soc_penalty_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            price = max(obs.get('electricity_pricing', 0.0), 0.0)
            avg_soc = (obs.get('dhw_storage_soc', 0.5) + obs.get('electrical_storage_soc', 0.5)) / 2.0
            if price > 0.3:
                soc_waste = max(0.5 - avg_soc, 0.0)
            else:
                soc_waste = max(avg_soc - 0.8, 0.0) * 0.5
            rewards.append(-(grid_import + self.soc_penalty_weight * soc_waste))
        return [sum(rewards)] if self.central_agent else rewards


class RampingPenaltyReward(RewardFunction):
    def __init__(self, env_metadata, ramping_weight: float = 0.5, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.ramping_weight = float(ramping_weight)
        self.previous_imports: dict[int, float] = {}

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for i, obs in enumerate(observations):
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            previous = self.previous_imports.get(i, grid_import)
            self.previous_imports[i] = grid_import
            rewards.append(-(grid_import + self.ramping_weight * abs(grid_import - previous)))
        return [sum(rewards)] if self.central_agent else rewards


class TimeOfUseReward(RewardFunction):
    def __init__(self, env_metadata, peak_multiplier: float = 3.0, night_discount: float = 0.5, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.peak_multiplier = float(peak_multiplier)
        self.night_discount = float(night_discount)
        self.step = 0

    def calculate(self, observations: list[dict]) -> list[float]:
        hour = self.step % 24
        self.step += 1
        if 17 <= hour <= 21:
            multiplier = self.peak_multiplier
        elif hour >= 22 or hour <= 6:
            multiplier = self.night_discount
        else:
            multiplier = 1.0

        rewards = [-(max(obs['net_electricity_consumption'], 0.0) * multiplier) for obs in observations]
        return [sum(rewards)] if self.central_agent else rewards


class SelfSufficiencyReward(RewardFunction):
    def __init__(self, env_metadata, export_bonus: float = 0.5, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.export_bonus = float(export_bonus)

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for obs in observations:
            net = obs['net_electricity_consumption']
            rewards.append(-net if net > 0 else self.export_bonus * abs(net))
        return [sum(rewards)] if self.central_agent else rewards


class CombinedMultiObjectiveReward(RewardFunction):
    def __init__(self, env_metadata, price_weight: float = 1.0, comfort_weight: float = 3.0, peak_weight: float = 0.1, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.price_weight = float(price_weight)
        self.comfort_weight = float(comfort_weight)
        self.peak_weight = float(peak_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            price = max(obs.get('electricity_pricing', 0.0), 0.0)
            cooling_delta = max(obs.get('indoor_dry_bulb_temperature_cooling_delta', 0.0), 0.0)
            rewards.append(-(grid_import + self.price_weight * grid_import * price + self.comfort_weight * cooling_delta + self.peak_weight * grid_import ** 2))
        return [sum(rewards)] if self.central_agent else rewards


class NightPrechargeReward(RewardFunction):
    def __init__(self, env_metadata, soc_bonus: float = 1.5, peak_import_multiplier: float = 2.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.soc_bonus = float(soc_bonus)
        self.peak_import_multiplier = float(peak_import_multiplier)
        self.step = 0

    def calculate(self, observations: list[dict]) -> list[float]:
        hour = self.step % 24
        self.step += 1
        rewards: list[float] = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            avg_soc = (obs.get('dhw_storage_soc', 0.5) + obs.get('electrical_storage_soc', 0.5)) / 2.0
            if 0 <= hour <= 6:
                reward = -grid_import * 0.4 + self.soc_bonus * avg_soc
            elif 17 <= hour <= 21:
                reward = -grid_import * self.peak_import_multiplier + self.soc_bonus * avg_soc * 0.5
            else:
                reward = -grid_import
            rewards.append(reward)
        return [sum(rewards)] if self.central_agent else rewards


REWARD_CONFIGS: list[tuple[str, str, type[RewardFunction]]] = [
    ('weather', 'Weather reward', WeatherOccupancyReward),
    ('energy', 'Energy-only reward', GridImportOnlyReward),
    ('pricing', 'Pricing reward', PricingAwareReward),
    ('comfort', 'Comfort reward', ComfortAwareReward),
    ('peak', 'Peak-shaving reward', PeakShavingReward),
    ('solar', 'Solar reward', SolarAlignmentReward),
    ('storage', 'Storage reward', StorageManagementReward),
    ('ramping', 'Ramping reward', RampingPenaltyReward),
    ('tou', 'TimeOfUse reward', TimeOfUseReward),
    ('selfsuff', 'SelfSufficiency reward', SelfSufficiencyReward),
    ('combined', 'Combined reward', CombinedMultiObjectiveReward),
    ('nightpre', 'NightPrecharge reward', NightPrechargeReward),
]

#
class FixedPolicy:
    def __init__(self, dhw_action: float = 0.0, electrical_action: float = 0.0, cooling_action: float = 0.5):
        self.actions = [float(dhw_action), float(electrical_action), float(cooling_action)]

    def act(self, observations: list[list[float]]) -> list[list[float]]:
        return [list(self.actions) for _ in observations]


class ObservationDiscretizer:
    def __init__(self, env: CityLearnEnv):
        self.observation_names = env.observation_names[0]
        self.index = {name: i for i, name in enumerate(self.observation_names)}
        self.feature_names = list(OBS_BINS.keys())
        self.bin_counts = [int(OBS_BINS[name]) for name in self.feature_names]
        self.state_shape = tuple(self.bin_counts)
        self.state_count = int(np.prod(self.state_shape))

        low_by_name = {name: float(value) for name, value in zip(self.observation_names, env.observation_space[0].low)}
        high_by_name = {name: float(value) for name, value in zip(self.observation_names, env.observation_space[0].high)}

        t1_lo, t1_hi = low_by_name['outdoor_dry_bulb_temperature_predicted_1'], high_by_name['outdoor_dry_bulb_temperature_predicted_1']
        t2_lo, t2_hi = low_by_name['outdoor_dry_bulb_temperature_predicted_2'], high_by_name['outdoor_dry_bulb_temperature_predicted_2']
        d1_lo, d1_hi = low_by_name['diffuse_solar_irradiance_predicted_1'], high_by_name['diffuse_solar_irradiance_predicted_1']
        d2_lo, d2_hi = low_by_name['diffuse_solar_irradiance_predicted_2'], high_by_name['diffuse_solar_irradiance_predicted_2']
        r1_lo, r1_hi = low_by_name['direct_solar_irradiance_predicted_1'], high_by_name['direct_solar_irradiance_predicted_1']
        r2_lo, r2_hi = low_by_name['direct_solar_irradiance_predicted_2'], high_by_name['direct_solar_irradiance_predicted_2']
        p0_lo, p0_hi = low_by_name['electricity_pricing'], high_by_name['electricity_pricing']
        p1_lo, p1_hi = low_by_name['electricity_pricing_predicted_1'], high_by_name['electricity_pricing_predicted_1']
        s1_lo, s1_hi = d1_lo + r1_lo, d1_hi + r1_hi
        s2_lo, s2_hi = d2_lo + r2_lo, d2_hi + r2_hi

        lows = [
            t1_lo,
            t2_lo - t1_hi,
            s1_lo,
            s2_lo - s1_hi,
            p0_lo,
            p1_lo - p0_hi,
            low_by_name['dhw_storage_soc'],
            low_by_name['electrical_storage_soc'],
            0.0,
        ]
        highs = [
            t1_hi,
            t2_hi - t1_lo,
            s1_hi,
            s2_hi - s1_lo,
            p0_hi,
            p1_hi - p0_lo,
            high_by_name['dhw_storage_soc'],
            high_by_name['electrical_storage_soc'],
            1.0,
        ]
        self.edges = [
            np.linspace(float(low), float(high), count + 1)[1:-1]
            for low, high, count in zip(lows, highs, self.bin_counts)
        ]

    def encode(self, observation: list[float]) -> int:
        idx = self.index
        values = [
            float(observation[idx['outdoor_dry_bulb_temperature_predicted_1']]),
            float(observation[idx['outdoor_dry_bulb_temperature_predicted_2']]) - float(observation[idx['outdoor_dry_bulb_temperature_predicted_1']]),
            float(observation[idx['diffuse_solar_irradiance_predicted_1']]) + float(observation[idx['direct_solar_irradiance_predicted_1']]),
            (float(observation[idx['diffuse_solar_irradiance_predicted_2']]) + float(observation[idx['direct_solar_irradiance_predicted_2']]))
            - (float(observation[idx['diffuse_solar_irradiance_predicted_1']]) + float(observation[idx['direct_solar_irradiance_predicted_1']])),
            float(observation[idx['electricity_pricing']]),
            float(observation[idx['electricity_pricing_predicted_1']]) - float(observation[idx['electricity_pricing']]),
            float(observation[idx['dhw_storage_soc']]),
            float(observation[idx['electrical_storage_soc']]),
            1.0 if float(observation[idx['occupant_count']]) > 0.0 else 0.0,
        ]
        digits = [int(np.digitize(value, edge, right=False)) for value, edge in zip(values, self.edges)]
        return int(np.ravel_multi_index(tuple(digits), self.state_shape))

#
class ActionDiscretizer:
    def __init__(self, env: CityLearnEnv):
        n_dims = env.action_space[0].shape[0] # koľko akčných zložiek má jedna akcia pre jednu budovu.
        if len(ACTION_BINS) != n_dims:
            raise ValueError(f'Expected {n_dims} action bins, got {len(ACTION_BINS)}.')
        #minimálne a maximálne hodnoty každej akcie
        lows = env.action_space[0].low.tolist()
        highs = env.action_space[0].high.tolist()
        grids = [
            np.linspace(float(low), float(high), int(count), dtype=float)
            for low, high, count in zip(lows, highs, ACTION_BINS)#rozdelí interval na rovnomerné body
        ]
        self.joint_actions = list(iterproduct(*grids))
        #všetky možné kombinácie

    @property
    def action_count(self) -> int:
        return len(self.joint_actions)

    def decode(self, action_index: int) -> list[float]:
        return list(self.joint_actions[action_index])


class TabularQLearning:
    def __init__(
        self,
        env: CityLearnEnv,
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        adaptive_patience: int,
        adaptive_epsilon_boost: float,
        adaptive_min_improvement: float,
        random_seed: int,
    ):
        self.observation_discretizer = ObservationDiscretizer(env)
        self.action_discretizer = ActionDiscretizer(env)
        self.learning_rate = float(learning_rate)#(α) ako rýchlo sa mení Q-hodnota
        self.discount_factor = float(discount_factor)#(γ) ako veľmi agent rieši budúcnosť
        self.epsilon = float(epsilon)
        self.epsilon_init = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.adaptive_patience = int(adaptive_patience)
        self.adaptive_epsilon_boost = float(adaptive_epsilon_boost)
        self.adaptive_min_improvement = float(adaptive_min_improvement)
        self.episode_index = 0
        self.best_rolling_reward = -np.inf
        self.episodes_without_improvement = 0
        self.random_state = np.random.RandomState(random_seed)
        self.q_table = np.zeros(
            (self.observation_discretizer.state_count, self.action_discretizer.action_count),
            dtype=np.float32,
        )
        self.last_states: list[int] = []
        self.last_actions: list[int] = []

    def reset(self) -> None:
        self.last_states = []
        self.last_actions = []

    def act_train(self, observations: list[list[float]]) -> list[list[float]]:
        self.last_states = []
        self.last_actions = []
        actions: list[list[float]] = []
        for observation in observations:
            state = self.observation_discretizer.encode(observation)
            if self.random_state.rand() < self.epsilon:
                action_index = int(self.random_state.randint(self.action_discretizer.action_count))
            else:
                action_index = int(np.argmax(self.q_table[state]))
            self.last_states.append(state)
            self.last_actions.append(action_index)
            actions.append(self.action_discretizer.decode(action_index))
        return actions

    def act_eval(self, observations: list[list[float]]) -> list[list[float]]:
        actions: list[list[float]] = []
        for observation in observations:
            state = self.observation_discretizer.encode(observation)
            action_index = int(np.argmax(self.q_table[state]))
            actions.append(self.action_discretizer.decode(action_index))
        return actions

    def update(self, rewards: list[float], next_observations: list[list[float]], terminated: bool) -> None:
        for state, action, reward, next_observation in zip(self.last_states, self.last_actions, rewards, next_observations):
            next_state = self.observation_discretizer.encode(next_observation)
            best_next = 0.0 if terminated else float(np.max(self.q_table[next_state]))
            td_target = float(reward) + self.discount_factor * best_next
            self.q_table[state, action] += self.learning_rate * (td_target - float(self.q_table[state, action]))

    def finish_episode(self, reward_history: list[float]) -> None:
        self.episode_index += 1
        self.epsilon = max(self.epsilon_min, self.epsilon_init * np.exp(-self.epsilon_decay * self.episode_index))
        if len(reward_history) < 5:
            return

        rolling_reward = float(np.mean(reward_history[-5:]))
        if rolling_reward > self.best_rolling_reward + self.adaptive_min_improvement:
            self.best_rolling_reward = rolling_reward
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1

        if self.episodes_without_improvement >= self.adaptive_patience:
            self.epsilon = min(0.35, self.epsilon + self.adaptive_epsilon_boost)
            self.episodes_without_improvement = 0


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


def make_env(schema_path: Path, building_names: list[str], random_seed: int, reward_function=WeatherOccupancyReward) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=False,
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
        previous = arr[index - (2 * window) + 1:index - window + 1]
        current = arr[index - window + 1:index + 1]
        previous_mean = float(np.mean(previous))
        current_mean = float(np.mean(current))
        scale = max(1.0, abs(previous_mean))
        stable_mean = abs(current_mean - previous_mean) / scale <= tolerance
        stable_std = float(np.std(current)) / max(1.0, abs(current_mean)) <= tolerance * 1.5
        if stable_mean and stable_std:
            return index + 1

    return None


def train(agent: TabularQLearning, env: CityLearnEnv, episodes: int) -> TrainingTrace:
    episode_rewards: list[float] = []
    epsilons: list[float] = []
    start_time = time.perf_counter()#uloží aktuálny presný čas (vysokorozlíšený časovač)

    for _ in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        terminated = False
        episode_reward = 0.0

        while not terminated:
            actions = agent.act_train(observations)
            next_observations, rewards, terminated, _, _ = env.step(actions)
            agent.update(rewards, next_observations, terminated)
            observations = next_observations
            episode_reward += float(np.sum(rewards))

        episode_rewards.append(episode_reward)
        agent.finish_episode(episode_rewards)
        epsilons.append(agent.epsilon)

    return TrainingTrace(
        episode_rewards=episode_rewards,
        epsilons=epsilons,
        training_seconds=time.perf_counter() - start_time,
        stability_episode=estimate_stability_episode(episode_rewards),
    )

#
def evaluate_policy(env: CityLearnEnv, act_fn) -> PolicyRun:
    observations, _ = env.reset()
    terminated = False
    reward_trace: list[float] = []
    cumulative_reward = 0.0

    while not terminated:
        actions = act_fn(observations)
        observations, rewards, terminated, _, _ = env.step(actions)
        step_reward = float(np.sum(rewards))
        reward_trace.append(step_reward)
        cumulative_reward += step_reward

    base_env = env.unwrapped
    buildings = base_env.buildings
    kpis = base_env.evaluate() #spočíta finálne metriky za celú simuláciu
    building_names = [building.name for building in buildings]
    discomfort_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_proportion')]
    discomfort = float(discomfort_rows['value'].mean()) if not discomfort_rows.empty else 0.0

    aggregate_net = np.zeros(len(reward_trace), dtype=float)
    trajectory_data: dict[str, np.ndarray | list[float]] = {
        'time_step': np.arange(len(reward_trace)),
        'reward': reward_trace,
        'cumulative_reward': np.cumsum(reward_trace),
    }
    for building in buildings:
        net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)[:len(reward_trace)]#získanie spotreby budovy
        aggregate_net += net_consumption #postupne sa sčíta spotreba všetkých budov
        trajectory_data[f'grid_import_{building.name}_kwh'] = np.clip(net_consumption, 0.0, None) #nový kľúč do slovníka (odstráni záporné hodnoty)

    trajectory_data['grid_import_kwh'] = np.clip(aggregate_net, 0.0, None)
    trajectory_data['cumulative_grid_import_kwh'] = np.cumsum(np.clip(aggregate_net, 0.0, None)) #

    return PolicyRun(
        result=ExperimentResult(
            policy='',
            total_grid_import_kwh=float(np.sum(np.clip(aggregate_net, 0.0, None))),
            total_net_consumption_kwh=float(np.sum(aggregate_net)),
            discomfort_proportion=discomfort,
            cumulative_reward=cumulative_reward,
        ),
        trajectory=pd.DataFrame(trajectory_data),
        kpis=kpis,
    )


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

    fig.suptitle(f'3 budovy: Fixed vs {len(labels) - 1} reward variantov Q-learningu')
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
    axes[0, 0].plot(learned_slice['time_step'], learned_slice['reward'], lw=1.9, color='#1d3557', label='Q-learning')
    axes[0, 0].set_title('Step reward in time')
    axes[0, 0].set_xlabel('Time step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(fixed_slice['time_step'], fixed_slice['cumulative_reward'], '--', lw=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[0, 1].plot(learned_slice['time_step'], learned_slice['cumulative_reward'], lw=1.9, color='#1d3557', label='Q-learning')
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
    axes[1, 1].plot(learned_slice['time_step'], learned_slice['cumulative_grid_import_kwh'], lw=1.9, color='#1d3557', label='Q-learning')
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
    fixed_policy = FixedPolicy(dhw_action=0.0, electrical_action=0.0, cooling_action=baseline_cooling)
    fixed_run = evaluate_policy(fixed_env, fixed_policy.act)
    fixed_run.result.policy = f'Fixed(cool={baseline_cooling:.2f})'
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    learned_runs: list[tuple[str, str, PolicyRun, TrainingTrace, TabularQLearning]] = []
    for key, display_name, reward_cls in REWARD_CONFIGS:
        train_env = make_env(schema_path, building_names, random_seed, reward_function=reward_cls)
        agent = TabularQLearning(
            train_env,
            learning_rate=0.15,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.03,
            adaptive_patience=8,
            adaptive_epsilon_boost=0.08,
            adaptive_min_improvement=0.01,
            random_seed=random_seed,
        )
        trace = train(agent, train_env, episodes)

        eval_env = make_env(schema_path, building_names, random_seed, reward_function=reward_cls)
        run = evaluate_policy(eval_env, agent.act_eval)
        run.result.policy = f'Q-learning ({display_name})'
        run.result.training_seconds = trace.training_seconds
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

#
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Jednoduchsia verzia Q-learning experimentu s 12 reward variantmi.')
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument('--buildings', nargs='+', default=DEFAULT_BUILDINGS)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--baseline-cooling', type=float, default=0.5)
    parser.add_argument('--random-seed', type=int, default=7)
    parser.add_argument('--comparison-horizon', type=int, default=719)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(
        schema_path=args.schema,
        building_names=args.buildings,
        episodes=args.episodes,
        baseline_cooling=args.baseline_cooling,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        comparison_horizon=args.comparison_horizon,
    )
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
