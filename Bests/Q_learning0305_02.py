from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1' / 'schema.json'
DEFAULT_BUILDINGS = ['Building_1', 'Building_2', 'Building_3']
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning0205_02'
DEFAULT_EPISODES = 50
DEFAULT_BASELINE_COOLING = 0.5
DEFAULT_RANDOM_SEED = 7
DEFAULT_COMPARISON_HORIZON = 719

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
    def __init__(
        self,
        env_metadata,
        expensive_import_weight: float = 1.25,
        cheap_charge_bonus_weight: float = 2.0,
        expensive_export_bonus_weight: float = 2.25,
        reserve_weight: float = 2.0,
        premature_discharge_penalty_weight: float = 1.25,
        target_soc_when_price_rises: float = 0.7,
        price_reference: float = 0.25,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.expensive_import_weight = float(expensive_import_weight)
        self.cheap_charge_bonus_weight = float(cheap_charge_bonus_weight)
        self.expensive_export_bonus_weight = float(expensive_export_bonus_weight)
        self.reserve_weight = float(reserve_weight)
        self.premature_discharge_penalty_weight = float(premature_discharge_penalty_weight)
        self.target_soc_when_price_rises = float(target_soc_when_price_rises)
        self.price_reference = float(price_reference)
        self.prev_avg_soc: dict[int, float] = {}

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for i, obs in enumerate(observations):
            net_consumption = float(obs['net_electricity_consumption'])
            grid_import = max(net_consumption, 0.0)
            export = max(-net_consumption, 0.0)
            price = max(obs.get('electricity_pricing', 0.0), 0.0)
            next_price = max(obs.get('electricity_pricing_predicted_1', price), 0.0)
            avg_soc = (obs.get('dhw_storage_soc', 0.5) + obs.get('electrical_storage_soc', 0.5)) / 2.0
            prev_avg_soc = self.prev_avg_soc.get(i, avg_soc)
            soc_change = avg_soc - prev_avg_soc
            self.prev_avg_soc[i] = avg_soc

            price_level = min(price / self.price_reference, 2.0)
            cheapness = max(1.0 - price_level, 0.0)
            future_rise = max(next_price - price, 0.0)

            expensive_import_penalty = grid_import * self.expensive_import_weight * max(price_level - 0.6, 0.0)
            reserve_gap = max(self.target_soc_when_price_rises - avg_soc, 0.0)
            reserve_penalty = self.reserve_weight * reserve_gap * (price_level + 4.0 * future_rise)
            cheap_charge_bonus = self.cheap_charge_bonus_weight * cheapness * max(soc_change, 0.0)
            expensive_export_bonus = self.expensive_export_bonus_weight * price_level * export
            premature_discharge_penalty = self.premature_discharge_penalty_weight * max(-soc_change, 0.0) * (cheapness + 3.0 * future_rise)

            rewards.append(-(expensive_import_penalty + reserve_penalty + premature_discharge_penalty) + cheap_charge_bonus + expensive_export_bonus)
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
            cooling_delta = max(obs.get('indoor_dry_bulb_temperature_cooling_delta', 0.0), 0.0)#o kolko je teplota vo vnutri mimo komfortu 
            occ_factor = 1.0 + self.occupancy_weight * occ_present
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            discomfort_factor = 1.0 + self.occupied_discomfort_multiplier * occ_present #neviem mozno sa dalo lepsiee ale ked ludia potom vacsi disc
            rewards.append(-(grid_import * occ_factor * weather_factor + self.discomfort_weight * cooling_delta * discomfort_factor))
        return [sum(rewards)] if self.central_agent else rewards


class PeakShavingReward(RewardFunction):
    def calculate(self, observations: list[dict]) -> list[float]:
        rewards = [-(max(obs['net_electricity_consumption'], 0.0) ** 2) for obs in observations]
        return [sum(rewards)] if self.central_agent else rewards
#maly odber mala penalta extremne velky odber extrene velka penalta vdaka **2

class SolarAlignmentReward(RewardFunction):
    def __init__(
        self,
        env_metadata,
        solar_import_weight: float = 0.35,
        storage_reserve_weight: float = 2.0,
        self_consumption_bonus_weight: float = 1.0,
        export_penalty_weight: float = 1.0,
        charging_bonus_weight: float = 0.75,
        low_soc_target: float = 0.8,
        solar_reference_irradiance: float = 750.0,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.solar_import_weight = float(solar_import_weight)
        self.storage_reserve_weight = float(storage_reserve_weight)
        self.self_consumption_bonus_weight = float(self_consumption_bonus_weight)
        self.export_penalty_weight = float(export_penalty_weight)
        self.charging_bonus_weight = float(charging_bonus_weight)
        self.low_soc_target = float(low_soc_target)
        self.solar_reference_irradiance = float(solar_reference_irradiance)
        self.prev_avg_soc: dict[int, float] = {}

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for i, obs in enumerate(observations):
            net_consumption = float(obs['net_electricity_consumption'])
            grid_import = max(net_consumption, 0.0)
            export = max(-net_consumption, 0.0)
            diffuse = max(obs.get('diffuse_solar_irradiance_predicted_1', 0.0), 0.0)
            direct = max(obs.get('direct_solar_irradiance_predicted_1', 0.0), 0.0)
            solar_level = min((diffuse + direct) / self.solar_reference_irradiance, 1.0)
            avg_soc = (obs.get('dhw_storage_soc', 0.5) + obs.get('electrical_storage_soc', 0.5)) / 2.0
            prev_avg_soc = self.prev_avg_soc.get(i, avg_soc)
            soc_change = avg_soc - prev_avg_soc
            self.prev_avg_soc[i] = avg_soc

            target_soc = 0.35 + (self.low_soc_target - 0.35) * solar_level
            low_soc_gap = max(target_soc - avg_soc, 0.0)
            import_penalty = grid_import * (1.0 + self.solar_import_weight * solar_level)
            missed_charge_penalty = self.storage_reserve_weight * solar_level * low_soc_gap * (1.0 if soc_change <= 0.0 else 0.0)
            wasted_solar_penalty = self.export_penalty_weight * solar_level * export * max(target_soc - avg_soc, 0.0)
            charging_bonus = self.charging_bonus_weight * solar_level * max(soc_change, 0.0)
            self_consumption_bonus = self.self_consumption_bonus_weight * solar_level * max(1.0 - export, 0.0) * max(avg_soc - prev_avg_soc, 0.0)

            rewards.append(-(import_penalty + missed_charge_penalty + wasted_solar_penalty) + charging_bonus + self_consumption_bonus)
        return [sum(rewards)] if self.central_agent else rewards


class StorageManagementReward(RewardFunction):
    #keď je elektrina drahá, je zlé nemať nič “v zásobe”
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
   # aby nerobil prudké prepínanie
    def __init__(self, env_metadata, ramping_weight: float = 0.5, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.ramping_weight = float(ramping_weight)
        self.prev_imports: dict[int, float] = {}

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for i, obs in enumerate(observations):
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            prev = self.prev_imports.get(i, grid_import)
            self.prev_imports[i] = grid_import
            rewards.append(-(grid_import + self.ramping_weight * abs(grid_import - prev)))
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
    # ('weather', 'Weather reward', WeatherOccupancyReward),
    # ('energy', 'Energy-only reward', GridImportOnlyReward),
    ('pricing', 'Pricing reward', PricingAwareReward),
    # ('comfort', 'Comfort reward', ComfortAwareReward),
    # ('peak', 'Peak-shaving reward', PeakShavingReward),
    # ('solar', 'Solar reward', SolarAlignmentReward),
    # ('storage', 'Storage reward', StorageManagementReward),
    # ('ramping', 'Ramping reward', RampingPenaltyReward),
    # ('tou', 'TimeOfUse reward', TimeOfUseReward),
    # ('selfsuff', 'SelfSufficiency reward', SelfSufficiencyReward),
    # ('combined', 'Combined reward', CombinedMultiObjectiveReward),
    # ('nightpre', 'NightPrecharge reward', NightPrechargeReward),
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
        alfa: float,
        gamma: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decrease: float,
        max_num_of_ep_without_imprv: int,
        epsilon_boost: float,
        adaptive_min_improvement: float,
        random_seed: int,
    ):
        self.observation_discretizer = ObservationDiscretizer(env)
        self.action_discretizer = ActionDiscretizer(env)
        self.alfa = float(alfa)#(α) ako rýchlo sa mení Q-hodnota
        self.gamma = float(gamma)#(γ) ako veľmi agent rieši budúcnosť
        self.epsilon = float(epsilon)
        self.ep_init = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decrease = float(epsilon_decrease)
        self.max_num_of_ep_without_imprv = int(max_num_of_ep_without_imprv)
        self.epsilon_boost = float(epsilon_boost)
        self.adaptive_min_improvement = float(adaptive_min_improvement)
        self.episode_idx = 0
        self.best_rolling_reward = -np.inf
        self.ep_without_imprv = 0
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
            td_target = float(reward) + self.gamma * best_next
            self.q_table[state, action] += self.alfa * (td_target - float(self.q_table[state, action]))
            #Bellmanova rovnica
            #$$\text{Cieľ} = \text{Odmena} + (\text{Zľava} \times \text{Budúci potenciál})$$
            #$$\text{Chyba} = \text{Cieľ} - \text{Stará hodnota v tabuľke}$$
            #$$\text{Nová hodnota} = \text{Stará hodnota} + (\text{Rýchlosť učenia} \times \text{Chyba})$$

    def finish_episode(self, reward_history: list[float]) -> None:
        self.episode_idx += 1
        self.epsilon = max(self.epsilon_min, self.ep_init * np.exp(-self.epsilon_decrease * self.episode_idx))#exp(-x) vytvorí peknú hladkú krivku, 
        if len(reward_history) < 5:
            return

        rolling_reward = float(np.mean(reward_history[-5:]))
        if rolling_reward > self.best_rolling_reward + self.adaptive_min_improvement:
            self.best_rolling_reward = rolling_reward
            self.ep_without_imprv = 0
        else:
            self.ep_without_imprv += 1

        if self.ep_without_imprv >= self.max_num_of_ep_without_imprv:
            self.epsilon = min(0.35, self.epsilon + self.epsilon_boost)
            self.ep_without_imprv = 0


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
    epsilons: list[float]
    train_sec: float
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

#
def estimate_stability_episode(rewards: list[float], window: int = 10, tolerance: float = 0.03) -> int | None:
    if len(rewards) < window * 2:
        return None

    arr = np.asarray(rewards, dtype=float) #Prevedie zoznam odmien na NumPy pole
    for i in range((window * 2) - 1, len(arr)):
        prev = arr[i - (2 * window) + 1:i - window + 1]
        curr = arr[i - window + 1:i + 1]
        prev_mean = float(np.mean(prev))
        curr_mean = float(np.mean(curr))
        scale = max(1.0, abs(prev_mean))
        stable_mean = abs(curr_mean - prev_mean) / scale <= tolerance
        stable_std = float(np.std(curr)) / max(1.0, abs(curr_mean)) <= tolerance * 2 #std = smerodajná odchýlka
        if stable_mean and stable_std:
            return i + 1

    return None

#
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

        episode_rewards.append(episode_reward) # či sa agent medzi epizódami učí lepšie alebo horšie
        agent.finish_episode(episode_rewards)
        epsilons.append(agent.epsilon)

    return TrainingTrace(
        episode_rewards=episode_rewards,
        epsilons=epsilons,
        train_sec=time.perf_counter() - start_time,
        stability_episode=estimate_stability_episode(episode_rewards),
    )

#
def eval_agent(env: CityLearnEnv, act_fn) -> PolicyRun:
    observations, _ = env.reset()
    terminated = False
    reward_trace: list[float] = []
    all_rewards = 0.0

    while not terminated:
        actions = act_fn(observations)
        observations, rewards, terminated, _, _ = env.step(actions)
        step_reward = float(np.sum(rewards))
        reward_trace.append(step_reward)
        all_rewards += step_reward

    base_env = env.unwrapped
    buildings = base_env.buildings
    kpis = base_env.evaluate() #spočíta finálne metriky za celú simuláciu
# - `cost_function` je názov sledovanej metriky (napr. celkové náklady, emisie CO2, tepelný komfort),
# - `value` je jej výsledná hodnota 
#   hodnota 0 môže predstavovať ideálny stav a NaN znamená, že metrika nebola vypočítaná alebo chýbajú dáta),
# - `name` určuje, na aký objekt sa metrika vzťahuje (napr. celý District alebo konkrétna budova ako Building_3),
# - `level` označuje úroveň detailu (district = agregované za celý systém, building = konkrétna budova).
    print(kpis)
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
        net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)[:len(reward_trace)]#získanie spotreby budovy
        aggregate_net += net_consumption #postupne sa sčíta spotreba všetkých budov
        trajectory_data[f'grid_import_{building.name}_kwh'] = np.clip(net_consumption, 0.0, None) #nový kľúč do slovníka (odstráni záporné hodnoty)
        trajectory_data[f'export_{building.name}_kwh'] = np.clip(-net_consumption, 0.0, None)

    grid_import = np.clip(aggregate_net, 0.0, None)
    export = np.clip(-aggregate_net, 0.0, None)
    trajectory_data['grid_import_kwh'] = grid_import
    trajectory_data['export_kwh'] = export
    trajectory_data['cumulative_grid_import_kwh'] = np.cumsum(grid_import) #
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

#zoberie zoznam výsledkov experimentov a premení ho na peknú tabuľku tiez mozno odstranit
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
    fixed_run = eval_agent(fixed_env, fixed_policy.act)
    fixed_run.result.policy = f'Fixed(cool={baseline_cooling:.2f})'
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    learned_runs: list[tuple[str, str, PolicyRun, TrainingTrace, TabularQLearning]] = []
    for key, display_name, reward_cls in REWARD_CONFIGS:
        print(f'Running reward: {display_name}')
        train_env = make_env(schema_path, building_names, random_seed, reward_function=reward_cls)
        agent = TabularQLearning(
            train_env,
            alfa=0.15,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decrease=0.03,
            max_num_of_ep_without_imprv=8,
            epsilon_boost=0.08,
            adaptive_min_improvement=0.01,
            random_seed=random_seed,
        )
        trace = train(agent, train_env, episodes)

        eval_env = make_env(schema_path, building_names, random_seed, reward_function=reward_cls)
        run = eval_agent(eval_env, agent.act_eval)
        run.result.policy = f'Q-learning ({display_name})'
        run.result.train_sec = trace.train_sec
        run.result.stability_episode = trace.stability_episode
        run.result.last_10_episode_reward_mean = float(np.mean(trace.episode_rewards[-10:])) if trace.episode_rewards else None
        if fixed_run.result.total_grid_import_kwh > 0.0:
            run.result.savings_vs_fixed_pct = 100.0 * (fixed_run.result.total_grid_import_kwh - run.result.total_grid_import_kwh) / fixed_run.result.total_grid_import_kwh

        results.append(run.result)
        learned_runs.append((key, display_name, run, trace, agent))

    results_frame = build_results_frame(results).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)

    for key, _display_name, run, trace, agent in learned_runs:
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
