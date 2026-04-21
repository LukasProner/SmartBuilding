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
DEFAULT_BUILDINGS = ['Building_1', 'Building_2', 'Building_3']
DEFAULT_SEEDS = [7]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning1804_05'

# ── Observations ─────────────────────────────────────────────────────────────
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
ENGINEERED_FEATURES = list(OBSERVATION_BIN_SIZES.keys())

# ── Akcie ────────────────────────────────────────────────────────────────────
ACTIVE_ACTIONS = ['dhw_storage', 'electrical_storage', 'cooling_device']
ACTION_BIN_COUNTS = [3, 3, 3]

# ── Farby pre grafy (13 politík: 1 fixed + 12 Q-learning) ────────────────────
POLICY_COLORS = [
    '#9aa0a6',   # Fixed – sivá
    '#1d3557',   # Weather – tmavomodrá
    '#2a9d8f',   # Energy – teal
    '#e76f51',   # Pricing – koralová
    '#6d597a',   # Comfort – fialová
    '#264653',   # PeakShaving – tmavý petrol
    '#e9c46a',   # Solar – zlatá
    '#f4a261',   # Storage – piesková
    '#606c38',   # Ramping – olivová
    '#bc6c25',   # TimeOfUse – hnedá
    '#023047',   # SelfSufficiency – polnočná modrá
    '#d62828',   # Combined – červená
    '#219ebc',   # NightPrecharge – nebeská modrá
]


# ══════════════════════════════════════════════════════════════════════════════
# REWARD FUNKCIE (12 variantov)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. WeatherOccupancyReward ─────────────────────────────────────────────────
class WeatherOccupancyReward(RewardFunction):
    """Penalizuje grid import × obsadenosť × horúce počasie."""

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


# ── 2. GridImportOnlyReward ──────────────────────────────────────────────────
class GridImportOnlyReward(RewardFunction):
    """Čistá lineárna penalizácia grid importu."""

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            reward_list.append(-grid_import)
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 3. PricingAwareReward ────────────────────────────────────────────────────
class PricingAwareReward(RewardFunction):
    """Penalizuje grid import × aktuálna cena elektrickej energie."""

    def __init__(self, env_metadata, price_weight: float = 2.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.price_weight = float(price_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            price = max(obs.get('electricity_pricing', 0.0), 0.0)
            reward_list.append(-(grid_import * (1.0 + self.price_weight * price)))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 4. ComfortAwareReward ────────────────────────────────────────────────────
class ComfortAwareReward(RewardFunction):
    """Penalizuje grid import + discomfort (prehrievanie) pri obsadení budovy."""

    def __init__(self, env_metadata, occupancy_weight: float = 1.0, hot_weather_weight: float = 0.05,
                 discomfort_weight: float = 6.0, occupied_discomfort_multiplier: float = 2.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.occupancy_weight = float(occupancy_weight)
        self.hot_weather_weight = float(hot_weather_weight)
        self.discomfort_weight = float(discomfort_weight)
        self.occupied_discomfort_multiplier = float(occupied_discomfort_multiplier)

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
            cooling_delta = max(obs.get('indoor_dry_bulb_temperature_cooling_delta', 0.0), 0.0)
            occ_factor = 1.0 + self.occupancy_weight * occ_present
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            discomfort_factor = 1.0 + self.occupied_discomfort_multiplier * occ_present
            reward_list.append(-(
                grid_import * occ_factor * weather_factor
                + self.discomfort_weight * cooling_delta * discomfort_factor
            ))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 5. PeakShavingReward ─────────────────────────────────────────────────────
class PeakShavingReward(RewardFunction):
    """Kvadratická penalizácia grid importu.
    Malý odber = malá penalta, veľký odber = obrovská penalta.
    Núti agenta vyhladzovať špičky namiesto presúvania celej spotreby."""

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            reward_list.append(-(grid_import ** 2))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 6. SolarAlignmentReward ──────────────────────────────────────────────────
class SolarAlignmentReward(RewardFunction):
    """Penalizuje grid import menej keď je veľa solárnej energie (lacná/čistá),
    a viac keď slnko nesvieti (agent by mal využiť zásoby)."""

    def __init__(self, env_metadata, solar_weight: float = 0.3, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.solar_weight = float(solar_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            diffuse = max(obs.get('diffuse_solar_irradiance_predicted_1', 0.0), 0.0)
            direct = max(obs.get('direct_solar_irradiance_predicted_1', 0.0), 0.0)
            solar_total = diffuse + direct
            normalized_solar = min(solar_total / 800.0, 1.0)
            solar_factor = 1.0 - self.solar_weight * normalized_solar
            reward_list.append(-(grid_import * solar_factor))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 7. StorageManagementReward ───────────────────────────────────────────────
class StorageManagementReward(RewardFunction):
    """Penalizuje grid import + zlé SOC stavy:
    - nízky SOC počas drahých hodín (nepreparoval sa)
    - plný SOC počas lacných hodín (nemôže nabiť lacno)."""

    def __init__(self, env_metadata, soc_penalty_weight: float = 2.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.soc_penalty_weight = float(soc_penalty_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            price = max(obs.get('electricity_pricing', 0.0), 0.0)
            dhw_soc = obs.get('dhw_storage_soc', 0.5)
            elec_soc = obs.get('electrical_storage_soc', 0.5)
            avg_soc = (dhw_soc + elec_soc) / 2.0
            # Drahá hodina + nízky SOC = penalta (nemal zásoby)
            # Lacná hodina + plný SOC = penalta (nemôže nabiť lacno)
            if price > 0.3:
                soc_waste = max(0.5 - avg_soc, 0.0)
            else:
                soc_waste = max(avg_soc - 0.8, 0.0) * 0.5
            reward_list.append(-(grid_import + self.soc_penalty_weight * soc_waste))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 8. RampingPenaltyReward ──────────────────────────────────────────────────
class RampingPenaltyReward(RewardFunction):
    """Penalizuje veľké skoky v grid importe medzi krokmi.
    Podporuje plynulý odber, čo je žiaduce pre distribučnú sieť."""

    def __init__(self, env_metadata, ramping_weight: float = 0.5, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.ramping_weight = float(ramping_weight)
        self._prev_imports: dict[int, float] = {}

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for i, obs in enumerate(observations):
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            prev = self._prev_imports.get(i, grid_import)
            ramp = abs(grid_import - prev)
            self._prev_imports[i] = grid_import
            reward_list.append(-(grid_import + self.ramping_weight * ramp))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 9. TimeOfUseReward ───────────────────────────────────────────────────────
class TimeOfUseReward(RewardFunction):
    """Tvrdý rozvrh podľa hodiny dňa:
    - špička 17:00–21:00 → 3× penalizácia
    - noc 22:00–06:00 → 0.5× penalizácia
    - zvyšok → 1× penalizácia
    Nezávisí na electricity_pricing – čisto časový signál."""

    def __init__(self, env_metadata, peak_multiplier: float = 3.0,
                 night_discount: float = 0.5, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.peak_multiplier = float(peak_multiplier)
        self.night_discount = float(night_discount)
        self._step = 0

    def calculate(self, observations: list[dict]) -> list[float]:
        hour = self._step % 24
        self._step += 1
        if 17 <= hour <= 21:
            multiplier = self.peak_multiplier
        elif hour >= 22 or hour <= 6:
            multiplier = self.night_discount
        else:
            multiplier = 1.0
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            reward_list.append(-(grid_import * multiplier))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 10. SelfSufficiencyReward ────────────────────────────────────────────────
class SelfSufficiencyReward(RewardFunction):
    """Penalizuje grid import, ale navyše odmieňa čistý export
    (spätné dodávanie do siete). Agent sa snaží maximalizovať sebestačnosť."""

    def __init__(self, env_metadata, export_bonus: float = 0.5, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.export_bonus = float(export_bonus)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for obs in observations:
            net = obs['net_electricity_consumption']
            if net > 0:
                reward_list.append(-net)
            else:
                reward_list.append(self.export_bonus * abs(net))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 11. CombinedMultiObjectiveReward ─────────────────────────────────────────
class CombinedMultiObjectiveReward(RewardFunction):
    """Vážený súčet všetkých faktorov: grid import + cena + komfort + peak shaving.
    Jedno „najlepšie" nastavenie pre viacero cieľov."""

    def __init__(self, env_metadata, price_weight: float = 1.0,
                 comfort_weight: float = 3.0, peak_weight: float = 0.1, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.price_weight = float(price_weight)
        self.comfort_weight = float(comfort_weight)
        self.peak_weight = float(peak_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            price = max(obs.get('electricity_pricing', 0.0), 0.0)
            cooling_delta = max(obs.get('indoor_dry_bulb_temperature_cooling_delta', 0.0), 0.0)
            reward_list.append(-(
                grid_import
                + self.price_weight * grid_import * price
                + self.comfort_weight * cooling_delta
                + self.peak_weight * grid_import ** 2
            ))
        return [sum(reward_list)] if self.central_agent else reward_list


# ── 12. NightPrechargeReward ─────────────────────────────────────────────────
class NightPrechargeReward(RewardFunction):
    """Odmeňuje vysoký SOC počas špičky a nízku penaltu za import v noci
    (keď agent nabíja zásobníky). Kombinuje časový signál + SOC awareness."""

    def __init__(self, env_metadata, soc_bonus: float = 1.5,
                 peak_import_multiplier: float = 2.0, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.soc_bonus = float(soc_bonus)
        self.peak_import_multiplier = float(peak_import_multiplier)
        self._step = 0

    def calculate(self, observations: list[dict]) -> list[float]:
        hour = self._step % 24
        self._step += 1
        reward_list = []
        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            dhw_soc = obs.get('dhw_storage_soc', 0.5)
            elec_soc = obs.get('electrical_storage_soc', 0.5)
            avg_soc = (dhw_soc + elec_soc) / 2.0
            reward = -grid_import
            if 0 <= hour <= 6:
                # Noc: znížená penalta za import + bonus za plné zásobníky
                reward = -grid_import * 0.4 + self.soc_bonus * avg_soc
            elif 17 <= hour <= 21:
                # Špička: zvýšená penalta + bonus za plné zásobníky
                reward = -grid_import * self.peak_import_multiplier + self.soc_bonus * avg_soc * 0.5
            reward_list.append(reward)
        return [sum(reward_list)] if self.central_agent else reward_list


# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURÁCIA REWARD VARIANTOV (kľúč, zobrazený názov, trieda)
# ══════════════════════════════════════════════════════════════════════════════
REWARD_CONFIGS: list[tuple[str, str, type]] = [
    ('weather',    'Weather reward',        WeatherOccupancyReward),
    # ('energy',     'Energy-only reward',     GridImportOnlyReward),
    # ('pricing',    'Pricing reward',         PricingAwareReward),
    # ('comfort',    'Comfort reward',         ComfortAwareReward),
    # ('peak',       'Peak-shaving reward',    PeakShavingReward),
    # ('solar',      'Solar reward',           SolarAlignmentReward),
    # ('storage',    'Storage reward',         StorageManagementReward),
    # ('ramping',    'Ramping reward',         RampingPenaltyReward),
    # ('tou',        'TimeOfUse reward',       TimeOfUseReward),
    # ('selfsuff',   'SelfSufficiency reward', SelfSufficiencyReward),
    # ('combined',   'Combined reward',        CombinedMultiObjectiveReward),
    # ('nightpre',   'NightPrecharge reward',  NightPrechargeReward),
]


# ══════════════════════════════════════════════════════════════════════════════
# POLITIKY, DISKRETIZÁCIA, AGENT
# ══════════════════════════════════════════════════════════════════════════════

class FixedPolicy:
    def __init__(self, dhw_action: float = 0.0, electrical_action: float = 0.0, cooling_action: float = 0.5):
        self.actions = [float(dhw_action), float(electrical_action), float(cooling_action)]

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
            'price_now', 'price_trend_01',
            'dhw_storage_soc', 'electrical_storage_soc',
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
        p0_lo, p0_hi = low_by_name['electricity_pricing'], high_by_name['electricity_pricing']
        p1_lo, p1_hi = low_by_name['electricity_pricing_predicted_1'], high_by_name['electricity_pricing_predicted_1']
        s1_lo, s1_hi = d1_lo + r1_lo, d1_hi + r1_hi
        s2_lo, s2_hi = d2_lo + r2_lo, d2_hi + r2_hi

        feature_lows = [t1_lo, t2_lo - t1_hi, s1_lo, s2_lo - s1_hi, p0_lo, p1_lo - p0_hi,
                        low_by_name['dhw_storage_soc'], low_by_name['electrical_storage_soc'], 0.0]
        feature_highs = [t1_hi, t2_hi - t1_lo, s1_hi, s2_hi - s1_lo, p0_hi, p1_hi - p0_lo,
                         high_by_name['dhw_storage_soc'], high_by_name['electrical_storage_soc'], 1.0]
        self.edges = [np.linspace(float(lo), float(hi), c + 1)[1:-1]
                      for lo, hi, c in zip(feature_lows, feature_highs, self.bin_counts)]

    def encode(self, observation: list[float]) -> int:
        idx = self.index_by_name
        t1 = float(observation[idx['outdoor_dry_bulb_temperature_predicted_1']])
        t2 = float(observation[idx['outdoor_dry_bulb_temperature_predicted_2']])
        d1 = float(observation[idx['diffuse_solar_irradiance_predicted_1']])
        d2 = float(observation[idx['diffuse_solar_irradiance_predicted_2']])
        r1 = float(observation[idx['direct_solar_irradiance_predicted_1']])
        r2 = float(observation[idx['direct_solar_irradiance_predicted_2']])
        p0 = float(observation[idx['electricity_pricing']])
        p1 = float(observation[idx['electricity_pricing_predicted_1']])
        dhw = float(observation[idx['dhw_storage_soc']])
        elec = float(observation[idx['electrical_storage_soc']])
        occ = 1.0 if float(observation[idx['occupant_count']]) > 0.0 else 0.0
        vals = [t1, t2 - t1, d1 + r1, (d2 + r2) - (d1 + r1), p0, p1 - p0, dhw, elec, occ]
        digits = [int(np.digitize(float(v), e, right=False)) for v, e in zip(vals, self.edges)]
        return int(np.ravel_multi_index(tuple(digits), self.state_shape))


class MultiActionDiscretizer:
    def __init__(self, env: CityLearnEnv, bin_counts_per_action: list[int]):
        n_dims = env.action_space[0].shape[0]
        if len(bin_counts_per_action) != n_dims:
            raise ValueError(f'Expected {n_dims} bin counts, got {len(bin_counts_per_action)}.')
        lows = env.action_space[0].low.tolist()
        highs = env.action_space[0].high.tolist()
        self.value_grids = [np.linspace(float(lo), float(hi), int(n), dtype=float)
                            for lo, hi, n in zip(lows, highs, bin_counts_per_action)]
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
        ns = self.observation_discretizer.state_count
        na = self.action_discretizer.action_count
        print(f'  [Q-tabuľka] stavy={ns}, joint_akcie={na}, '
              f'veľkosť={ns * na:,} buniek ({ns * na * 4 / 1024 / 1024:.1f} MB)', flush=True)

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
        for s, a, r, nobs in zip(self.last_state_indices, self.last_action_indices, rewards, next_observations):
            ns = self.observation_discretizer.encode(nobs)
            best_next = 0.0 if terminated else float(np.max(self.q_table[ns]))
            td_target = float(r) + self.discount_factor * best_next
            self.q_table[s, a] += self.learning_rate * (td_target - float(self.q_table[s, a]))

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


# ══════════════════════════════════════════════════════════════════════════════
# DÁTOVÉ TRIEDY
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# POMOCNÉ FUNKCIE
# ══════════════════════════════════════════════════════════════════════════════

def make_env(schema_path: Path, building_names: list[str], random_seed: int,
             reward_function=WeatherOccupancyReward) -> CityLearnEnv:
    return CityLearnEnv(str(schema_path), central_agent=False, buildings=building_names,
                        active_observations=ACTIVE_OBSERVATIONS, active_actions=ACTIVE_ACTIONS,
                        reward_function=reward_function, random_seed=random_seed)


def train_q_learning(agent: OwnAdaptiveTabularQLearning, env: CityLearnEnv,
                     episodes: int, progress_every: int) -> TrainingTrace:
    episode_rewards: list[float] = []
    epsilons: list[float] = []
    t0 = time.perf_counter()
    for ep in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        terminated = False
        cum_reward = 0.0
        while not terminated:
            actions = agent.predict(observations, deterministic=False)
            next_obs, rewards, terminated, _, _ = env.step(actions)
            agent.update(rewards, next_obs, terminated)
            observations = next_obs
            cum_reward += float(np.sum(rewards))
        episode_rewards.append(cum_reward)
        agent.finish_episode(episode_rewards)
        epsilons.append(agent.epsilon)
        if progress_every > 0 and ((ep + 1) % progress_every == 0 or ep == 0 or ep + 1 == episodes):
            elapsed = time.perf_counter() - t0
            roll = float(np.mean(episode_rewards[-5:])) if len(episode_rewards) >= 5 else cum_reward
            print(f'  Episode {ep + 1}/{episodes} | reward={cum_reward:.2f} | rolling5={roll:.2f} | '
                  f'epsilon={agent.epsilon:.3f} | elapsed={elapsed:.1f}s', flush=True)
    return TrainingTrace(episode_rewards=episode_rewards, epsilons=epsilons,
                         training_seconds=time.perf_counter() - t0,
                         stability_episode=estimate_stability_episode(episode_rewards))


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
    building_names = [b.name for b in buildings]
    dc = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_proportion')]
    discomfort = float(dc['value'].mean()) if not dc.empty else 0.0
    agg_net = np.zeros(len(reward_trace), dtype=float)
    traj_data: dict = {
        'time_step': np.arange(len(reward_trace)),
        'reward': reward_trace,
        'cumulative_reward': np.cumsum(reward_trace),
    }
    for b in buildings:
        nc = np.asarray(b.net_electricity_consumption, dtype=float)[:len(reward_trace)]
        agg_net += nc
        traj_data[f'grid_import_{b.name}_kwh'] = np.clip(nc, 0.0, None)
    traj_data['grid_import_kwh'] = np.clip(agg_net, 0.0, None)
    traj_data['cumulative_grid_import_kwh'] = np.cumsum(np.clip(agg_net, 0.0, None))
    return PolicyRun(
        result=ExperimentResult(
            policy=agent.__class__.__name__,
            seed=None,
            total_grid_import_kwh=float(np.sum(np.clip(agg_net, 0.0, None))),
            total_net_consumption_kwh=float(np.sum(agg_net)),
            discomfort_proportion=discomfort,
            cumulative_reward=cumulative_reward,
        ),
        trajectory=pd.DataFrame(traj_data),
        kpis=kpis,
    )


def estimate_stability_episode(rewards: Sequence[float], window: int = 10, tolerance: float = 0.03) -> int | None:
    if len(rewards) < window * 2:
        return None
    arr = np.asarray(rewards, dtype=float)
    for i in range((window * 2) - 1, len(arr)):
        prev = arr[i - (2 * window) + 1:i - window + 1]
        curr = arr[i - window + 1:i + 1]
        pm, cm = float(np.mean(prev)), float(np.mean(curr))
        scale = max(1.0, abs(pm))
        if abs(cm - pm) / scale <= tolerance and float(np.std(curr)) / max(1.0, abs(cm)) <= tolerance * 1.5:
            return i + 1
    return None


def build_results_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        'policy': r.policy,
        'seed': r.seed,
        'grid_import_kwh': round(r.total_grid_import_kwh, 3),
        'net_consumption_kwh': round(r.total_net_consumption_kwh, 3),
        'discomfort_proportion': round(r.discomfort_proportion, 4),
        'cumulative_reward': round(r.cumulative_reward, 3),
        'savings_vs_fixed_pct': None if r.savings_vs_fixed_pct is None else round(r.savings_vs_fixed_pct, 3),
        'training_seconds': None if r.training_seconds is None else round(r.training_seconds, 2),
        'stability_episode': r.stability_episode,
        'last_10_episode_reward_mean': None if r.last_10_episode_reward_mean is None else round(r.last_10_episode_reward_mean, 3),
    } for r in results])


# ══════════════════════════════════════════════════════════════════════════════
# GRAFY
# ══════════════════════════════════════════════════════════════════════════════

def save_policy_comparison_figure(results_frame: pd.DataFrame, policy_runs: list[PolicyRun],
                                  output_path: Path) -> None:
    labels = results_frame['policy'].tolist()
    x = np.arange(len(labels))
    colors = POLICY_COLORS[:len(labels)]

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))

    axes[0, 0].bar(x, results_frame['grid_import_kwh'], color=colors)
    axes[0, 0].set_title('Total grid import', fontsize=12)
    axes[0, 0].set_ylabel('kWh')
    axes[0, 0].set_xticks(x, labels, rotation=35, ha='right', fontsize=8)
    axes[0, 0].grid(axis='y', alpha=0.25)

    axes[0, 1].bar(x, results_frame['savings_vs_fixed_pct'].fillna(0.0), color=colors)
    axes[0, 1].set_title('Savings vs fixed strategy', fontsize=12)
    axes[0, 1].set_ylabel('%')
    axes[0, 1].set_xticks(x, labels, rotation=35, ha='right', fontsize=8)
    axes[0, 1].grid(axis='y', alpha=0.25)

    axes[1, 0].bar(x, results_frame['discomfort_proportion'], color=colors)
    axes[1, 0].set_title('Discomfort proportion', fontsize=12)
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].set_xticks(x, labels, rotation=35, ha='right', fontsize=8)
    axes[1, 0].grid(axis='y', alpha=0.25)

    profile_hours = min(14 * 24, min(len(pr.trajectory) for pr in policy_runs))
    pidx = np.arange(profile_hours)
    for color, pr in zip(colors, policy_runs):
        profile = pr.trajectory.groupby(pr.trajectory['time_step'] % profile_hours)['grid_import_kwh'].mean()
        ls = '--' if pr.result.policy.startswith('Fixed') else '-'
        axes[1, 1].plot(pidx, profile.reindex(pidx, fill_value=np.nan).to_numpy(),
                        linestyle=ls, linewidth=1.8, color=color, label=pr.result.policy)
    axes[1, 1].set_title('Average grid import over 14-day profile', fontsize=12)
    axes[1, 1].set_xlabel('Hour in 14-day cycle')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].set_xticks(range(0, profile_hours + 1, 24))
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc='best', fontsize=7, ncol=2)

    n_reward = len(labels) - 1
    fig.suptitle(f'3 budovy: Fixed vs {n_reward} reward variantov Q-learningu', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_time_and_learning_comparison(fixed_run: PolicyRun, learned_run: PolicyRun,
                                      training_trace: TrainingTrace, output_path: Path,
                                      horizon: int) -> None:
    max_h = min(len(fixed_run.trajectory), len(learned_run.trajectory))
    horizon = max_h if horizon <= 0 else min(horizon, max_h)
    fs = fixed_run.trajectory.head(horizon)
    ls = learned_run.trajectory.head(horizon)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(fs['time_step'], fs['reward'], '--', lw=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[0, 0].plot(ls['time_step'], ls['reward'], lw=1.9, color='#1d3557', label='Q-learning')
    axes[0, 0].set_title('Step reward in time')
    axes[0, 0].set_xlabel('Time step'); axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(alpha=0.25); axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(fs['time_step'], fs['cumulative_reward'], '--', lw=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[0, 1].plot(ls['time_step'], ls['cumulative_reward'], lw=1.9, color='#1d3557', label='Q-learning')
    axes[0, 1].set_title('Cumulative reward in time')
    axes[0, 1].set_xlabel('Time step'); axes[0, 1].set_ylabel('Cumulative reward')
    axes[0, 1].grid(alpha=0.25); axes[0, 1].legend(fontsize=9)

    eps = np.arange(1, len(training_trace.episode_rewards) + 1)
    rolling = pd.Series(np.asarray(training_trace.episode_rewards, dtype=float)).rolling(
        min(10, len(training_trace.episode_rewards)), min_periods=1).mean()
    axes[1, 0].plot(eps, rolling, lw=2.0, color='#2a9d8f')
    axes[1, 0].set_title('Learning progress (rolling episode reward)')
    axes[1, 0].set_xlabel('Episode'); axes[1, 0].set_ylabel('Rolling episode reward')
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(fs['time_step'], fs['cumulative_grid_import_kwh'], '--', lw=1.6, color='#9aa0a6', label='Fixed strategy')
    axes[1, 1].plot(ls['time_step'], ls['cumulative_grid_import_kwh'], lw=1.9, color='#1d3557', label='Q-learning')
    axes[1, 1].set_title('Cumulative grid import in time')
    axes[1, 1].set_xlabel('Time step'); axes[1, 1].set_ylabel('kWh')
    axes[1, 1].grid(alpha=0.25); axes[1, 1].legend(fontsize=9)

    fig.suptitle('3 akcie: DHW zásobník + batéria + chladenie', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENTÁLNA SLUČKA
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(schema_path: Path, building_names: list[str], episodes: int,
                   baseline_cooling: float, random_seeds: list[int], output_dir: Path,
                   comparison_horizon: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    # ── Fixná referencia ──
    print('Evaluating fixed strategy (DHW=0, battery=0, cooling=fixed)...', flush=True)
    fixed_env = make_env(schema_path, building_names, random_seeds[0])
    fixed_policy = FixedPolicy(dhw_action=0.0, electrical_action=0.0, cooling_action=baseline_cooling)
    fixed_run = run_policy(fixed_policy, fixed_env, deterministic=True)
    fixed_run.result.policy = f'Fixed(cool={baseline_cooling:.2f})'
    fixed_run.result.seed = None
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    # ── Q-learning pre každý reward variant a seed ──
    learned_runs: list[tuple[str, str, PolicyRun, TrainingTrace, OwnAdaptiveTabularQLearning]] = []

    for seed in random_seeds:
        for key, display_name, reward_cls in REWARD_CONFIGS:
            print(f'\nTraining Q-learning with {display_name} ({reward_cls.__name__}), seed={seed}...', flush=True)
            env = make_env(schema_path, building_names, seed, reward_function=reward_cls)
            agent = OwnAdaptiveTabularQLearning(
                env, observation_bin_sizes=OBSERVATION_BIN_SIZES, action_bin_counts=ACTION_BIN_COUNTS,
                epsilon=1.0, minimum_epsilon=0.05, epsilon_decay=0.03,
                learning_rate=0.15, discount_factor=0.95,
                adaptive_patience=8, adaptive_epsilon_boost=0.08, adaptive_min_improvement=0.01,
                random_seed=seed,
            )
            trace = train_q_learning(agent, env, episodes=episodes, progress_every=max(1, episodes // 20))
            run = run_policy(agent, env, deterministic=True)
            run.result.policy = f'Q-learning ({display_name}, seed={seed})'
            run.result.seed = seed
            run.result.training_seconds = trace.training_seconds
            run.result.stability_episode = trace.stability_episode
            run.result.last_10_episode_reward_mean = (
                float(np.mean(trace.episode_rewards[-10:])) if trace.episode_rewards else None
            )
            if fixed_run.result.total_grid_import_kwh > 0.0:
                run.result.savings_vs_fixed_pct = 100.0 * (
                    fixed_run.result.total_grid_import_kwh - run.result.total_grid_import_kwh
                ) / fixed_run.result.total_grid_import_kwh
            results.append(run.result)
            learned_runs.append((f'{key}_seed_{seed}', f'{display_name} (seed={seed})', run, trace, agent))

    # ── Ukladanie výsledkov ──
    results_frame = build_results_frame(results).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)

    # Porovnávací graf všetkých politík
    all_policy_runs = [fixed_run] + [lr[2] for lr in learned_runs]
    save_policy_comparison_figure(results_frame, all_policy_runs, output_dir / 'policy_comparison.png')

    # Per-reward grafy + ukladanie dát
    for key, display_name, run, trace, agent in learned_runs:
        save_time_and_learning_comparison(
            fixed_run, run, trace,
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Q-learning: povodna verzia s reward konfiguraciami, aktualne spustena len pre Weather reward s jednym seedom na citylearn_challenge_2023_phase_3_3.'
    )
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument('--buildings', nargs='+', default=DEFAULT_BUILDINGS)
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--baseline-cooling', type=float, default=0.5)
    parser.add_argument('--random-seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--comparison-horizon', type=int, default=2208)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    reward_names = [name for _, name, _ in REWARD_CONFIGS]
    print(f'Schema: {args.schema}', flush=True)
    print(f'Buildings: {", ".join(args.buildings)}', flush=True)
    print(f'Engineered features: {", ".join(ENGINEERED_FEATURES)}', flush=True)
    print(f'Feature bins: {OBSERVATION_BIN_SIZES}  ->  {int(np.prod(list(OBSERVATION_BIN_SIZES.values())))} states', flush=True)
    print(f'Actions: {", ".join(ACTIVE_ACTIONS)}', flush=True)
    print(f'Action bins: {ACTION_BIN_COUNTS}  →  {int(np.prod(ACTION_BIN_COUNTS))} joint akcií', flush=True)
    print(f'Reward functions ({len(reward_names)}): {", ".join(reward_names)}', flush=True)
    print(f'Seeds: {args.random_seeds}', flush=True)
    print(f'Episodes: {args.episodes}', flush=True)
    print(f'Baseline cooling: {args.baseline_cooling}', flush=True)
    print(f'Output dir: {args.output_dir}', flush=True)

    results = run_experiment(
        schema_path=args.schema, building_names=args.buildings, episodes=args.episodes,
        baseline_cooling=args.baseline_cooling, random_seeds=args.random_seeds,
        output_dir=args.output_dir, comparison_horizon=args.comparison_horizon,
    )

    print('\nVýsledky:', flush=True)
    print(results.to_string(index=False))
    print(f'\nVšetky súbory uložené v: {args.output_dir}', flush=True)


if __name__ == '__main__':
    main()
