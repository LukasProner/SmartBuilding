#!/usr/bin/env python
"""
Q_learning1304_01.py
===================

Tabular shared-Q learning for CityLearn 2022 Phase 1.

This script is built around the actual assignment goals:
1. learn a battery policy from weather forecasts and occupancy proxy,
2. compare against a fixed baseline,
3. compare multiple reward functions under the same training/evaluation setup.

Important dataset note
----------------------
The 2022 Phase 1 dataset does not expose room occupancy directly.
For the assignment requirement, `non_shiftable_load` is used as a practical
proxy for occupancy/activity in the building.

Main design fixes compared to previous attempts
-----------------------------------------------
1. Use the real CityLearn action bounds from `env.action_space`.
2. Use one shared Q-table across all buildings to improve sample efficiency.
3. Keep the state compact but meaningful:
   (hour, price_class, future_peak, solar_forecast_bin, occupancy_bin, soc_bin)
4. Train several reward variants and keep the best validation checkpoint.
5. Evaluate every policy on the same deterministic 52-week schedule.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import citylearn
from citylearn.citylearn import CityLearnEnv


# ============================================================================
# CONFIG
# ============================================================================

DATASET_NAME = "citylearn_challenge_2022_phase_1"
OUTPUT_PREFIX = "q_learning1304_01"
RANDOM_SEED = 42

FAST_MODE = True

if FAST_MODE:
    EPISODE_TIME_STEPS = 24 * 7
    TRAIN_EPISODES = 385
    VALIDATION_EPISODES = 12
    EVAL_EPISODES = 52
    EVAL_EVERY = 20
else:
    EPISODE_TIME_STEPS = 24 * 7
    TRAIN_EPISODES = 900
    VALIDATION_EPISODES = 16
    EVAL_EPISODES = 52
    EVAL_EVERY = 25

ALPHA_START = 0.22
ALPHA_END = 0.05
GAMMA = 0.985
EPSILON_START = 1.0
EPSILON_END = 0.03

CHEAP_PRICE_MAX = 0.22
PEAK_PRICE_MIN = 0.50

SOC_EMPTY_THRESH = 0.05
SOC_FULL_THRESH = 0.95
SOC_BINS = np.array([0.15, 0.40, 0.70, 0.90], dtype=np.float32)

SOLAR_FALLBACK = np.array([150.0, 450.0, 750.0], dtype=np.float32)
LOAD_FALLBACK = np.array([0.9, 1.8], dtype=np.float32)

REWARD_MODES = ["energy_pv", "cost", "balanced"]


# ============================================================================
# HELPERS
# ============================================================================

StateT = Tuple[int, int, int, int, int, int, int]


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def resolve_schema(dataset_name: str) -> str:
    localapp = os.environ.get("LOCALAPPDATA", "")
    cached = (
        Path(localapp)
        / "intelligent-environments-lab"
        / "citylearn"
        / "Cache"
        / f"v{citylearn.__version__}"
        / "datasets"
        / dataset_name
        / "schema.json"
    )
    return str(cached) if cached.exists() else dataset_name


def make_env(random_seed: int, random_episode_split: bool) -> CityLearnEnv:
    return CityLearnEnv(
        schema=resolve_schema(DATASET_NAME),
        central_agent=False,
        episode_time_steps=EPISODE_TIME_STEPS,
        random_episode_split=random_episode_split,
        random_seed=random_seed,
    )


def get_obs_index(env: CityLearnEnv) -> Dict[str, int]:
    return {name: i for i, name in enumerate(env.observation_names[0])}


def first_key(idx: Dict[str, int], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in idx:
            return key
    return None


def quantile_bins(values: List[float], n: int, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size < max(20, n * 3):
        return fallback
    qs = np.linspace(0.0, 1.0, n + 1)[1:-1]
    bins = np.unique(np.quantile(arr, qs))
    return bins.astype(np.float32) if bins.size > 0 else fallback


def dig(value: float, bins: np.ndarray) -> int:
    safe_value = value if np.isfinite(value) else 0.0
    return int(np.digitize(safe_value, bins, right=False))


def moving_average(values: List[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if window <= 1 or arr.size == 0:
        return arr
    return np.convolve(arr, np.ones(window, dtype=np.float32) / window, mode="same")


def safe_pct(base: float, value: float, cap: float = 500.0) -> Optional[float]:
    if base <= 1e-6:
        return None
    return float(np.clip(100.0 * (base - value) / base, -cap, cap))


def price_class(price: float) -> int:
    if price <= CHEAP_PRICE_MAX:
        return 0
    if price < PEAK_PRICE_MIN:
        return 1
    return 2


def net_demand_class(net_value: float) -> int:
    if net_value < -0.10:
        return 0
    if net_value <= 1.00:
        return 1
    return 2


def district_grid_import(env: CityLearnEnv) -> float:
    if not env.net_electricity_consumption:
        return 0.0
    return max(0.0, float(env.net_electricity_consumption[-1]))


def district_cost(env: CityLearnEnv) -> float:
    if hasattr(env, "net_electricity_consumption_cost") and env.net_electricity_consumption_cost:
        return float(env.net_electricity_consumption_cost[-1])
    return 0.0


def format_bins(bins: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:.3g}" for x in bins) + "]"


def summarize_values(values: List[float]) -> str:
    if not values:
        return "-"
    arr = np.asarray(values, dtype=np.float32)
    return f"min={arr.min():.3g}  mean={arr.mean():.3g}  max={arr.max():.3g}  (n={arr.size})"


# ============================================================================
# STATE BINS
# ============================================================================

class BinSet:
    def __init__(self, env: CityLearnEnv) -> None:
        idx = get_obs_index(env)

        self.k_hour = first_key(idx, ["hour"])
        self.k_price = first_key(idx, ["electricity_pricing"])
        self.k_price_6h = first_key(idx, ["electricity_pricing_predicted_6h", "electricity_pricing_predicted_1"])
        self.k_price_12h = first_key(idx, ["electricity_pricing_predicted_12h", "electricity_pricing_predicted_2"])
        self.k_price_24h = first_key(idx, ["electricity_pricing_predicted_24h", "electricity_pricing_predicted_3"])
        self.k_load = first_key(idx, ["non_shiftable_load"])
        self.k_soc = first_key(idx, ["electrical_storage_soc"])
        self.k_net = first_key(idx, ["net_electricity_consumption"])
        self.k_solar_gen = first_key(idx, ["solar_generation"])
        self.k_diff_6h = first_key(idx, ["diffuse_solar_irradiance_predicted_6h", "diffuse_solar_irradiance_predicted_1"])
        self.k_dir_6h = first_key(idx, ["direct_solar_irradiance_predicted_6h", "direct_solar_irradiance_predicted_1"])
        self.k_diff_12h = first_key(idx, ["diffuse_solar_irradiance_predicted_12h", "diffuse_solar_irradiance_predicted_2"])
        self.k_dir_12h = first_key(idx, ["direct_solar_irradiance_predicted_12h", "direct_solar_irradiance_predicted_2"])

        self.i_hour = idx[self.k_hour] if self.k_hour else None
        self.i_price = idx[self.k_price] if self.k_price else None
        self.i_price_6h = idx[self.k_price_6h] if self.k_price_6h else None
        self.i_price_12h = idx[self.k_price_12h] if self.k_price_12h else None
        self.i_price_24h = idx[self.k_price_24h] if self.k_price_24h else None
        self.i_load = idx[self.k_load] if self.k_load else None
        self.i_soc = idx[self.k_soc] if self.k_soc else None
        self.i_net = idx[self.k_net] if self.k_net else None
        self.i_solar_gen = idx[self.k_solar_gen] if self.k_solar_gen else None
        self.i_diff_6h = idx[self.k_diff_6h] if self.k_diff_6h else None
        self.i_dir_6h = idx[self.k_dir_6h] if self.k_dir_6h else None
        self.i_diff_12h = idx[self.k_diff_12h] if self.k_diff_12h else None
        self.i_dir_12h = idx[self.k_dir_12h] if self.k_dir_12h else None

        self.load_bins: np.ndarray = LOAD_FALLBACK
        self.solar_bins: np.ndarray = SOLAR_FALLBACK
        self.soc_bins: np.ndarray = SOC_BINS

        self._load_samples: List[float] = []
        self._solar_samples: List[float] = []

    def solar_forecast(self, obs: List[float]) -> float:
        candidates: List[float] = []

        def add_pair(i_diff: Optional[int], i_dir: Optional[int]) -> None:
            total = 0.0
            used = False
            if i_diff is not None:
                total += float(obs[i_diff])
                used = True
            if i_dir is not None:
                total += float(obs[i_dir])
                used = True
            if used:
                candidates.append(total)

        add_pair(self.i_diff_6h, self.i_dir_6h)
        add_pair(self.i_diff_12h, self.i_dir_12h)

        if not candidates and self.i_solar_gen is not None:
            candidates.append(max(0.0, float(obs[self.i_solar_gen])) * 250.0)

        return max(candidates) if candidates else 0.0

    def future_peak_flag(self, obs: List[float]) -> int:
        values = []
        for i_price in [self.i_price_6h, self.i_price_12h, self.i_price_24h]:
            if i_price is not None:
                values.append(float(obs[i_price]))
        if self.i_price is not None:
            values.append(float(obs[self.i_price]))
        return int(max(values) >= PEAK_PRICE_MIN) if values else 0

    def fit(self, env: CityLearnEnv, episodes: int = 6) -> None:
        load_values: List[float] = []
        solar_values: List[float] = []

        for _ in range(episodes):
            obs_list = env.reset()
            for _ in range(EPISODE_TIME_STEPS):
                for obs in obs_list:
                    if self.i_load is not None:
                        load_values.append(float(obs[self.i_load]))
                    solar_values.append(self.solar_forecast(obs))
                obs_list, _, done, _ = env.step([[0.0] for _ in env.buildings])
                if done:
                    break

        self._load_samples = load_values
        self._solar_samples = solar_values
        self.load_bins = quantile_bins(load_values, 3, LOAD_FALLBACK)
        self.solar_bins = quantile_bins(solar_values, 4, SOLAR_FALLBACK)

    def encode(self, obs: List[float], soc: float) -> StateT:
        hour = int(np.clip((int(round(float(obs[self.i_hour]))) - 1) if self.i_hour is not None else 0, 0, 23))
        price = float(obs[self.i_price]) if self.i_price is not None else CHEAP_PRICE_MAX
        load = float(obs[self.i_load]) if self.i_load is not None else 0.0
        solar_forecast = self.solar_forecast(obs)
        net_value = float(obs[self.i_net]) if self.i_net is not None else 0.0

        return (
            hour,
            price_class(price),
            self.future_peak_flag(obs),
            net_demand_class(net_value),
            dig(solar_forecast, self.solar_bins),
            dig(load, self.load_bins),
            dig(soc, self.soc_bins),
        )

    def print_summary(self) -> None:
        sep = "-" * 92
        print(f"\n    {sep}")
        print(f"    {'Feature':<18}  {'Source':<42}  Summary")
        print(f"    {sep}")
        print(f"    {'hour':<18}  {str(self.k_hour):<42}  24 exact hours")
        print(f"    {'price class':<18}  {str(self.k_price):<42}  0=cheap,1=mid,2=peak")
        print(f"    {'future peak':<18}  {str(self.k_price_6h)} / {str(self.k_price_12h):<20}  max forecast >= {PEAK_PRICE_MIN}")
        print(f"    {'net class':<18}  {str(self.k_net):<42}  0=export,1=small import,2=high import")
        print(f"    {'solar forecast':<18}  6h/12h irradiance forecast sums{'':<12}  {summarize_values(self._solar_samples)}")
        print(f"    {'':<18}  {'':<42}  bins: {format_bins(self.solar_bins)}")
        print(f"    {'occupancy proxy':<18}  {str(self.k_load):<42}  {summarize_values(self._load_samples)}")
        print(f"    {'':<18}  {'':<42}  bins: {format_bins(self.load_bins)}")
        print(f"    {'battery soc':<18}  {str(self.k_soc):<42}  bins: {format_bins(self.soc_bins)}")
        print(f"    {sep}")
        state_count = 24 * 3 * 2 * 3 * (len(self.solar_bins) + 1) * (len(self.load_bins) + 1) * (len(self.soc_bins) + 1)
        print(f"    State space: 24 x 3 x 2 x 3 x {len(self.solar_bins) + 1} x {len(self.load_bins) + 1} x {len(self.soc_bins) + 1} = {state_count:,}")
        print(f"    Note: occupancy requirement is approximated with non_shiftable_load.\n")


# ============================================================================
# Q AGENT
# ============================================================================

class SharedTabularQAgent:
    def __init__(self, action_levels: np.ndarray) -> None:
        self.action_levels = action_levels.astype(np.float32)
        self.q_table: Dict[StateT, np.ndarray] = {}
        self.epsilon = EPSILON_START
        self.alpha = ALPHA_START

    def _q(self, state: StateT) -> np.ndarray:
        q = self.q_table.get(state)
        if q is None:
            q = np.zeros((len(self.action_levels),), dtype=np.float32)
            self.q_table[state] = q
        return q

    def set_schedules(self, episode: int, total_episodes: int) -> None:
        progress = float(episode) / max(1, total_episodes - 1)
        self.alpha = float(ALPHA_END + (ALPHA_START - ALPHA_END) * (1.0 - progress))
        self.epsilon = float(EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-5.0 * progress))

    def valid_action_ids(self, soc: float) -> List[int]:
        valid: List[int] = []
        for i, value in enumerate(self.action_levels):
            if soc <= SOC_EMPTY_THRESH and value < 0.0:
                continue
            if soc >= SOC_FULL_THRESH and value > 0.0:
                continue
            valid.append(i)
        return valid or [int(np.argmin(np.abs(self.action_levels)))]

    def act(self, state: StateT, soc: float, training: bool) -> int:
        valid = self.valid_action_ids(soc)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid))
        q = self._q(state)
        return int(max(valid, key=lambda action_id: q[action_id]))

    def update(self, state: StateT, action_id: int, reward: float, next_state: StateT, next_soc: float, done: bool) -> None:
        q = self._q(state)
        next_q = self._q(next_state)
        next_valid = self.valid_action_ids(next_soc)
        best_next = float(max(next_q[i] for i in next_valid))
        target = float(reward) + (0.0 if done else GAMMA * best_next)
        q[action_id] = (1.0 - self.alpha) * q[action_id] + self.alpha * target

    @property
    def total_states(self) -> int:
        return len(self.q_table)


# ============================================================================
# METRICS AND POLICY EVAL
# ============================================================================

@dataclass
class EvalMetrics:
    avg_energy_import: float
    avg_cost: float
    episode_energies: List[float]
    episode_costs: List[float]


@dataclass
class PolicyResult:
    reward_mode: str
    validation_score: float
    eval_metrics: EvalMetrics
    training_rewards: List[float]
    training_energy: List[float]
    training_cost: List[float]
    validation_history: List[Tuple[int, float]]
    visited_states: int


def build_actions_from_values(n_buildings: int, values: List[float]) -> List[List[float]]:
    return [[float(values[i])] for i in range(n_buildings)]


def heuristic_action_value(obs: List[float], bins: BinSet, low: float, high: float) -> float:
    price = float(obs[bins.i_price]) if bins.i_price is not None else CHEAP_PRICE_MAX
    load = float(obs[bins.i_load]) if bins.i_load is not None else 0.0
    soc = float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0
    solar = max(0.0, float(obs[bins.i_solar_gen])) if bins.i_solar_gen is not None else 0.0
    net = float(obs[bins.i_net]) if bins.i_net is not None else max(0.0, load - solar)

    if net < -0.10 and soc < 0.95:
        return high
    if price >= PEAK_PRICE_MIN and soc > 0.08:
        return low
    if solar > load + 0.20 and soc < 0.92:
        return 0.5 * high
    if net > 1.50 and soc > 0.12:
        return 0.5 * low
    return 0.0


def score_against_fixed(fixed: EvalMetrics, other: EvalMetrics, reward_mode: str) -> float:
    energy_pct = safe_pct(fixed.avg_energy_import, other.avg_energy_import) or -500.0
    cost_pct = safe_pct(fixed.avg_cost, other.avg_cost) or -500.0
    if reward_mode == "energy_pv":
        return 0.80 * energy_pct + 0.20 * cost_pct
    if reward_mode == "cost":
        return 0.20 * energy_pct + 0.80 * cost_pct
    return 0.50 * energy_pct + 0.50 * cost_pct


def compute_local_reward(
    reward_mode: str,
    prev_obs: List[float],
    building,
    bins: BinSet,
    action_value: float,
    action_max_abs: float,
) -> float:
    net = float(building.net_electricity_consumption[-1])
    net_import = max(0.0, net)
    net_wo_storage = float(building.net_electricity_consumption_without_storage[-1])
    import_wo_storage = max(0.0, net_wo_storage)
    export_wo_storage = max(0.0, -net_wo_storage)
    storage_electricity = float(building.electrical_storage_electricity_consumption[-1])
    storage_charge = max(0.0, storage_electricity)
    storage_discharge = max(0.0, -storage_electricity)
    price = float(prev_obs[bins.i_price]) if bins.i_price is not None else CHEAP_PRICE_MAX
    load = float(prev_obs[bins.i_load]) if bins.i_load is not None else 0.0
    solar = max(0.0, float(prev_obs[bins.i_solar_gen])) if bins.i_solar_gen is not None else 0.0
    soc = float(prev_obs[bins.i_soc]) if bins.i_soc is not None else 0.0
    future_peak = bins.future_peak_flag(prev_obs)

    action_scale = abs(action_value) / max(1e-6, action_max_abs)
    charging = action_value > 0.0
    discharging = action_value < 0.0
    pv_capture = min(storage_charge, export_wo_storage)
    harmful_grid_charge = max(0.0, storage_charge - export_wo_storage)
    import_reduction = max(0.0, import_wo_storage - net_import)

    if reward_mode == "energy_pv":
        reward = -(1.20 * net_import + 0.08 * action_scale + 0.10 * harmful_grid_charge)
        reward += 0.55 * pv_capture
        reward += 0.20 * import_reduction
        if charging and price >= PEAK_PRICE_MIN:
            reward -= 0.25
        if charging and harmful_grid_charge > 0.05:
            reward -= 0.35 * harmful_grid_charge
    elif reward_mode == "cost":
        reward = -(net_import * price) - 0.01 * action_scale
        reward += 0.10 * pv_capture
        if charging and price >= PEAK_PRICE_MIN:
            reward -= 0.12
    elif reward_mode == "balanced":
        reward = -(1.10 * net_import * price + 0.55 * net_import + 0.04 * action_scale + 0.06 * harmful_grid_charge)
        reward += 0.30 * pv_capture
        reward += 0.12 * import_reduction
        if charging and soc < 0.92 and solar > load + 0.1:
            reward += 0.12
        if discharging and soc > 0.10 and price >= PEAK_PRICE_MIN:
            reward += 0.18
        if charging and price >= PEAK_PRICE_MIN:
            reward -= 0.22
        if charging and harmful_grid_charge > 0.05 and future_peak == 0:
            reward -= 0.08
        if discharging and price <= CHEAP_PRICE_MAX and future_peak:
            reward -= 0.08
    else:
        raise ValueError(f"Unsupported reward mode: {reward_mode}")

    return float(reward)


def evaluate_fixed_policy(value: float, episodes: int, bins: BinSet) -> EvalMetrics:
    env = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    ep_energies: List[float] = []
    ep_costs: List[float] = []

    for _ in range(episodes):
        env.reset()
        ep_energy = 0.0
        ep_cost = 0.0
        for _ in range(EPISODE_TIME_STEPS):
            _, _, done, _ = env.step([[value] for _ in env.buildings])
            ep_energy += district_grid_import(env)
            ep_cost += district_cost(env)
            if done:
                break
        ep_energies.append(ep_energy)
        ep_costs.append(ep_cost)

    return EvalMetrics(
        avg_energy_import=float(np.mean(ep_energies)),
        avg_cost=float(np.mean(ep_costs)),
        episode_energies=ep_energies,
        episode_costs=ep_costs,
    )


def evaluate_heuristic_policy(episodes: int, bins: BinSet, action_low: float, action_high: float) -> EvalMetrics:
    env = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    ep_energies: List[float] = []
    ep_costs: List[float] = []

    for _ in range(episodes):
        obs_list = env.reset()
        ep_energy = 0.0
        ep_cost = 0.0

        for _ in range(EPISODE_TIME_STEPS):
            actions = [[heuristic_action_value(obs, bins, action_low, action_high)] for obs in obs_list]
            obs_list, _, done, _ = env.step(actions)
            ep_energy += district_grid_import(env)
            ep_cost += district_cost(env)
            if done:
                break

        ep_energies.append(ep_energy)
        ep_costs.append(ep_cost)

    return EvalMetrics(
        avg_energy_import=float(np.mean(ep_energies)),
        avg_cost=float(np.mean(ep_costs)),
        episode_energies=ep_energies,
        episode_costs=ep_costs,
    )


def evaluate_agent(agent: SharedTabularQAgent, bins: BinSet, episodes: int) -> EvalMetrics:
    env = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    ep_energies: List[float] = []
    ep_costs: List[float] = []

    for _ in range(episodes):
        obs_list = env.reset()
        socs = [float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0 for obs in obs_list]
        states = [bins.encode(obs, soc) for obs, soc in zip(obs_list, socs)]
        ep_energy = 0.0
        ep_cost = 0.0

        for _ in range(EPISODE_TIME_STEPS):
            action_ids = [agent.act(state, soc, training=False) for state, soc in zip(states, socs)]
            action_values = [float(agent.action_levels[i]) for i in action_ids]
            obs_list, _, done, _ = env.step(build_actions_from_values(len(env.buildings), action_values))
            ep_energy += district_grid_import(env)
            ep_cost += district_cost(env)

            socs = [float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0 for obs in obs_list]
            states = [bins.encode(obs, soc) for obs, soc in zip(obs_list, socs)]
            if done:
                break

        ep_energies.append(ep_energy)
        ep_costs.append(ep_cost)

    return EvalMetrics(
        avg_energy_import=float(np.mean(ep_energies)),
        avg_cost=float(np.mean(ep_costs)),
        episode_energies=ep_energies,
        episode_costs=ep_costs,
    )


# ============================================================================
# TRAINING
# ============================================================================

def train_reward_mode(
    reward_mode: str,
    bins: BinSet,
    action_levels: np.ndarray,
    validation_baseline: EvalMetrics,
) -> PolicyResult:
    train_env = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    agent = SharedTabularQAgent(action_levels=action_levels)
    action_max_abs = float(np.max(np.abs(action_levels)))

    best_score = -float("inf")
    best_q_table: Dict[StateT, np.ndarray] = {}
    training_rewards: List[float] = []
    training_energy: List[float] = []
    training_cost: List[float] = []
    validation_history: List[Tuple[int, float]] = []

    for episode in tqdm(range(TRAIN_EPISODES), desc=f"Train {reward_mode}"):
        agent.set_schedules(episode, TRAIN_EPISODES)
        obs_list = train_env.reset()
        socs = [float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0 for obs in obs_list]
        states = [bins.encode(obs, soc) for obs, soc in zip(obs_list, socs)]

        episode_reward = 0.0
        episode_energy = 0.0
        episode_cost = 0.0

        for _ in range(EPISODE_TIME_STEPS):
            action_ids = [agent.act(state, soc, training=True) for state, soc in zip(states, socs)]
            action_values = [float(action_levels[i]) for i in action_ids]
            next_obs_list, _, done, _ = train_env.step(build_actions_from_values(len(train_env.buildings), action_values))
            next_socs = [float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0 for obs in next_obs_list]
            next_states = [bins.encode(obs, soc) for obs, soc in zip(next_obs_list, next_socs)]

            local_rewards = [
                compute_local_reward(reward_mode, states_obs, train_env.buildings[i], bins, action_values[i], action_max_abs)
                for i, states_obs in enumerate(obs_list)
            ]

            for state, action_id, reward, next_state, next_soc in zip(states, action_ids, local_rewards, next_states, next_socs):
                agent.update(state, action_id, reward, next_state, next_soc, done)

            episode_reward += float(np.sum(local_rewards))
            episode_energy += district_grid_import(train_env)
            episode_cost += district_cost(train_env)

            states = next_states
            socs = next_socs
            if done:
                break

        training_rewards.append(episode_reward)
        training_energy.append(episode_energy)
        training_cost.append(episode_cost)

        if (episode + 1) % EVAL_EVERY == 0 or episode == TRAIN_EPISODES - 1:
            validation_metrics = evaluate_agent(agent, bins, VALIDATION_EPISODES)
            validation_score = score_against_fixed(validation_baseline, validation_metrics, reward_mode)
            validation_history.append((episode + 1, validation_score))
            if validation_score > best_score:
                best_score = validation_score
                best_q_table = copy.deepcopy(agent.q_table)

    if best_q_table:
        agent.q_table = best_q_table

    final_metrics = evaluate_agent(agent, bins, EVAL_EPISODES)

    return PolicyResult(
        reward_mode=reward_mode,
        validation_score=best_score,
        eval_metrics=final_metrics,
        training_rewards=training_rewards,
        training_energy=training_energy,
        training_cost=training_cost,
        validation_history=validation_history,
        visited_states=agent.total_states,
    )


# ============================================================================
# PLOTS AND OUTPUTS
# ============================================================================

def plot_training_histories(results: List[PolicyResult]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Q_learning1304_01 - Reward comparison during training\n"
        "Shared Q-table | state=(hour, price, future peak, solar forecast, occupancy proxy, SOC)",
        fontsize=11,
        fontweight="bold",
    )

    for result in results:
        axes[0, 0].plot(result.training_rewards, alpha=0.28, linewidth=0.9)
        axes[0, 0].plot(moving_average(result.training_rewards, 20), linewidth=2, label=result.reward_mode)
        axes[0, 1].plot(moving_average(result.training_energy, 20), linewidth=2, label=result.reward_mode)
        axes[1, 0].plot(moving_average(result.training_cost, 20), linewidth=2, label=result.reward_mode)
        if result.validation_history:
            xs = [x for x, _ in result.validation_history]
            ys = [y for _, y in result.validation_history]
            axes[1, 1].plot(xs, ys, marker="o", linewidth=1.8, label=result.reward_mode)

    axes[0, 0].set_title("Training reward")
    axes[0, 1].set_title("Training district grid import")
    axes[1, 0].set_title("Training district cost")
    axes[1, 1].set_title("Validation score vs fixed baseline")

    axes[0, 0].set_ylabel("Reward")
    axes[0, 1].set_ylabel("kWh")
    axes[1, 0].set_ylabel("$")
    axes[1, 1].set_ylabel("score")

    for ax in axes.ravel():
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_training.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_eval_summary(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df[summary_df["policy"] != "fixed_baseline"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q_learning1304_01 - Evaluation vs fixed baseline", fontsize=11, fontweight="bold")

    colors = ["steelblue" if p.startswith("q_learning") else "darkorange" for p in plot_df["policy"]]

    axes[0].bar(plot_df["policy"], plot_df["energy_sav_%"], color=colors, alpha=0.88)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Energy savings vs fixed baseline")
    axes[0].set_ylabel("%")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, alpha=0.25, axis="y")

    axes[1].bar(plot_df["policy"], plot_df["cost_sav_%"], color=colors, alpha=0.88)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Cost savings vs fixed baseline")
    axes[1].set_ylabel("%")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_eval_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def build_summary_rows(fixed: EvalMetrics, heuristic: EvalMetrics, results: List[PolicyResult]) -> List[dict]:
    rows: List[dict] = [
        {
            "policy": "fixed_baseline",
            "reward_mode": "-",
            "avg_energy_import": fixed.avg_energy_import,
            "avg_cost": fixed.avg_cost,
            "energy_sav": 0.0,
            "energy_sav_%": 0.0,
            "cost_sav": 0.0,
            "cost_sav_%": 0.0,
            "validation_score": 0.0,
            "visited_states": 0,
        },
        {
            "policy": "heuristic_pv_rule",
            "reward_mode": "pv_handcrafted",
            "avg_energy_import": heuristic.avg_energy_import,
            "avg_cost": heuristic.avg_cost,
            "energy_sav": fixed.avg_energy_import - heuristic.avg_energy_import,
            "energy_sav_%": safe_pct(fixed.avg_energy_import, heuristic.avg_energy_import),
            "cost_sav": fixed.avg_cost - heuristic.avg_cost,
            "cost_sav_%": safe_pct(fixed.avg_cost, heuristic.avg_cost),
            "validation_score": score_against_fixed(fixed, heuristic, "balanced"),
            "visited_states": 0,
        },
    ]

    for result in results:
        metrics = result.eval_metrics
        rows.append(
            {
                "policy": f"q_learning_{result.reward_mode}",
                "reward_mode": result.reward_mode,
                "avg_energy_import": metrics.avg_energy_import,
                "avg_cost": metrics.avg_cost,
                "energy_sav": fixed.avg_energy_import - metrics.avg_energy_import,
                "energy_sav_%": safe_pct(fixed.avg_energy_import, metrics.avg_energy_import),
                "cost_sav": fixed.avg_cost - metrics.avg_cost,
                "cost_sav_%": safe_pct(fixed.avg_cost, metrics.avg_cost),
                "validation_score": result.validation_score,
                "visited_states": result.visited_states,
            }
        )

    return rows


def build_episode_comparison_df(fixed: EvalMetrics, heuristic: EvalMetrics, results: List[PolicyResult]) -> pd.DataFrame:
    n = EVAL_EPISODES
    rows: List[dict] = []

    for episode in range(n):
        row = {
            "episode": episode + 1,
            "fixed_energy": round(fixed.episode_energies[episode], 4),
            "fixed_cost": round(fixed.episode_costs[episode], 4),
            "heuristic_energy": round(heuristic.episode_energies[episode], 4),
            "heuristic_cost": round(heuristic.episode_costs[episode], 4),
        }

        for result in results:
            key = result.reward_mode
            energy = result.eval_metrics.episode_energies[episode]
            cost = result.eval_metrics.episode_costs[episode]
            row[f"{key}_energy"] = round(energy, 4)
            row[f"{key}_cost"] = round(cost, 4)
            row[f"{key}_energy_sav_%"] = round(safe_pct(fixed.episode_energies[episode], energy) or float("nan"), 4)
            row[f"{key}_cost_sav_%"] = round(safe_pct(fixed.episode_costs[episode], cost) or float("nan"), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def build_training_log_df(results: List[PolicyResult]) -> pd.DataFrame:
    rows: List[dict] = []
    for result in results:
        for episode, (reward, energy, cost) in enumerate(zip(result.training_rewards, result.training_energy, result.training_cost), start=1):
            rows.append(
                {
                    "reward_mode": result.reward_mode,
                    "episode": episode,
                    "training_reward": reward,
                    "training_energy": energy,
                    "training_cost": cost,
                }
            )
    return pd.DataFrame(rows)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    set_seed(RANDOM_SEED)
    sep = "=" * 84

    print(sep)
    print("  Q_learning1304_01 - Shared tabular Q-learning for SmartBuilding assignment")
    print("  Compare reward functions against fixed baseline on the same deterministic year")
    print(sep)
    print(f"  citylearn version    : {citylearn.__version__}")
    print(f"  dataset              : {DATASET_NAME}")
    print(f"  episode_time_steps   : {EPISODE_TIME_STEPS} ({EPISODE_TIME_STEPS // 24} days)")
    print(f"  train_episodes       : {TRAIN_EPISODES}")
    print(f"  validation_episodes  : {VALIDATION_EPISODES}")
    print(f"  eval_episodes        : {EVAL_EPISODES}")
    print(f"  reward variants      : {REWARD_MODES}")
    print("  occupancy signal     : non_shiftable_load (proxy, direct room occupancy not in 2022 dataset)")

    print("\n[1/6] Inspect environment and state bins...")
    probe_env = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    bins = BinSet(probe_env)
    bins.fit(probe_env)
    bins.print_summary()

    action_low = float(probe_env.action_space[0].low[0])
    action_high = float(probe_env.action_space[0].high[0])
    action_max_abs = min(abs(action_low), abs(action_high))
    action_levels = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32) * action_max_abs
    print(f"  action bounds        : [{action_low:.6f}, {action_high:.6f}]")
    print(f"  discrete actions     : {action_levels.tolist()}")

    print("\n[2/6] Fixed and heuristic baselines...")
    fixed_baseline = evaluate_fixed_policy(value=0.0, episodes=EVAL_EPISODES, bins=bins)
    validation_baseline = evaluate_fixed_policy(value=0.0, episodes=VALIDATION_EPISODES, bins=bins)
    heuristic_baseline = evaluate_heuristic_policy(EVAL_EPISODES, bins, action_low, action_high)
    print(f"  fixed baseline       : energy={fixed_baseline.avg_energy_import:.4f} kWh, cost={fixed_baseline.avg_cost:.4f} $")
    print(f"  heuristic rule       : energy={heuristic_baseline.avg_energy_import:.4f} kWh, cost={heuristic_baseline.avg_cost:.4f} $")

    print("\n[3/6] Train Q-learning variants...")
    results: List[PolicyResult] = []
    for reward_mode in REWARD_MODES:
        result = train_reward_mode(
            reward_mode=reward_mode,
            bins=bins,
            action_levels=action_levels,
            validation_baseline=validation_baseline,
        )
        results.append(result)
        print(
            f"  {reward_mode:<9} -> val_score={result.validation_score:+.3f}, "
            f"energy={result.eval_metrics.avg_energy_import:.4f} kWh, "
            f"cost={result.eval_metrics.avg_cost:.4f} $, states={result.visited_states}"
        )

    print("\n[4/6] Build comparison tables...")
    summary_df = pd.DataFrame(build_summary_rows(fixed_baseline, heuristic_baseline, results))
    summary_df.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)

    per_episode_df = build_episode_comparison_df(fixed_baseline, heuristic_baseline, results)
    per_episode_df.to_csv(f"{OUTPUT_PREFIX}_per_episode.csv", index=False)

    training_log_df = build_training_log_df(results)
    training_log_df.to_csv(f"{OUTPUT_PREFIX}_training_log.csv", index=False)

    print("\n[5/6] Create plots...")
    plot_training_histories(results)
    plot_eval_summary(summary_df)

    print("\n[6/6] Final summary...")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    best_row = summary_df[summary_df["policy"].str.startswith("q_learning_")].sort_values(
        by=["cost_sav_%", "energy_sav_%"], ascending=False
    ).iloc[0]
    print("\n  Best learned policy by evaluation ranking:")
    print(
        f"    {best_row['policy']} | energy_sav={best_row['energy_sav_%']:+.2f}% | "
        f"cost_sav={best_row['cost_sav_%']:+.2f}%"
    )

    print("\n  Saved files:")
    for file_name in [
        f"{OUTPUT_PREFIX}_results.csv",
        f"{OUTPUT_PREFIX}_per_episode.csv",
        f"{OUTPUT_PREFIX}_training_log.csv",
        f"{OUTPUT_PREFIX}_training.png",
        f"{OUTPUT_PREFIX}_eval_summary.png",
    ]:
        print(f"    {file_name}")

    print(sep)


if __name__ == "__main__":
    main()