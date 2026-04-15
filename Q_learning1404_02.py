#!/usr/bin/env python
"""
Q_learning1404_02.py
===================

Energy-first tabular Q-learning for CityLearn 2022 Phase 1.

Design choices in this version:
1. Compare only against the fixed baseline.
2. Focus on energy import reduction, not price arbitrage.
3. Use weather + occupancy proxy signals that matter for a battery-only task:
   (hour, solar_forecast_bin, pv_surplus_flag, occupancy_bin, soc_bin).
4. Drop temperature from the state because this dataset only exposes electrical
   battery control, so temperature mostly adds noise instead of useful control
   structure.
5. Run multiple training restarts and keep the best model on validation energy.
6. Use stricter penalties for grid charging and wasted discharge.
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


DATASET_NAME = "citylearn_challenge_2022_phase_1"
OUTPUT_PREFIX = "q_learning1404_02"
RANDOM_SEED = 42

FAST_MODE = True

if FAST_MODE:
    EPISODE_TIME_STEPS = 24 * 7
    TRAIN_EPISODES = 240
    VALIDATION_EPISODES = 12
    EVAL_EPISODES = 52
    EVAL_EVERY = 15
    TRAINING_SEEDS = [42, 52, 62]
else:
    EPISODE_TIME_STEPS = 24 * 7
    TRAIN_EPISODES = 480
    VALIDATION_EPISODES = 12
    EVAL_EPISODES = 52
    EVAL_EVERY = 20
    TRAINING_SEEDS = [42, 52, 62, 72]

ALPHA_START = 0.20
ALPHA_END = 0.04
GAMMA = 0.985
EPSILON_START = 1.0
EPSILON_END = 0.03

SOC_EMPTY_THRESH = 0.05
SOC_FULL_THRESH = 0.95

SOC_BINS = np.array([0.15, 0.40, 0.70, 0.90], dtype=np.float32)
SOLAR_FALLBACK = np.array([120.0, 500.0, 950.0], dtype=np.float32)
LOAD_FALLBACK = np.array([0.65, 1.60], dtype=np.float32)

StateT = Tuple[int, int, int, int, int]


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


class BinSet:
    def __init__(self, env: CityLearnEnv) -> None:
        idx = get_obs_index(env)

        self.k_hour = first_key(idx, ["hour"])
        self.k_month = first_key(idx, ["month"])
        self.k_load = first_key(idx, ["non_shiftable_load"])
        self.k_soc = first_key(idx, ["electrical_storage_soc"])
        self.k_net = first_key(idx, ["net_electricity_consumption"])
        self.k_solar_gen = first_key(idx, ["solar_generation"])
        self.k_diff_6h = first_key(idx, ["diffuse_solar_irradiance_predicted_6h", "diffuse_solar_irradiance_predicted_1"])
        self.k_dir_6h = first_key(idx, ["direct_solar_irradiance_predicted_6h", "direct_solar_irradiance_predicted_1"])
        self.k_diff_12h = first_key(idx, ["diffuse_solar_irradiance_predicted_12h", "diffuse_solar_irradiance_predicted_2"])
        self.k_dir_12h = first_key(idx, ["direct_solar_irradiance_predicted_12h", "direct_solar_irradiance_predicted_2"])

        self.i_hour = idx[self.k_hour] if self.k_hour else None
        self.i_month = idx[self.k_month] if self.k_month else None
        self.i_load = idx[self.k_load] if self.k_load else None
        self.i_soc = idx[self.k_soc] if self.k_soc else None
        self.i_net = idx[self.k_net] if self.k_net else None
        self.i_solar_gen = idx[self.k_solar_gen] if self.k_solar_gen else None
        self.i_diff_6h = idx[self.k_diff_6h] if self.k_diff_6h else None
        self.i_dir_6h = idx[self.k_dir_6h] if self.k_dir_6h else None
        self.i_diff_12h = idx[self.k_diff_12h] if self.k_diff_12h else None
        self.i_dir_12h = idx[self.k_dir_12h] if self.k_dir_12h else None

        self.solar_bins: np.ndarray = SOLAR_FALLBACK
        self.load_bins: np.ndarray = LOAD_FALLBACK
        self.soc_bins: np.ndarray = SOC_BINS

        self._solar_samples: List[float] = []
        self._load_samples: List[float] = []

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

    def pv_surplus_flag(self, obs: List[float]) -> int:
        load = float(obs[self.i_load]) if self.i_load is not None else 0.0
        solar = float(obs[self.i_solar_gen]) if self.i_solar_gen is not None else 0.0
        return int(solar > load + 0.1)

    def fit(self, env: CityLearnEnv, episodes: int = 6) -> None:
        solar_values: List[float] = []
        load_values: List[float] = []

        for _ in range(episodes):
            obs_list = env.reset()
            for _ in range(EPISODE_TIME_STEPS):
                for obs in obs_list:
                    solar_values.append(self.solar_forecast(obs))
                    if self.i_load is not None:
                        load_values.append(float(obs[self.i_load]))
                obs_list, _, done, _ = env.step([[0.0] for _ in env.buildings])
                if done:
                    break

        self._solar_samples = solar_values
        self._load_samples = load_values
        self.solar_bins = quantile_bins(solar_values, 4, SOLAR_FALLBACK)
        self.load_bins = quantile_bins(load_values, 3, LOAD_FALLBACK)

    def encode(self, obs: List[float], soc: float) -> StateT:
        hour = int(np.clip((int(round(float(obs[self.i_hour]))) - 1) if self.i_hour is not None else 0, 0, 23))
        return (
            hour,
            dig(self.solar_forecast(obs), self.solar_bins),
            self.pv_surplus_flag(obs),
            dig(float(obs[self.i_load]) if self.i_load is not None else 0.0, self.load_bins),
            dig(soc, self.soc_bins),
        )

    def print_summary(self) -> None:
        sep = "-" * 92
        print(f"\n    {sep}")
        print(f"    {'Feature':<18}  {'Source':<42}  Summary")
        print(f"    {sep}")
        print(f"    {'hour':<18}  {str(self.k_hour):<42}  24 exact hours")
        print(f"    {'solar forecast':<18}  6h/12h irradiance forecasts{'':<15}  {summarize_values(self._solar_samples)}")
        print(f"    {'':<18}  {'':<42}  bins: {format_bins(self.solar_bins)}")
        print(f"    {'occupancy proxy':<18}  {str(self.k_load):<42}  {summarize_values(self._load_samples)}")
        print(f"    {'':<18}  {'':<42}  bins: {format_bins(self.load_bins)}")
        print(f"    {'pv surplus flag':<18}  {str(self.k_solar_gen):<42}  0=no, 1=current solar > load")
        print(f"    {'battery soc':<18}  {str(self.k_soc):<42}  bins: {format_bins(self.soc_bins)}")
        print(f"    {sep}")
        state_count = 24 * (len(self.solar_bins) + 1) * 2 * (len(self.load_bins) + 1) * (len(self.soc_bins) + 1)
        print(f"    State space: 24 x {len(self.solar_bins)+1} x 2 x {len(self.load_bins)+1} x {len(self.soc_bins)+1} = {state_count:,}")
        print("    Direct occupancy is unavailable; non_shiftable_load is used as occupancy proxy.\n")


class SharedTabularQAgent:
    def __init__(self, action_levels: np.ndarray) -> None:
        self.action_levels = action_levels.astype(np.float32)
        self.q_table: Dict[StateT, np.ndarray] = {}
        self.alpha = ALPHA_START
        self.epsilon = EPSILON_START

    def _q(self, state: StateT) -> np.ndarray:
        q = self.q_table.get(state)
        if q is None:
            q = np.zeros((len(self.action_levels),), dtype=np.float32)
            self.q_table[state] = q
        return q

    def valid_action_ids(self, state: StateT, soc: float) -> List[int]:
        hour, solar_bin, pv_surplus_flag, load_bin, _soc_bin = state
        positive_context = pv_surplus_flag == 1 or (solar_bin >= 2 and 8 <= hour <= 13)
        evening_discharge_context = hour >= 17 or load_bin >= 2

        valid: List[int] = []
        for i, value in enumerate(self.action_levels):
            if soc <= SOC_EMPTY_THRESH and value < 0.0:
                continue
            if soc >= SOC_FULL_THRESH and value > 0.0:
                continue
            if value > 0.0 and not positive_context:
                continue
            if value > 0.0 and hour >= 15 and pv_surplus_flag == 0:
                continue
            if value < 0.0 and not evening_discharge_context and load_bin == 0 and hour < 12:
                continue
            valid.append(i)

        if not valid:
            zero_id = int(np.argmin(np.abs(self.action_levels)))
            valid = [zero_id]

        return valid

    def set_schedules(self, episode: int, total_episodes: int) -> None:
        progress = float(episode) / max(1, total_episodes - 1)
        self.alpha = float(ALPHA_END + (ALPHA_START - ALPHA_END) * (1.0 - progress))
        self.epsilon = float(EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-5.0 * progress))

    def act(self, state: StateT, soc: float, training: bool) -> int:
        valid = self.valid_action_ids(state, soc)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.choice(valid))
        q = self._q(state)
        return int(max(valid, key=lambda action_id: q[action_id]))

    def update(self, state: StateT, action_id: int, reward: float, next_state: StateT, next_soc: float, done: bool) -> None:
        q = self._q(state)
        next_q = self._q(next_state)
        next_valid = self.valid_action_ids(next_state, next_soc)
        best_next = float(max(next_q[i] for i in next_valid))
        target = float(reward) + (0.0 if done else GAMMA * best_next)
        q[action_id] = (1.0 - self.alpha) * q[action_id] + self.alpha * target

    @property
    def total_states(self) -> int:
        return len(self.q_table)


@dataclass
class EvalMetrics:
    avg_energy_import: float
    avg_cost: float
    episode_energies: List[float]
    episode_costs: List[float]


@dataclass
class EvalTraceResult:
    metrics: EvalMetrics
    trace_df: pd.DataFrame


@dataclass
class TrainingRunResult:
    seed: int
    validation_score: float
    validation_energy: float
    eval_metrics: EvalMetrics
    agent: SharedTabularQAgent
    train_rewards: List[float]
    train_energy: List[float]
    train_cost: List[float]
    validation_history: List[Tuple[int, float]]


def build_actions_from_values(values: List[float]) -> List[List[float]]:
    return [[float(value)] for value in values]


def compute_energy_reward(prev_obs: List[float], building, bins: BinSet, action_value: float, action_max_abs: float) -> float:
    net_import = max(0.0, float(building.net_electricity_consumption[-1]))
    net_wo_storage = float(building.net_electricity_consumption_without_storage[-1])
    import_wo_storage = max(0.0, net_wo_storage)
    export_wo_storage = max(0.0, -net_wo_storage)
    storage_electricity = float(building.electrical_storage_electricity_consumption[-1])
    storage_charge = max(0.0, storage_electricity)
    storage_discharge = max(0.0, -storage_electricity)
    load = float(prev_obs[bins.i_load]) if bins.i_load is not None else 0.0
    solar = max(0.0, float(prev_obs[bins.i_solar_gen])) if bins.i_solar_gen is not None else 0.0
    pv_surplus = solar > load + 0.1
    hour = int(round(float(prev_obs[bins.i_hour]))) if bins.i_hour is not None else 12

    action_scale = abs(action_value) / max(1e-6, action_max_abs)
    pv_capture = min(storage_charge, export_wo_storage)
    harmful_grid_charge = max(0.0, storage_charge - export_wo_storage)
    import_reduction = max(0.0, import_wo_storage - net_import)
    wasted_discharge = max(0.0, storage_discharge - import_wo_storage)

    reward = 1.40 * import_reduction + 1.00 * pv_capture
    reward -= 1.15 * net_import
    reward -= 1.40 * harmful_grid_charge
    reward -= 0.45 * wasted_discharge
    reward -= 0.05 * action_scale

    if pv_surplus and storage_charge > 0.0:
        reward += 0.10
    if not pv_surplus and storage_charge > 0.0:
        reward -= 0.20
    if hour >= 17 and storage_discharge > 0.0 and import_wo_storage > 0.5:
        reward += 0.12
    if hour <= 11 and storage_discharge > 0.0 and import_wo_storage < 0.2:
        reward -= 0.10

    return float(reward)


def energy_score_against_fixed(fixed: EvalMetrics, other: EvalMetrics) -> float:
    energy_pct = safe_pct(fixed.avg_energy_import, other.avg_energy_import) or -500.0
    cost_pct = safe_pct(fixed.avg_cost, other.avg_cost) or -500.0
    return 0.97 * energy_pct + 0.03 * cost_pct


def evaluate_fixed_policy(value: float, episodes: int) -> EvalMetrics:
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
            action_values = [float(agent.action_levels[action_id]) for action_id in action_ids]
            obs_list, _, done, _ = env.step(build_actions_from_values(action_values))
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


def evaluate_fixed_with_trace(value: float, bins: BinSet, episodes: int, label: str) -> EvalTraceResult:
    env = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    rows: List[dict] = []
    ep_energies: List[float] = []
    ep_costs: List[float] = []

    for episode in range(episodes):
        obs_list = env.reset()
        ep_energy = 0.0
        ep_cost = 0.0

        for step in range(EPISODE_TIME_STEPS):
            actions = [[value] for _ in env.buildings]
            next_obs_list, _, done, _ = env.step(actions)

            step_energy = district_grid_import(env)
            step_cost = district_cost(env)
            obs0 = next_obs_list[0]
            month = int(round(float(obs0[bins.i_month]))) if bins.i_month is not None else 1
            hour = int(round(float(obs0[bins.i_hour]))) if bins.i_hour is not None else step % 24
            load = float(obs0[bins.i_load]) if bins.i_load is not None else 0.0
            solar = float(obs0[bins.i_solar_gen]) if bins.i_solar_gen is not None else 0.0
            soc = float(obs0[bins.i_soc]) if bins.i_soc is not None else 0.0

            rows.append(
                {
                    "policy": label,
                    "episode": episode + 1,
                    "step": step + 1,
                    "month": month,
                    "hour": hour,
                    "energy_import": step_energy,
                    "cost": step_cost,
                    "load": load,
                    "solar_generation": solar,
                    "soc": soc,
                    "action_mean": float(value),
                }
            )

            ep_energy += step_energy
            ep_cost += step_cost
            obs_list = next_obs_list
            if done:
                break

        ep_energies.append(ep_energy)
        ep_costs.append(ep_cost)

    return EvalTraceResult(
        metrics=EvalMetrics(
            avg_energy_import=float(np.mean(ep_energies)),
            avg_cost=float(np.mean(ep_costs)),
            episode_energies=ep_energies,
            episode_costs=ep_costs,
        ),
        trace_df=pd.DataFrame(rows),
    )


def evaluate_agent_with_trace(agent: SharedTabularQAgent, bins: BinSet, episodes: int, label: str) -> EvalTraceResult:
    env = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    rows: List[dict] = []
    ep_energies: List[float] = []
    ep_costs: List[float] = []

    for episode in range(episodes):
        obs_list = env.reset()
        socs = [float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0 for obs in obs_list]
        states = [bins.encode(obs, soc) for obs, soc in zip(obs_list, socs)]
        ep_energy = 0.0
        ep_cost = 0.0

        for step in range(EPISODE_TIME_STEPS):
            action_ids = [agent.act(state, soc, training=False) for state, soc in zip(states, socs)]
            action_values = [float(agent.action_levels[action_id]) for action_id in action_ids]
            next_obs_list, _, done, _ = env.step(build_actions_from_values(action_values))

            step_energy = district_grid_import(env)
            step_cost = district_cost(env)
            obs0 = next_obs_list[0]
            month = int(round(float(obs0[bins.i_month]))) if bins.i_month is not None else 1
            hour = int(round(float(obs0[bins.i_hour]))) if bins.i_hour is not None else step % 24
            load = float(obs0[bins.i_load]) if bins.i_load is not None else 0.0
            solar = float(obs0[bins.i_solar_gen]) if bins.i_solar_gen is not None else 0.0
            soc = float(obs0[bins.i_soc]) if bins.i_soc is not None else 0.0

            rows.append(
                {
                    "policy": label,
                    "episode": episode + 1,
                    "step": step + 1,
                    "month": month,
                    "hour": hour,
                    "energy_import": step_energy,
                    "cost": step_cost,
                    "load": load,
                    "solar_generation": solar,
                    "soc": soc,
                    "action_mean": float(np.mean(action_values)),
                }
            )

            ep_energy += step_energy
            ep_cost += step_cost
            obs_list = next_obs_list
            socs = [float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0 for obs in obs_list]
            states = [bins.encode(obs, soc) for obs, soc in zip(obs_list, socs)]
            if done:
                break

        ep_energies.append(ep_energy)
        ep_costs.append(ep_cost)

    return EvalTraceResult(
        metrics=EvalMetrics(
            avg_energy_import=float(np.mean(ep_energies)),
            avg_cost=float(np.mean(ep_costs)),
            episode_energies=ep_energies,
            episode_costs=ep_costs,
        ),
        trace_df=pd.DataFrame(rows),
    )


def train_one_run(bins: BinSet, action_levels: np.ndarray, validation_baseline: EvalMetrics, seed: int) -> TrainingRunResult:
    train_env = make_env(random_seed=seed, random_episode_split=True)
    agent = SharedTabularQAgent(action_levels)
    action_max_abs = float(np.max(np.abs(action_levels)))

    best_score = -float("inf")
    best_q_table: Dict[StateT, np.ndarray] = {}
    train_rewards: List[float] = []
    train_energy: List[float] = []
    train_cost: List[float] = []
    validation_history: List[Tuple[int, float]] = []

    for episode in tqdm(range(TRAIN_EPISODES), desc=f"Train seed={seed}"):
        agent.set_schedules(episode, TRAIN_EPISODES)
        obs_list = train_env.reset()
        socs = [float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0 for obs in obs_list]
        states = [bins.encode(obs, soc) for obs, soc in zip(obs_list, socs)]

        episode_reward = 0.0
        episode_energy = 0.0
        episode_cost = 0.0

        for _ in range(EPISODE_TIME_STEPS):
            action_ids = [agent.act(state, soc, training=True) for state, soc in zip(states, socs)]
            action_values = [float(action_levels[action_id]) for action_id in action_ids]
            next_obs_list, _, done, _ = train_env.step(build_actions_from_values(action_values))
            next_socs = [float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0 for obs in next_obs_list]
            next_states = [bins.encode(obs, soc) for obs, soc in zip(next_obs_list, next_socs)]

            local_rewards = [
                compute_energy_reward(obs_list[i], train_env.buildings[i], bins, action_values[i], action_max_abs)
                for i in range(len(train_env.buildings))
            ]

            for state, action_id, reward, next_state, next_soc in zip(states, action_ids, local_rewards, next_states, next_socs):
                agent.update(state, action_id, reward, next_state, next_soc, done)

            episode_reward += float(np.sum(local_rewards))
            episode_energy += district_grid_import(train_env)
            episode_cost += district_cost(train_env)

            states = next_states
            socs = next_socs
            obs_list = next_obs_list
            if done:
                break

        train_rewards.append(episode_reward)
        train_energy.append(episode_energy)
        train_cost.append(episode_cost)

        if (episode + 1) % EVAL_EVERY == 0 or episode == TRAIN_EPISODES - 1:
            validation_metrics = evaluate_agent(agent, bins, VALIDATION_EPISODES)
            validation_score = energy_score_against_fixed(validation_baseline, validation_metrics)
            validation_history.append((episode + 1, validation_score))
            if validation_score > best_score:
                best_score = validation_score
                best_q_table = copy.deepcopy(agent.q_table)

    if best_q_table:
        agent.q_table = best_q_table

    final_metrics = evaluate_agent(agent, bins, EVAL_EPISODES)

    return TrainingRunResult(
        seed=seed,
        validation_score=best_score,
        validation_energy=final_metrics.avg_energy_import,
        eval_metrics=final_metrics,
        agent=agent,
        train_rewards=train_rewards,
        train_energy=train_energy,
        train_cost=train_cost,
        validation_history=validation_history,
    )


def train_best_agent(bins: BinSet, action_levels: np.ndarray, validation_baseline: EvalMetrics) -> tuple[TrainingRunResult, pd.DataFrame]:
    results: List[TrainingRunResult] = []
    for seed in TRAINING_SEEDS:
        results.append(train_one_run(bins, action_levels, validation_baseline, seed))

    results.sort(key=lambda item: item.validation_score, reverse=True)
    best = results[0]
    run_summary_df = pd.DataFrame(
        [
            {
                "seed": result.seed,
                "validation_score": result.validation_score,
                "eval_energy_import": result.eval_metrics.avg_energy_import,
                "eval_cost": result.eval_metrics.avg_cost,
                "energy_sav_%": safe_pct(validation_baseline.avg_energy_import, result.eval_metrics.avg_energy_import),
                "cost_sav_%": safe_pct(validation_baseline.avg_cost, result.eval_metrics.avg_cost),
            }
            for result in results
        ]
    )
    return best, run_summary_df


def build_weekly_comparison_df(fixed: EvalMetrics, q_metrics: EvalMetrics) -> pd.DataFrame:
    rows: List[dict] = []
    for episode, (fixed_energy, fixed_cost, q_energy, q_cost) in enumerate(
        zip(fixed.episode_energies, fixed.episode_costs, q_metrics.episode_energies, q_metrics.episode_costs),
        start=1,
    ):
        energy_pct = safe_pct(fixed_energy, q_energy)
        cost_pct = safe_pct(fixed_cost, q_cost)
        rows.append(
            {
                "episode": episode,
                "fixed_energy": fixed_energy,
                "fixed_cost": fixed_cost,
                "q_energy": q_energy,
                "q_cost": q_cost,
                "energy_sav": fixed_energy - q_energy,
                "energy_sav_%": energy_pct,
                "cost_sav": fixed_cost - q_cost,
                "cost_sav_%": cost_pct,
                "energy_better": bool(energy_pct is not None and energy_pct > 0.0),
                "cost_better": bool(cost_pct is not None and cost_pct > 0.0),
            }
        )
    return pd.DataFrame(rows)


def build_hourly_comparison_df(combined_trace_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        combined_trace_df
        .groupby(["policy", "hour"], as_index=False)
        .agg(
            avg_energy_import=("energy_import", "mean"),
            avg_cost=("cost", "mean"),
            avg_load=("load", "mean"),
            avg_solar_generation=("solar_generation", "mean"),
            avg_soc=("soc", "mean"),
            avg_action=("action_mean", "mean"),
        )
    )

    fixed_df = grouped[grouped["policy"] == "fixed"].rename(
        columns={
            "avg_energy_import": "fixed_avg_energy_import",
            "avg_cost": "fixed_avg_cost",
            "avg_load": "fixed_avg_load",
            "avg_solar_generation": "fixed_avg_solar_generation",
            "avg_soc": "fixed_avg_soc",
            "avg_action": "fixed_avg_action",
        }
    )
    q_df = grouped[grouped["policy"] == "q_agent"].rename(
        columns={
            "avg_energy_import": "q_avg_energy_import",
            "avg_cost": "q_avg_cost",
            "avg_load": "q_avg_load",
            "avg_solar_generation": "q_avg_solar_generation",
            "avg_soc": "q_avg_soc",
            "avg_action": "q_avg_action",
        }
    )
    merged = fixed_df.merge(q_df, on="hour", how="inner")
    merged["energy_delta"] = merged["fixed_avg_energy_import"] - merged["q_avg_energy_import"]
    merged["cost_delta"] = merged["fixed_avg_cost"] - merged["q_avg_cost"]
    merged["energy_delta_%"] = [safe_pct(b, q) for b, q in zip(merged["fixed_avg_energy_import"], merged["q_avg_energy_import"])]
    merged["cost_delta_%"] = [safe_pct(b, q) for b, q in zip(merged["fixed_avg_cost"], merged["q_avg_cost"])]
    return merged.sort_values("hour").reset_index(drop=True)


def build_overview_df(weekly_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "energy_win_weeks", "value": int(weekly_df["energy_better"].sum())},
            {"metric": "cost_win_weeks", "value": int(weekly_df["cost_better"].sum())},
            {"metric": "both_win_weeks", "value": int((weekly_df["energy_better"] & weekly_df["cost_better"]).sum())},
            {"metric": "energy_win_hours", "value": int((hourly_df["energy_delta"] > 0.0).sum())},
            {"metric": "cost_win_hours", "value": int((hourly_df["cost_delta"] > 0.0).sum())},
            {"metric": "best_energy_week", "value": int(weekly_df.loc[weekly_df["energy_sav_%"].idxmax(), "episode"])},
            {"metric": "worst_energy_week", "value": int(weekly_df.loc[weekly_df["energy_sav_%"].idxmin(), "episode"])},
            {"metric": "best_energy_hour", "value": int(hourly_df.loc[hourly_df["energy_delta"].idxmax(), "hour"])},
            {"metric": "worst_energy_hour", "value": int(hourly_df.loc[hourly_df["energy_delta"].idxmin(), "hour"])},
        ]
    )


def plot_training(train_rewards: List[float], train_energy: List[float], train_cost: List[float], validation_history: List[Tuple[int, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Q_learning1404_02 - Training diagnostics", fontsize=11, fontweight="bold")

    axes[0, 0].plot(train_rewards, alpha=0.25, linewidth=0.8, color="steelblue")
    axes[0, 0].plot(moving_average(train_rewards, 20), linewidth=2, color="navy")
    axes[0, 0].set_title("Training reward")

    axes[0, 1].plot(train_energy, alpha=0.25, linewidth=0.8, color="teal")
    axes[0, 1].plot(moving_average(train_energy, 20), linewidth=2, color="darkslategray")
    axes[0, 1].set_title("Training district import")

    axes[1, 0].plot(train_cost, alpha=0.25, linewidth=0.8, color="darkorange")
    axes[1, 0].plot(moving_average(train_cost, 20), linewidth=2, color="saddlebrown")
    axes[1, 0].set_title("Training district cost")

    xs = [x for x, _ in validation_history]
    ys = [y for _, y in validation_history]
    axes[1, 1].plot(xs, ys, marker="o", linewidth=2, color="crimson")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Validation energy score vs fixed")

    for ax in axes.ravel():
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("reward")
    axes[0, 1].set_ylabel("kWh")
    axes[1, 0].set_ylabel("$")
    axes[1, 1].set_ylabel("score")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_training.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_weekly_comparison(weekly_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle("Q_learning1404_02 - Weekly Q vs fixed", fontsize=11, fontweight="bold")

    episodes = weekly_df["episode"].to_numpy()
    energy_colors = ["#2e8b57" if v > 0 else "#d04f4f" for v in weekly_df["energy_sav_%"]]
    cost_colors = ["#2e8b57" if v > 0 else "#d04f4f" for v in weekly_df["cost_sav_%"]]

    axes[0].bar(episodes, weekly_df["energy_sav_%"], color=energy_colors, alpha=0.9)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_title("Energy savings by week [%]")
    axes[0].set_ylabel("%")
    axes[0].grid(True, alpha=0.25, axis="y")

    axes[1].bar(episodes, weekly_df["cost_sav_%"], color=cost_colors, alpha=0.9)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Cost savings by week [%]")
    axes[1].set_ylabel("%")
    axes[1].grid(True, alpha=0.25, axis="y")
    axes[1].set_xlabel("Evaluation week")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_weekly_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hourly_comparison(hourly_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Q_learning1404_02 - Hourly Q vs fixed", fontsize=11, fontweight="bold")

    hours = hourly_df["hour"].to_numpy()
    energy_colors = ["#2e8b57" if v > 0 else "#d04f4f" for v in hourly_df["energy_delta"]]
    cost_colors = ["#2e8b57" if v > 0 else "#d04f4f" for v in hourly_df["cost_delta"]]

    axes[0, 0].plot(hours, hourly_df["fixed_avg_energy_import"], label="fixed", color="gray", linewidth=2)
    axes[0, 0].plot(hours, hourly_df["q_avg_energy_import"], label="q_agent", color="steelblue", linewidth=2)
    axes[0, 0].set_title("Average energy import by hour")
    axes[0, 0].set_ylabel("kWh")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.25)

    axes[0, 1].plot(hours, hourly_df["fixed_avg_cost"], label="fixed", color="gray", linewidth=2)
    axes[0, 1].plot(hours, hourly_df["q_avg_cost"], label="q_agent", color="darkorange", linewidth=2)
    axes[0, 1].set_title("Average cost by hour")
    axes[0, 1].set_ylabel("$")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].bar(hours, hourly_df["energy_delta"], color=energy_colors, alpha=0.9)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Energy advantage vs fixed by hour")
    axes[1, 0].set_ylabel("kWh saved")
    axes[1, 0].grid(True, alpha=0.25, axis="y")

    axes[1, 1].bar(hours, hourly_df["cost_delta"], color=cost_colors, alpha=0.9)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("Cost advantage vs fixed by hour")
    axes[1, 1].set_ylabel("$ saved")
    axes[1, 1].grid(True, alpha=0.25, axis="y")

    for ax in axes.ravel():
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(1, 25))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_hourly_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(RANDOM_SEED)
    sep = "=" * 84

    print(sep)
    print("  Q_learning1404_02 - Energy-first multi-restart Q-learning")
    print("  Focus: beat fixed baseline in energy using weather forecasts and occupancy proxy")
    print(sep)
    print(f"  citylearn version    : {citylearn.__version__}")
    print(f"  dataset              : {DATASET_NAME}")
    print(f"  episode_time_steps   : {EPISODE_TIME_STEPS} ({EPISODE_TIME_STEPS // 24} days)")
    print(f"  train_episodes/run   : {TRAIN_EPISODES}")
    print(f"  validation_episodes  : {VALIDATION_EPISODES}")
    print(f"  eval_episodes        : {EVAL_EPISODES}")
    print(f"  training_seeds       : {TRAINING_SEEDS}")
    print("  reward               : strong energy-only reward with grid-charge penalty")
    print("  state                : hour + solar forecast + pv surplus + occupancy proxy + soc")

    print("\n[1/6] Inspect environment and state bins...")
    probe_env = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    bins = BinSet(probe_env)
    bins.fit(probe_env)
    bins.print_summary()

    action_low = float(probe_env.action_space[0].low[0])
    action_high = float(probe_env.action_space[0].high[0])
    action_max_abs = min(abs(action_low), abs(action_high))
    action_levels = np.array([-1.0, -0.5, 0.0, 0.15, 0.35, 0.60], dtype=np.float32) * action_max_abs
    print(f"  action bounds        : [{action_low:.6f}, {action_high:.6f}]")
    print(f"  discrete actions     : {action_levels.tolist()}")

    print("\n[2/6] Fixed baseline only...")
    fixed_baseline = evaluate_fixed_policy(0.0, EVAL_EPISODES)
    validation_baseline = evaluate_fixed_policy(0.0, VALIDATION_EPISODES)
    print(f"  fixed baseline       : energy={fixed_baseline.avg_energy_import:.4f} kWh, cost={fixed_baseline.avg_cost:.4f} $")

    print("\n[3/6] Train multiple Q-agent restarts and keep the best...")
    best_run, run_summary_df = train_best_agent(bins, action_levels, validation_baseline)
    q_metrics = best_run.eval_metrics
    q_score = energy_score_against_fixed(fixed_baseline, q_metrics)
    print(f"  best seed            : {best_run.seed}")
    print(f"  q_learning focused   : energy={q_metrics.avg_energy_import:.4f} kWh, cost={q_metrics.avg_cost:.4f} $")
    print(f"  score vs fixed       : {q_score:+.4f}")
    print(f"  visited states       : {best_run.agent.total_states}")
    print(run_summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\n[4/6] Collect detailed evaluation traces...")
    fixed_trace = evaluate_fixed_with_trace(0.0, bins, EVAL_EPISODES, label="fixed")
    q_trace = evaluate_agent_with_trace(best_run.agent, bins, EVAL_EPISODES, label="q_agent")
    combined_trace_df = pd.concat([fixed_trace.trace_df, q_trace.trace_df], ignore_index=True)

    summary_df = pd.DataFrame(
        [
            {
                "policy": "fixed_baseline",
                "avg_energy_import": fixed_trace.metrics.avg_energy_import,
                "avg_cost": fixed_trace.metrics.avg_cost,
                "energy_sav": 0.0,
                "energy_sav_%": 0.0,
                "cost_sav": 0.0,
                "cost_sav_%": 0.0,
                "score_vs_fixed": 0.0,
            },
            {
                "policy": "q_learning_weather_occ_best",
                "avg_energy_import": q_trace.metrics.avg_energy_import,
                "avg_cost": q_trace.metrics.avg_cost,
                "energy_sav": fixed_trace.metrics.avg_energy_import - q_trace.metrics.avg_energy_import,
                "energy_sav_%": safe_pct(fixed_trace.metrics.avg_energy_import, q_trace.metrics.avg_energy_import),
                "cost_sav": fixed_trace.metrics.avg_cost - q_trace.metrics.avg_cost,
                "cost_sav_%": safe_pct(fixed_trace.metrics.avg_cost, q_trace.metrics.avg_cost),
                "score_vs_fixed": q_score,
            },
        ]
    )

    weekly_df = build_weekly_comparison_df(fixed_trace.metrics, q_trace.metrics)
    hourly_df = build_hourly_comparison_df(combined_trace_df)
    overview_df = build_overview_df(weekly_df, hourly_df)

    print("\n[5/6] Save results and plots...")
    summary_df.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)
    weekly_df.to_csv(f"{OUTPUT_PREFIX}_per_episode.csv", index=False)
    hourly_df.to_csv(f"{OUTPUT_PREFIX}_per_hour.csv", index=False)
    overview_df.to_csv(f"{OUTPUT_PREFIX}_overview.csv", index=False)
    combined_trace_df.to_csv(f"{OUTPUT_PREFIX}_trace.csv", index=False)
    run_summary_df.to_csv(f"{OUTPUT_PREFIX}_run_summary.csv", index=False)

    training_df = pd.DataFrame(
        {
            "episode": np.arange(1, TRAIN_EPISODES + 1),
            "training_reward": best_run.train_rewards,
            "training_energy": best_run.train_energy,
            "training_cost": best_run.train_cost,
        }
    )
    training_df.to_csv(f"{OUTPUT_PREFIX}_training_log.csv", index=False)

    plot_training(best_run.train_rewards, best_run.train_energy, best_run.train_cost, best_run.validation_history)
    plot_weekly_comparison(weekly_df)
    plot_hourly_comparison(hourly_df)

    print("\n[6/6] Final summary...")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n  Diagnostic overview:")
    print(overview_df.to_string(index=False))

    print("\n  Saved files:")
    for file_name in [
        f"{OUTPUT_PREFIX}_results.csv",
        f"{OUTPUT_PREFIX}_per_episode.csv",
        f"{OUTPUT_PREFIX}_per_hour.csv",
        f"{OUTPUT_PREFIX}_overview.csv",
        f"{OUTPUT_PREFIX}_trace.csv",
        f"{OUTPUT_PREFIX}_run_summary.csv",
        f"{OUTPUT_PREFIX}_training_log.csv",
        f"{OUTPUT_PREFIX}_training.png",
        f"{OUTPUT_PREFIX}_weekly_comparison.png",
        f"{OUTPUT_PREFIX}_hourly_comparison.png",
    ]:
        print(f"    {file_name}")

    print(sep)


if __name__ == "__main__":
    main()