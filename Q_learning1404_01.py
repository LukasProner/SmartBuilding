#!/usr/bin/env python
"""
Q_learning1404_01.py
===================

Analysis-focused version of the weather/occupancy Q-learning experiment.

Goal:
- keep the same learning setup as Q_learning1304_02,
- but produce clearer diagnostics showing exactly when the learned Q-agent is
  better or worse than the fixed baseline in energy and cost.

Outputs:
- summary table
- per-week comparison table
- per-hour aggregated comparison table
- step-level evaluation trace table
- plots that clearly highlight win/loss regions
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
OUTPUT_PREFIX = "q_learning1404_01"
RANDOM_SEED = 42

FAST_MODE = True

if FAST_MODE:
    EPISODE_TIME_STEPS = 24 * 7
    TRAIN_EPISODES = 360
    VALIDATION_EPISODES = 12
    EVAL_EPISODES = 52
    EVAL_EVERY = 18
else:
    EPISODE_TIME_STEPS = 24 * 7
    TRAIN_EPISODES = 700
    VALIDATION_EPISODES = 12
    EVAL_EPISODES = 52
    EVAL_EVERY = 20

ALPHA_START = 0.18
ALPHA_END = 0.04
GAMMA = 0.985
EPSILON_START = 1.0
EPSILON_END = 0.03

SOC_EMPTY_THRESH = 0.05
SOC_FULL_THRESH = 0.95

SOC_BINS = np.array([0.15, 0.40, 0.70, 0.90], dtype=np.float32)
SOLAR_FALLBACK = np.array([120.0, 500.0, 950.0], dtype=np.float32)
TEMP_FALLBACK = np.array([10.0, 18.0, 26.0], dtype=np.float32)
LOAD_FALLBACK = np.array([0.65, 1.60], dtype=np.float32)

StateT = Tuple[int, int, int, int, int, int]


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
        self.k_temp_6h = first_key(idx, ["outdoor_dry_bulb_temperature_predicted_6h", "outdoor_dry_bulb_temperature_predicted_1"])
        self.k_temp_12h = first_key(idx, ["outdoor_dry_bulb_temperature_predicted_12h", "outdoor_dry_bulb_temperature_predicted_2"])
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
        self.i_temp_6h = idx[self.k_temp_6h] if self.k_temp_6h else None
        self.i_temp_12h = idx[self.k_temp_12h] if self.k_temp_12h else None
        self.i_diff_6h = idx[self.k_diff_6h] if self.k_diff_6h else None
        self.i_dir_6h = idx[self.k_dir_6h] if self.k_dir_6h else None
        self.i_diff_12h = idx[self.k_diff_12h] if self.k_diff_12h else None
        self.i_dir_12h = idx[self.k_dir_12h] if self.k_dir_12h else None

        self.solar_bins: np.ndarray = SOLAR_FALLBACK
        self.temp_bins: np.ndarray = TEMP_FALLBACK
        self.load_bins: np.ndarray = LOAD_FALLBACK
        self.soc_bins: np.ndarray = SOC_BINS

        self._solar_samples: List[float] = []
        self._temp_samples: List[float] = []
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

    def temperature_forecast(self, obs: List[float]) -> float:
        values: List[float] = []
        for i_temp in [self.i_temp_6h, self.i_temp_12h]:
            if i_temp is not None:
                values.append(float(obs[i_temp]))
        return float(np.mean(values)) if values else 20.0

    def pv_surplus_flag(self, obs: List[float]) -> int:
        load = float(obs[self.i_load]) if self.i_load is not None else 0.0
        solar = float(obs[self.i_solar_gen]) if self.i_solar_gen is not None else 0.0
        return int(solar > load + 0.1)

    def fit(self, env: CityLearnEnv, episodes: int = 6) -> None:
        solar_values: List[float] = []
        temp_values: List[float] = []
        load_values: List[float] = []

        for _ in range(episodes):
            obs_list = env.reset()
            for _ in range(EPISODE_TIME_STEPS):
                for obs in obs_list:
                    solar_values.append(self.solar_forecast(obs))
                    temp_values.append(self.temperature_forecast(obs))
                    if self.i_load is not None:
                        load_values.append(float(obs[self.i_load]))
                obs_list, _, done, _ = env.step([[0.0] for _ in env.buildings])
                if done:
                    break

        self._solar_samples = solar_values
        self._temp_samples = temp_values
        self._load_samples = load_values
        self.solar_bins = quantile_bins(solar_values, 4, SOLAR_FALLBACK)
        self.temp_bins = quantile_bins(temp_values, 4, TEMP_FALLBACK)
        self.load_bins = quantile_bins(load_values, 3, LOAD_FALLBACK)

    def encode(self, obs: List[float], soc: float) -> StateT:
        hour = int(np.clip((int(round(float(obs[self.i_hour]))) - 1) if self.i_hour is not None else 0, 0, 23))
        return (
            hour,
            dig(self.solar_forecast(obs), self.solar_bins),
            dig(self.temperature_forecast(obs), self.temp_bins),
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
        print(f"    {'temp forecast':<18}  {str(self.k_temp_6h)} / {str(self.k_temp_12h):<20}  {summarize_values(self._temp_samples)}")
        print(f"    {'':<18}  {'':<42}  bins: {format_bins(self.temp_bins)}")
        print(f"    {'occupancy proxy':<18}  {str(self.k_load):<42}  {summarize_values(self._load_samples)}")
        print(f"    {'':<18}  {'':<42}  bins: {format_bins(self.load_bins)}")
        print(f"    {'pv surplus flag':<18}  {str(self.k_solar_gen):<42}  0=no, 1=current solar > load")
        print(f"    {'battery soc':<18}  {str(self.k_soc):<42}  bins: {format_bins(self.soc_bins)}")
        print(f"    {sep}")
        state_count = 24 * (len(self.solar_bins) + 1) * (len(self.temp_bins) + 1) * 2 * (len(self.load_bins) + 1) * (len(self.soc_bins) + 1)
        print(f"    State space: 24 x {len(self.solar_bins)+1} x {len(self.temp_bins)+1} x 2 x {len(self.load_bins)+1} x {len(self.soc_bins)+1} = {state_count:,}")
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

    def valid_action_ids(self, soc: float) -> List[int]:
        valid: List[int] = []
        for i, value in enumerate(self.action_levels):
            if soc <= SOC_EMPTY_THRESH and value < 0.0:
                continue
            if soc >= SOC_FULL_THRESH and value > 0.0:
                continue
            valid.append(i)
        return valid or [int(np.argmin(np.abs(self.action_levels)))]

    def set_schedules(self, episode: int, total_episodes: int) -> None:
        progress = float(episode) / max(1, total_episodes - 1)
        self.alpha = float(ALPHA_END + (ALPHA_START - ALPHA_END) * (1.0 - progress))
        self.epsilon = float(EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-5.0 * progress))

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

    action_scale = abs(action_value) / max(1e-6, action_max_abs)
    pv_capture = min(storage_charge, export_wo_storage)
    harmful_grid_charge = max(0.0, storage_charge - export_wo_storage)
    import_reduction = max(0.0, import_wo_storage - net_import)

    reward = -(1.35 * net_import + 0.05 * action_scale + 0.14 * harmful_grid_charge)
    reward += 0.70 * pv_capture
    reward += 0.30 * import_reduction

    if pv_surplus and storage_charge > 0.0:
        reward += 0.10
    if export_wo_storage > 0.05 and storage_discharge > 0.0:
        reward -= 0.08
    if harmful_grid_charge > 0.05:
        reward -= 0.35 * harmful_grid_charge

    return float(reward)


def energy_score_against_fixed(fixed: EvalMetrics, other: EvalMetrics) -> float:
    energy_pct = safe_pct(fixed.avg_energy_import, other.avg_energy_import) or -500.0
    cost_pct = safe_pct(fixed.avg_cost, other.avg_cost) or -500.0
    return 0.90 * energy_pct + 0.10 * cost_pct


def heuristic_action_value(obs: List[float], bins: BinSet, action_low: float, action_high: float) -> float:
    soc = float(obs[bins.i_soc]) if bins.i_soc is not None else 0.0
    load = float(obs[bins.i_load]) if bins.i_load is not None else 0.0
    solar = max(0.0, float(obs[bins.i_solar_gen])) if bins.i_solar_gen is not None else 0.0
    net = float(obs[bins.i_net]) if bins.i_net is not None else max(0.0, load - solar)

    if solar > load + 0.15 and soc < 0.92:
        return 0.5 * action_high
    if net < -0.10 and soc < 0.95:
        return action_high
    if net > 1.40 and soc > 0.12:
        return 0.5 * action_low
    return 0.0


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


def evaluate_heuristic_policy(episodes: int, bins: BinSet, action_low: float, action_high: float) -> EvalMetrics:
    env = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    ep_energies: List[float] = []
    ep_costs: List[float] = []

    for _ in range(episodes):
        obs_list = env.reset()
        ep_energy = 0.0
        ep_cost = 0.0
        for _ in range(EPISODE_TIME_STEPS):
            actions = build_actions_from_values([heuristic_action_value(obs, bins, action_low, action_high) for obs in obs_list])
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


def train_agent(bins: BinSet, action_levels: np.ndarray, validation_baseline: EvalMetrics) -> tuple[SharedTabularQAgent, List[float], List[float], List[float], List[Tuple[int, float]]]:
    train_env = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    agent = SharedTabularQAgent(action_levels)
    action_max_abs = float(np.max(np.abs(action_levels)))

    best_score = -float("inf")
    best_q_table: Dict[StateT, np.ndarray] = {}
    train_rewards: List[float] = []
    train_energy: List[float] = []
    train_cost: List[float] = []
    validation_history: List[Tuple[int, float]] = []

    for episode in tqdm(range(TRAIN_EPISODES), desc="Train weather-occ-diagnostics"):
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

    return agent, train_rewards, train_energy, train_cost, validation_history


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
    merged["energy_delta_%"] = [
        safe_pct(b, q)
        for b, q in zip(merged["fixed_avg_energy_import"], merged["q_avg_energy_import"])
    ]
    merged["cost_delta_%"] = [
        safe_pct(b, q)
        for b, q in zip(merged["fixed_avg_cost"], merged["q_avg_cost"])
    ]
    return merged.sort_values("hour").reset_index(drop=True)


def build_summary_df(fixed: EvalMetrics, heuristic: EvalMetrics, q_metrics: EvalMetrics) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy": "fixed_baseline",
                "avg_energy_import": fixed.avg_energy_import,
                "avg_cost": fixed.avg_cost,
                "energy_sav": 0.0,
                "energy_sav_%": 0.0,
                "cost_sav": 0.0,
                "cost_sav_%": 0.0,
                "score_vs_fixed": 0.0,
            },
            {
                "policy": "heuristic_pv_rule",
                "avg_energy_import": heuristic.avg_energy_import,
                "avg_cost": heuristic.avg_cost,
                "energy_sav": fixed.avg_energy_import - heuristic.avg_energy_import,
                "energy_sav_%": safe_pct(fixed.avg_energy_import, heuristic.avg_energy_import),
                "cost_sav": fixed.avg_cost - heuristic.avg_cost,
                "cost_sav_%": safe_pct(fixed.avg_cost, heuristic.avg_cost),
                "score_vs_fixed": energy_score_against_fixed(fixed, heuristic),
            },
            {
                "policy": "q_learning_weather_occ",
                "avg_energy_import": q_metrics.avg_energy_import,
                "avg_cost": q_metrics.avg_cost,
                "energy_sav": fixed.avg_energy_import - q_metrics.avg_energy_import,
                "energy_sav_%": safe_pct(fixed.avg_energy_import, q_metrics.avg_energy_import),
                "cost_sav": fixed.avg_cost - q_metrics.avg_cost,
                "cost_sav_%": safe_pct(fixed.avg_cost, q_metrics.avg_cost),
                "score_vs_fixed": energy_score_against_fixed(fixed, q_metrics),
            },
        ]
    )


def build_overview_df(weekly_df: pd.DataFrame, hourly_df: pd.DataFrame) -> pd.DataFrame:
    energy_win_weeks = int(weekly_df["energy_better"].sum())
    cost_win_weeks = int(weekly_df["cost_better"].sum())
    both_win_weeks = int((weekly_df["energy_better"] & weekly_df["cost_better"]).sum())
    energy_win_hours = int((hourly_df["energy_delta"] > 0.0).sum())
    cost_win_hours = int((hourly_df["cost_delta"] > 0.0).sum())
    best_energy_week = int(weekly_df.loc[weekly_df["energy_sav_%"].idxmax(), "episode"])
    worst_energy_week = int(weekly_df.loc[weekly_df["energy_sav_%"].idxmin(), "episode"])
    best_cost_week = int(weekly_df.loc[weekly_df["cost_sav_%"].idxmax(), "episode"])
    worst_cost_week = int(weekly_df.loc[weekly_df["cost_sav_%"].idxmin(), "episode"])
    best_energy_hour = int(hourly_df.loc[hourly_df["energy_delta"].idxmax(), "hour"])
    worst_energy_hour = int(hourly_df.loc[hourly_df["energy_delta"].idxmin(), "hour"])
    best_cost_hour = int(hourly_df.loc[hourly_df["cost_delta"].idxmax(), "hour"])
    worst_cost_hour = int(hourly_df.loc[hourly_df["cost_delta"].idxmin(), "hour"])

    return pd.DataFrame(
        [
            {"metric": "energy_win_weeks", "value": energy_win_weeks},
            {"metric": "cost_win_weeks", "value": cost_win_weeks},
            {"metric": "both_win_weeks", "value": both_win_weeks},
            {"metric": "energy_win_hours", "value": energy_win_hours},
            {"metric": "cost_win_hours", "value": cost_win_hours},
            {"metric": "best_energy_week", "value": best_energy_week},
            {"metric": "worst_energy_week", "value": worst_energy_week},
            {"metric": "best_cost_week", "value": best_cost_week},
            {"metric": "worst_cost_week", "value": worst_cost_week},
            {"metric": "best_energy_hour", "value": best_energy_hour},
            {"metric": "worst_energy_hour", "value": worst_energy_hour},
            {"metric": "best_cost_hour", "value": best_cost_hour},
            {"metric": "worst_cost_hour", "value": worst_cost_hour},
        ]
    )


def plot_training(train_rewards: List[float], train_energy: List[float], train_cost: List[float], validation_history: List[Tuple[int, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Q_learning1404_01 - Training diagnostics",
        fontsize=11,
        fontweight="bold",
    )

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
    axes[1, 1].set_title("Validation score vs fixed baseline")

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
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Q_learning1404_01 - Where Q-agent beats fixed by week",
        fontsize=11,
        fontweight="bold",
    )

    episodes = weekly_df["episode"].to_numpy()
    energy_colors = ["#2e8b57" if v > 0 else "#d04f4f" for v in weekly_df["energy_sav_%"]]
    cost_colors = ["#2e8b57" if v > 0 else "#d04f4f" for v in weekly_df["cost_sav_%"]]

    axes[0, 0].bar(episodes, weekly_df["energy_sav_%"], color=energy_colors, alpha=0.9)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Energy savings by week [%]")
    axes[0, 0].set_ylabel("%")
    axes[0, 0].grid(True, alpha=0.25, axis="y")

    axes[0, 1].bar(episodes, weekly_df["cost_sav_%"], color=cost_colors, alpha=0.9)
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_title("Cost savings by week [%]")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].grid(True, alpha=0.25, axis="y")

    axes[1, 0].plot(episodes, weekly_df["fixed_energy"], label="fixed", color="gray", linewidth=2)
    axes[1, 0].plot(episodes, weekly_df["q_energy"], label="q_agent", color="steelblue", linewidth=2)
    axes[1, 0].set_title("Weekly energy import")
    axes[1, 0].set_ylabel("kWh")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(episodes, weekly_df["fixed_cost"], label="fixed", color="gray", linewidth=2)
    axes[1, 1].plot(episodes, weekly_df["q_cost"], label="q_agent", color="darkorange", linewidth=2)
    axes[1, 1].set_title("Weekly cost")
    axes[1, 1].set_ylabel("$")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.25)

    for ax in axes.ravel():
        ax.set_xlabel("Evaluation week")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_weekly_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hourly_comparison(hourly_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Q_learning1404_01 - Where Q-agent beats fixed by hour of day",
        fontsize=11,
        fontweight="bold",
    )

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


def plot_win_loss_matrix(weekly_df: pd.DataFrame) -> None:
    matrix = np.vstack([
        weekly_df["energy_better"].astype(int).to_numpy(),
        weekly_df["cost_better"].astype(int).to_numpy(),
    ])

    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("Q-agent win/loss map vs fixed (green = better)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Energy", "Cost"])
    ax.set_xticks(np.arange(EVAL_EPISODES))
    ax.set_xticklabels(np.arange(1, EVAL_EPISODES + 1), fontsize=7)
    ax.set_xlabel("Evaluation week")

    for x, energy_better, cost_better in zip(range(EVAL_EPISODES), weekly_df["energy_better"], weekly_df["cost_better"]):
        ax.text(x, 0, "W" if energy_better else "L", ha="center", va="center", fontsize=7)
        ax.text(x, 1, "W" if cost_better else "L", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_win_loss_matrix.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(RANDOM_SEED)
    sep = "=" * 84

    print(sep)
    print("  Q_learning1404_01 - Detailed diagnostics for fixed vs Q-agent")
    print("  Focus: show clearly when the Q-agent is energetically and financially better")
    print(sep)
    print(f"  citylearn version    : {citylearn.__version__}")
    print(f"  dataset              : {DATASET_NAME}")
    print(f"  episode_time_steps   : {EPISODE_TIME_STEPS} ({EPISODE_TIME_STEPS // 24} days)")
    print(f"  train_episodes       : {TRAIN_EPISODES}")
    print(f"  validation_episodes  : {VALIDATION_EPISODES}")
    print(f"  eval_episodes        : {EVAL_EPISODES}")
    print("  reward               : energy_pv_only")
    print("  state                : hour + solar forecast + temperature forecast + occupancy proxy + pv surplus + soc")

    print("\n[1/6] Inspect environment and state bins...")
    probe_env = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    bins = BinSet(probe_env)
    bins.fit(probe_env)
    bins.print_summary()

    action_low = float(probe_env.action_space[0].low[0])
    action_high = float(probe_env.action_space[0].high[0])
    action_max_abs = min(abs(action_low), abs(action_high))
    action_levels = np.array([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0], dtype=np.float32) * action_max_abs
    print(f"  action bounds        : [{action_low:.6f}, {action_high:.6f}]")
    print(f"  discrete actions     : {action_levels.tolist()}")

    print("\n[2/6] Fixed and heuristic baselines...")
    fixed_baseline = evaluate_fixed_policy(0.0, EVAL_EPISODES)
    validation_baseline = evaluate_fixed_policy(0.0, VALIDATION_EPISODES)
    heuristic_baseline = evaluate_heuristic_policy(EVAL_EPISODES, bins, action_low, action_high)
    print(f"  fixed baseline       : energy={fixed_baseline.avg_energy_import:.4f} kWh, cost={fixed_baseline.avg_cost:.4f} $")
    print(f"  heuristic pv rule    : energy={heuristic_baseline.avg_energy_import:.4f} kWh, cost={heuristic_baseline.avg_cost:.4f} $")

    print("\n[3/6] Train focused Q-agent...")
    agent, train_rewards, train_energy, train_cost, validation_history = train_agent(bins, action_levels, validation_baseline)
    q_metrics = evaluate_agent(agent, bins, EVAL_EPISODES)
    q_score = energy_score_against_fixed(fixed_baseline, q_metrics)
    print(f"  q_learning focused   : energy={q_metrics.avg_energy_import:.4f} kWh, cost={q_metrics.avg_cost:.4f} $")
    print(f"  score vs fixed       : {q_score:+.4f}")
    print(f"  visited states       : {agent.total_states}")

    print("\n[4/6] Collect detailed evaluation traces...")
    fixed_trace = evaluate_fixed_with_trace(0.0, bins, EVAL_EPISODES, label="fixed")
    q_trace = evaluate_agent_with_trace(agent, bins, EVAL_EPISODES, label="q_agent")
    combined_trace_df = pd.concat([fixed_trace.trace_df, q_trace.trace_df], ignore_index=True)

    summary_df = build_summary_df(fixed_trace.metrics, heuristic_baseline, q_trace.metrics)
    weekly_df = build_weekly_comparison_df(fixed_trace.metrics, q_trace.metrics)
    hourly_df = build_hourly_comparison_df(combined_trace_df)
    overview_df = build_overview_df(weekly_df, hourly_df)

    print("\n[5/6] Save results and plots...")
    summary_df.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)
    weekly_df.to_csv(f"{OUTPUT_PREFIX}_per_episode.csv", index=False)
    hourly_df.to_csv(f"{OUTPUT_PREFIX}_per_hour.csv", index=False)
    combined_trace_df.to_csv(f"{OUTPUT_PREFIX}_trace.csv", index=False)
    overview_df.to_csv(f"{OUTPUT_PREFIX}_overview.csv", index=False)

    training_df = pd.DataFrame(
        {
            "episode": np.arange(1, TRAIN_EPISODES + 1),
            "training_reward": train_rewards,
            "training_energy": train_energy,
            "training_cost": train_cost,
        }
    )
    training_df.to_csv(f"{OUTPUT_PREFIX}_training_log.csv", index=False)

    plot_training(train_rewards, train_energy, train_cost, validation_history)
    plot_weekly_comparison(weekly_df)
    plot_hourly_comparison(hourly_df)
    plot_win_loss_matrix(weekly_df)

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
        f"{OUTPUT_PREFIX}_training_log.csv",
        f"{OUTPUT_PREFIX}_training.png",
        f"{OUTPUT_PREFIX}_weekly_comparison.png",
        f"{OUTPUT_PREFIX}_hourly_comparison.png",
        f"{OUTPUT_PREFIX}_win_loss_matrix.png",
    ]:
        print(f"    {file_name}")

    print(sep)


if __name__ == "__main__":
    main()