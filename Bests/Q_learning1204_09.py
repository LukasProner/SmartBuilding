"""Q_learning0904_02 - CityLearn based tabular Q-learning.

Zaklad je podobny ako povodny Q_learning.py, ale prostredie je CityLearnEnv.
Pouziva sa:
- stav z CityLearn observations, pripadne z env.get_state(...) ak je dostupne,
- naklady z env.costs() ak je dostupne, inak z env.net_electricity_consumption_cost.

Ciel: porovnat Q-learning vs fixna strategia z pohladu energie a realnych nakladov.
"""

from __future__ import annotations

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
RANDOM_SEED = 42

FAST_MODE = True

if FAST_MODE:
    # Rychly iteracny rezim: menej krokov a evaluacii, aby bol skript citelne sviznejsi.
    EPISODE_TIME_STEPS = 24 * 7
    TRAIN_EPISODES = 350
    EVAL_EPISODES = 20
else:
    EPISODE_TIME_STEPS = 24 * 14
    TRAIN_EPISODES = 1000
    EVAL_EPISODES = 30

# Stabilizacia: trening sa moze ukoncit skor, ked su reward aj cost stabilne.
STABILITY_WINDOW = 20
STABILITY_TOL = 0.01
STABILITY_PATIENCE = 15
STABILITY_MIN_EPISODES = 120
STABILITY_CHECK_EVERY = 10
STABILITY_RELAXED_MIN_EPISODES = 300

ALPHA = 0.20
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.97

# Mensi pocet akcii = menej hladania v priestore akcii a rychlejsie ucenie.
ACTION_LEVELS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
FIXED_BASELINE_ACTION = 0.0


# ============================================================================
# HELPERS
# ============================================================================


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def resolve_schema(dataset_name: str) -> str:
    localapp = os.environ.get("LOCALAPPDATA", "")
    version_tag = f"v{citylearn.__version__}"
    cached = (
        Path(localapp)
        / "intelligent-environments-lab"
        / "citylearn"
        / "Cache"
        / version_tag
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
    names = env.observation_names[0]
    return {name: i for i, name in enumerate(names)}


def existing_key(obs_index: Dict[str, int], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in obs_index:
            return name
    return None


def build_constant_actions(env: CityLearnEnv, value: float) -> List[List[float]]:
    actions: List[List[float]] = []
    for space in env.action_space:
        dim = int(space.shape[0])
        actions.append([float(value)] * dim)
    return actions


def build_actions_from_ids(env: CityLearnEnv, action_ids: List[int]) -> List[List[float]]:
    actions: List[List[float]] = []
    for i, space in enumerate(env.action_space):
        dim = int(space.shape[0])
        v = float(ACTION_LEVELS[action_ids[i]])
        actions.append([v] * dim)
    return actions


def moving_average(values: List[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if window <= 1 or arr.size == 0:
        return arr
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(arr, kernel, mode="same")


def reward_to_building_array(reward: object, n_buildings: int) -> np.ndarray:
    arr = np.asarray(reward, dtype=np.float32).reshape(-1)
    if arr.size == n_buildings:
        return arr
    if arr.size == 1:
        return np.full((n_buildings,), float(arr[0]) / max(1, n_buildings), dtype=np.float32)

    out = np.zeros((n_buildings,), dtype=np.float32)
    k = min(n_buildings, arr.size)
    out[:k] = arr[:k]
    return out


def collect_state_samples(env: CityLearnEnv, max_episodes: int = 12) -> Dict[str, List[float]]:
    obs_index = get_obs_index(env)

    def first_existing(candidates: List[str]) -> Optional[int]:
        for name in candidates:
            if name in obs_index:
                return obs_index[name]
        return None

    i_temp = first_existing([
        "outdoor_dry_bulb_temperature_predicted_6h",
        "outdoor_dry_bulb_temperature_predicted_1",
        "outdoor_dry_bulb_temperature",
    ])
    i_humidity = first_existing([
        "outdoor_relative_humidity_predicted_6h",
        "outdoor_relative_humidity_predicted_1",
        "outdoor_relative_humidity",
    ])
    i_solar_diff = first_existing([
        "diffuse_solar_irradiance_predicted_6h",
        "diffuse_solar_irradiance_predicted_1",
        "diffuse_solar_irradiance",
    ])
    i_solar_dir = first_existing([
        "direct_solar_irradiance_predicted_6h",
        "direct_solar_irradiance_predicted_1",
        "direct_solar_irradiance",
    ])
    i_price = first_existing([
        "electricity_pricing_predicted_6h",
        "electricity_pricing_predicted_1",
        "electricity_pricing",
    ])
    i_load = first_existing(["non_shiftable_load"])
    i_soc = first_existing(["electrical_storage_soc"])

    samples: Dict[str, List[float]] = {
        "temp": [],
        "humidity": [],
        "solar": [],
        "price": [],
        "load": [],
        "soc": [],
    }

    for _ in range(max_episodes):
        obs_list = env.reset()
        for _ in range(min(EPISODE_TIME_STEPS, 48)):
            actions = build_constant_actions(env, FIXED_BASELINE_ACTION)
            next_obs_list, _, done, _ = env.step(actions)
            for obs in next_obs_list:
                if i_temp is not None and i_temp < len(obs):
                    samples["temp"].append(float(obs[i_temp]))
                if i_humidity is not None and i_humidity < len(obs):
                    samples["humidity"].append(float(obs[i_humidity]))
                solar_present = False
                solar = 0.0
                if i_solar_diff is not None and i_solar_diff < len(obs):
                    solar += float(obs[i_solar_diff])
                    solar_present = True
                if i_solar_dir is not None and i_solar_dir < len(obs):
                    solar += float(obs[i_solar_dir])
                    solar_present = True
                if solar_present:
                    samples["solar"].append(solar)
                if i_price is not None and i_price < len(obs):
                    samples["price"].append(float(obs[i_price]))
                if i_load is not None and i_load < len(obs):
                    samples["load"].append(float(obs[i_load]))
                if i_soc is not None and i_soc < len(obs):
                    samples["soc"].append(float(obs[i_soc]))
            if done:
                break

    return samples


def quantile_bins(values: List[float], n_bins: int, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size < max(20, n_bins * 3):
        return fallback
    qs = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)[1:-1]
    bins = np.unique(np.quantile(arr, qs))
    if bins.size == 0:
        return fallback
    return bins.astype(np.float32)


def get_last_district_energy_import(env: CityLearnEnv) -> float:
    if len(env.net_electricity_consumption) == 0:
        return 0.0
    return max(float(env.net_electricity_consumption[-1]), 0.0)


def get_last_district_cost(env: CityLearnEnv) -> float:
    """Try env.costs() first, fallback to env.net_electricity_consumption_cost."""

    # V instalovanej verzii je toto najlacnejsia cesta, preto je prva.
    if hasattr(env, "net_electricity_consumption_cost") and len(env.net_electricity_consumption_cost) > 0:
        return float(env.net_electricity_consumption_cost[-1])

    # Optional API in some CityLearn versions
    if hasattr(env, "costs"):
        try:
            c = env.costs()
            if isinstance(c, dict):
                # try district total cost keys
                for key in ("cost_total", "cost", "district_cost"):
                    if key in c:
                        v = c[key]
                        if isinstance(v, (list, tuple, np.ndarray)):
                            return float(v[-1])
                        return float(v)
            elif isinstance(c, (list, tuple, np.ndarray)) and len(c) > 0:
                return float(c[-1])
        except Exception:
            pass

    return 0.0


# ============================================================================
# STATE ENCODER
# ============================================================================


class StateEncoder:
    """Extract state from env.get_state(...) if present, otherwise observations."""

    def __init__(self, env: CityLearnEnv, samples: Optional[Dict[str, List[float]]] = None):
        self.env = env
        self.obs_index = get_obs_index(env)

        samples = samples or {}
        self.TEMP_BINS = quantile_bins(samples.get("temp", []), 6, np.array([-10.0, 0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32))
        self.HUMIDITY_BINS = quantile_bins(samples.get("humidity", []), 5, np.array([20.0, 40.0, 60.0, 80.0], dtype=np.float32))
        self.SOLAR_BINS = quantile_bins(samples.get("solar", []), 6, np.array([50.0, 150.0, 300.0, 500.0, 800.0], dtype=np.float32))
        self.PRICE_BINS = quantile_bins(samples.get("price", []), 6, np.array([0.08, 0.14, 0.20, 0.30, 0.45], dtype=np.float32))
        self.LOAD_BINS = quantile_bins(samples.get("load", []), 5, np.array([0.5, 1.5, 3.0, 6.0], dtype=np.float32))
        self.SOC_BINS = quantile_bins(samples.get("soc", []), 6, np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32))

        self.k_hour = existing_key(self.obs_index, ["hour"])
        self.k_temp = existing_key(
            self.obs_index,
            [
                "outdoor_dry_bulb_temperature_predicted_6h",
                "outdoor_dry_bulb_temperature_predicted_1",
                "outdoor_dry_bulb_temperature",
            ],
        )
        self.k_humidity = existing_key(
            self.obs_index,
            [
                "outdoor_relative_humidity_predicted_6h",
                "outdoor_relative_humidity_predicted_1",
                "outdoor_relative_humidity",
            ],
        )
        self.k_price = existing_key(
            self.obs_index,
            ["electricity_pricing_predicted_6h", "electricity_pricing_predicted_1", "electricity_pricing"],
        )
        self.k_occ = existing_key(self.obs_index, ["non_shiftable_load"])
        self.k_soc = existing_key(self.obs_index, ["electrical_storage_soc"])
        self.k_solar_diff = existing_key(
            self.obs_index,
            ["diffuse_solar_irradiance_predicted_6h", "diffuse_solar_irradiance_predicted_1", "diffuse_solar_irradiance"],
        )
        self.k_solar_dir = existing_key(
            self.obs_index,
            ["direct_solar_irradiance_predicted_6h", "direct_solar_irradiance_predicted_1", "direct_solar_irradiance"],
        )

        required = [self.k_hour, self.k_temp, self.k_humidity, self.k_price, self.k_occ, self.k_soc]
        if any(k is None for k in required):
            raise RuntimeError("Missing required observation keys for state encoding.")

        self.i_hour = self.obs_index[self.k_hour]
        self.i_temp = self.obs_index[self.k_temp]
        self.i_humidity = self.obs_index[self.k_humidity]
        self.i_price = self.obs_index[self.k_price]
        self.i_occ = self.obs_index[self.k_occ]
        self.i_soc = self.obs_index[self.k_soc]
        self.i_solar_diff = self.obs_index[self.k_solar_diff] if self.k_solar_diff is not None else None
        self.i_solar_dir = self.obs_index[self.k_solar_dir] if self.k_solar_dir is not None else None

    @staticmethod
    def _q(value: float, decimals: int) -> float:
        if not np.isfinite(value):
            return 0.0
        return float(np.round(value, decimals=decimals))

    @staticmethod
    def _bin(value: float, bins: np.ndarray) -> int:
        if not np.isfinite(value):
            return 0
        return int(np.digitize(value, bins, right=False))

    def _encode_from_obs(self, obs: List[float]) -> Tuple[int, ...]:
        hour = int(np.clip(int(round(float(obs[self.i_hour]))) - 1, 0, 23))
        temp = float(obs[self.i_temp])
        humidity = float(obs[self.i_humidity])
        price = float(obs[self.i_price])
        occ = float(obs[self.i_occ])
        soc = float(obs[self.i_soc])
        diffuse = float(obs[self.i_solar_diff]) if self.i_solar_diff is not None else 0.0
        direct = float(obs[self.i_solar_dir]) if self.i_solar_dir is not None else 0.0
        solar = diffuse + direct

        return (
            hour,
            self._bin(temp, self.TEMP_BINS),
            self._bin(humidity, self.HUMIDITY_BINS),
            self._bin(solar, self.SOLAR_BINS),
            self._bin(price, self.PRICE_BINS),
            self._bin(occ, self.LOAD_BINS),
            self._bin(soc, self.SOC_BINS),
        )

    def encode(self, obs: List[float], building_index: int) -> Tuple[int, ...]:
        # Optional API in some CityLearn variants.
        if hasattr(self.env, "get_state"):
            try:
                b_name = self.env.buildings[building_index].name
                s = self.env.get_state(b_name)
                if isinstance(s, dict):
                    hour = int(np.clip(int(round(float(s.get("hour", 1.0)))) - 1, 0, 23))
                    temp = float(s.get("temp_pred", s.get("outdoor_temperature", 0.0)))
                    humidity = float(s.get("humidity_pred", 0.0))
                    solar = float(s.get("solar_pred", 0.0))
                    price = float(s.get("price_pred", s.get("price", 0.0)))
                    occ = float(s.get("occupancy", s.get("non_shiftable_load", 0.0)))
                    soc = float(s.get("soc", s.get("electrical_storage_soc", 0.0)))
                    return (
                        hour,
                        self._bin(temp, self.TEMP_BINS),
                        self._bin(humidity, self.HUMIDITY_BINS),
                        self._bin(solar, self.SOLAR_BINS),
                        self._bin(price, self.PRICE_BINS),
                        self._bin(occ, self.LOAD_BINS),
                        self._bin(soc, self.SOC_BINS),
                    )
            except Exception:
                pass

        # Fallback that works in this installed version
        return self._encode_from_obs(obs)


# ============================================================================
# AGENT
# ============================================================================


class TabularQAgent:
    def __init__(self, n_buildings: int, n_actions: int):
        self.n_buildings = n_buildings
        self.n_actions = n_actions
        self.epsilon = EPSILON_START
        self.q_tables: List[Dict[Tuple[int, ...], np.ndarray]] = [dict() for _ in range(n_buildings)]

    def _q(self, b_idx: int, state: Tuple[int, ...]) -> np.ndarray:
        q = self.q_tables[b_idx].get(state)
        if q is None:
            q = np.zeros((self.n_actions,), dtype=np.float32)
            self.q_tables[b_idx][state] = q
        return q

    def act(self, states: List[Tuple[int, ...]], training: bool) -> List[int]:
        out: List[int] = []
        for b_idx, state in enumerate(states):
            q = self._q(b_idx, state)
            if training and np.random.rand() < self.epsilon:
                out.append(int(np.random.randint(0, self.n_actions)))
            else:
                out.append(int(np.argmax(q)))
        return out

    def update(
        self,
        states: List[Tuple[int, ...]],
        action_ids: List[int],
        rewards: np.ndarray,
        next_states: List[Tuple[int, ...]],
        done: bool,
    ) -> None:
        for b_idx in range(self.n_buildings):
            q = self._q(b_idx, states[b_idx])
            q_next = self._q(b_idx, next_states[b_idx])
            target = float(rewards[b_idx]) + (0.0 if done else GAMMA * float(np.max(q_next)))
            a = action_ids[b_idx]
            q[a] = (1.0 - ALPHA) * q[a] + ALPHA * target

    def decay_epsilon(self) -> None:
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def epsilon_for_episode(self, episode_idx: int, total_episodes: int) -> float:
        if total_episodes <= 1:
            return EPSILON_END
        progress = float(episode_idx) / float(total_episodes - 1)
        decay = np.exp(-5.0 * progress)
        return float(EPSILON_END + (EPSILON_START - EPSILON_END) * decay)


# ============================================================================
# TRAIN / EVAL
# ============================================================================


@dataclass
class EvalMetrics:
    avg_energy_import: float
    episode_energies: List[float]
    avg_cost: float
    episode_costs: List[float]


def run_training_episode(env: CityLearnEnv, agent: TabularQAgent, encoder: StateEncoder) -> Tuple[float, float, float]:
    obs_list = env.reset()
    states = [encoder.encode(obs, i) for i, obs in enumerate(obs_list)]

    ep_reward = 0.0
    ep_energy = 0.0
    ep_cost = 0.0

    for _ in range(EPISODE_TIME_STEPS):
        action_ids = agent.act(states, training=True)
        actions = build_actions_from_ids(env, action_ids)
        next_obs_list, reward_raw, done, _ = env.step(actions)

        rewards = reward_to_building_array(reward_raw, len(env.buildings))
        next_states = [encoder.encode(obs, i) for i, obs in enumerate(next_obs_list)]
        agent.update(states, action_ids, rewards, next_states, done)

        ep_reward += float(np.sum(rewards))
        ep_energy += get_last_district_energy_import(env)
        ep_cost += get_last_district_cost(env)

        states = next_states
        if done:
            break

    agent.decay_epsilon()
    return ep_reward, ep_energy, ep_cost


def evaluate_fixed_policy(env: CityLearnEnv, action_value: float) -> EvalMetrics:
    episode_energies: List[float] = []
    episode_costs: List[float] = []

    for _ in range(EVAL_EPISODES):
        env.reset()
        ep_energy = 0.0
        ep_cost = 0.0

        for _ in range(EPISODE_TIME_STEPS):
            actions = build_constant_actions(env, action_value)
            _, _, done, _ = env.step(actions)
            ep_energy += get_last_district_energy_import(env)
            ep_cost += get_last_district_cost(env)
            if done:
                break

        episode_energies.append(ep_energy)
        episode_costs.append(ep_cost)

    return EvalMetrics(
        avg_energy_import=float(np.mean(episode_energies)),
        episode_energies=episode_energies,
        avg_cost=float(np.mean(episode_costs)),
        episode_costs=episode_costs,
    )


def evaluate_q_policy(env: CityLearnEnv, agent: TabularQAgent, encoder: StateEncoder) -> EvalMetrics:
    episode_energies: List[float] = []
    episode_costs: List[float] = []

    for _ in range(EVAL_EPISODES):
        obs_list = env.reset()
        states = [encoder.encode(obs, i) for i, obs in enumerate(obs_list)]

        ep_energy = 0.0
        ep_cost = 0.0

        for _ in range(EPISODE_TIME_STEPS):
            action_ids = agent.act(states, training=False)
            actions = build_actions_from_ids(env, action_ids)
            next_obs_list, _, done, _ = env.step(actions)

            ep_energy += get_last_district_energy_import(env)
            ep_cost += get_last_district_cost(env)

            states = [encoder.encode(obs, i) for i, obs in enumerate(next_obs_list)]
            if done:
                break

        episode_energies.append(ep_energy)
        episode_costs.append(ep_cost)

    return EvalMetrics(
        avg_energy_import=float(np.mean(episode_energies)),
        episode_energies=episode_energies,
        avg_cost=float(np.mean(episode_costs)),
        episode_costs=episode_costs,
    )


def stabilization_episode(
    rewards: List[float],
    costs: Optional[List[float]] = None,
    window: int = STABILITY_WINDOW,
    tol: float = STABILITY_TOL,
    patience: int = STABILITY_PATIENCE,
) -> Optional[int]:
    """Return first episode where moving averages are stable for `patience` updates.

    If `costs` is provided, both reward and cost must be stable.
    """

    if len(rewards) < max(2 * window, window + patience + 1):
        return None

    r_ma = moving_average(rewards, window)
    r_prev = r_ma[:-1]
    r_cur = r_ma[1:]
    r_rel = np.abs(r_cur - r_prev) / np.maximum(1e-6, np.abs(r_prev))

    if costs is None:
        stable_flags = r_rel < tol
    else:
        if len(costs) != len(rewards):
            return None
        c_ma = moving_average(costs, window)
        c_prev = c_ma[:-1]
        c_cur = c_ma[1:]
        c_rel = np.abs(c_cur - c_prev) / np.maximum(1e-6, np.abs(c_prev))
        k = min(len(r_rel), len(c_rel))
        stable_flags = (r_rel[:k] < tol) & (c_rel[:k] < tol)

    streak = 0
    for i, ok in enumerate(stable_flags):
        streak = streak + 1 if bool(ok) else 0
        if streak >= patience:
            # i je index zmeny moving-average; +1 mapuje na epizodu, +window kompenzuje MA.
            return int(i + window + 1)

    return None


def plot_training_curves(train_rewards: List[float], train_energy: List[float], train_cost: List[float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(train_rewards, alpha=0.45, label="reward")
    axes[0].plot(moving_average(train_rewards, 20), linewidth=2, label="ma(20)")
    axes[0].set_title("Training reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(train_energy, alpha=0.45, label="energy")
    axes[1].plot(moving_average(train_energy, 20), linewidth=2, label="ma(20)")
    axes[1].set_title("Training energy import")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Energy [kWh]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(train_cost, alpha=0.45, label="cost")
    axes[2].plot(moving_average(train_cost, 20), linewidth=2, label="ma(20)")
    axes[2].set_title("Training cost")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Cost [$]")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("q_learning0904_02_training.png", dpi=130, bbox_inches="tight")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("Q_learning0904_02 - CityLearn Q-learning (energy + real cost)")
    print("=" * 80)
    print(f"citylearn version: {citylearn.__version__}")
    print(f"schema: {resolve_schema(DATASET_NAME)}")
    print(f"episode_time_steps: {EPISODE_TIME_STEPS}")
    print(f"train_episodes: {TRAIN_EPISODES}")
    print(f"eval_episodes: {EVAL_EPISODES}")
    print(f"fast_mode: {FAST_MODE}")

    print("\n[1] Init env / encoder / agent")
    train_env = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    eval_env = make_env(random_seed=RANDOM_SEED, random_episode_split=False)

    print("    collecting samples for data-driven bins...")
    state_samples = collect_state_samples(train_env, max_episodes=12)
    encoder = StateEncoder(train_env, samples=state_samples)
    agent = TabularQAgent(n_buildings=len(train_env.buildings), n_actions=len(ACTION_LEVELS))

    print("\n[2] Training Q-learning")
    train_rewards: List[float] = []
    train_energy: List[float] = []
    train_cost: List[float] = []

    actual_train_episodes = 0
    stable_ep: Optional[int] = None

    for _ep in tqdm(range(TRAIN_EPISODES), desc="Training Q-learning"):
        agent.epsilon = agent.epsilon_for_episode(_ep, TRAIN_EPISODES)
        ep_reward, ep_energy, ep_cost = run_training_episode(train_env, agent, encoder)
        train_rewards.append(ep_reward)
        train_energy.append(ep_energy)
        train_cost.append(ep_cost)
        actual_train_episodes = _ep + 1

        if actual_train_episodes >= STABILITY_RELAXED_MIN_EPISODES and actual_train_episodes % STABILITY_CHECK_EVERY == 0:
            stable_ep = stabilization_episode(train_rewards, train_cost)
            if stable_ep is not None and actual_train_episodes >= stable_ep + STABILITY_PATIENCE:
                print(f"    early stop at episode: {actual_train_episodes} (stable since ~{stable_ep})")
                break

    print(f"    stabilization episode (heuristic): {stable_ep}")
    print(f"    trained episodes: {actual_train_episodes}")
    print(f"    last-10 mean reward: {float(np.mean(train_rewards[-10:])):.4f}")

    print("\n[3] Evaluate fixed baseline")
    baseline = evaluate_fixed_policy(eval_env, FIXED_BASELINE_ACTION)
    print(f"    baseline avg energy import: {baseline.avg_energy_import:.4f}")
    print(f"    baseline avg cost:          {baseline.avg_cost:.4f}")

    print("\n[4] Evaluate Q-learning")
    q_res = evaluate_q_policy(eval_env, agent, encoder)
    print(f"    q-learning avg energy import: {q_res.avg_energy_import:.4f}")
    print(f"    q-learning avg cost:          {q_res.avg_cost:.4f}")

    energy_saving = baseline.avg_energy_import - q_res.avg_energy_import
    energy_saving_pct = 100.0 * energy_saving / max(1e-6, baseline.avg_energy_import)
    cost_saving = baseline.avg_cost - q_res.avg_cost
    cost_saving_pct = 100.0 * cost_saving / max(1e-6, baseline.avg_cost)

    print("\n[5] Savings vs baseline")
    print("-" * 80)
    print(f"energy saving: {energy_saving:.4f} ({energy_saving_pct:.2f}%)")
    print(f"cost saving:   {cost_saving:.4f} ({cost_saving_pct:.2f}%)")
    print("-" * 80)

    summary = pd.DataFrame(
        [
            {
                "policy": "fixed_baseline",
                "avg_energy_import": baseline.avg_energy_import,
                "avg_cost": baseline.avg_cost,
                "energy_saving_vs_baseline": 0.0,
                "energy_saving_vs_baseline_pct": 0.0,
                "cost_saving_vs_baseline": 0.0,
                "cost_saving_vs_baseline_pct": 0.0,
                "stabilization_episode": np.nan,
            },
            {
                "policy": "q_learning",
                "avg_energy_import": q_res.avg_energy_import,
                "avg_cost": q_res.avg_cost,
                "energy_saving_vs_baseline": energy_saving,
                "energy_saving_vs_baseline_pct": energy_saving_pct,
                "cost_saving_vs_baseline": cost_saving,
                "cost_saving_vs_baseline_pct": cost_saving_pct,
                "stabilization_episode": stable_ep if stable_ep is not None else np.nan,
            },
        ]
    )

    summary.to_csv("q_learning0904_02_results.csv", index=False)
    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    plot_training_curves(train_rewards, train_energy, train_cost)
    print("\nSaved: q_learning0904_02_results.csv")
    print("Saved: q_learning0904_02_training.png")


if __name__ == "__main__":
    main()
