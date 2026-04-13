#!/usr/bin/env python
"""
Q_learning1204_15.py
====================
Tabulárny Q-learning pre CityLearn Challenge 2022 Phase 1.

Prečo V10 bol najlepší a čo robíme inak
-----------------------------------------
V10 stav: (hodina=24, solar_bin=6, price_bin=6, soc_bin=6) = 5 184 stavov
V10 výsledok: +8.2 % energia, +30.7 % náklady

V11–V14: každá verzia pridala dimenzie → väčší priestor → menej vzoriek/stav.

Kľúčové zistenie: v datasete 2022 je cena BING len 5 unikátnych hodnôt:
    [0.21, 0.22, 0.40, 0.50, 0.54]
    - 75 % dát je na 0.21–0.22  (lacná)
    - 10 % je 0.50–0.54          (drahá — práve toto nás zaujíma!)
Správna stratégia: nabiť batériu počas lacných hodín (noc, poludnie + solar),
    vybiť počas drahých hodín večer (peak 17–20h).

V15 opravy
----------
1. SPÄŤ NA V10 STAV + pridaj explicitný peak_price príznak
   stav = (hodina, solar_bin, peak_price, soc_bin)
     - hodina    : 24         (presná hodina — kritická pre peak scheduling)
     - solar_bin : 4           (noc/slabé/stredné/silné - daytime quantile bins)
     - peak_price: 2           (0=lacné ≤ 0.22, 1=drahé > 0.22)
     - soc_bin   : 6           (pevných 5 hraníc)
   = 24 × 4 × 2 × 6 = 1 152 stavov/budovu

   Prečo peak_price 2 triedy: 75 % lacné, 25 % drahé. Jednoduchý binárny
   signál stačí — agent sa naučí "vybíjaj keď peak=1".

2. SOLAR: irradiance SUM (diffuse + direct W/m²) namiesto solar_generation kWh
   Dôvod: V10 používal irradianciu a fungoval. solar_generation (kWh) je
   integrál cez 1h, má iné rozdelenie a silnejší ceiling = ťažšie kvantilovať.

3. FIXNÝ α = 0.25  (späť k V10)
   Pokles α v V12–V14 nepomohol; s malým priestorom (1 152 stavov) stačí
   konštantný α lebo konvergencia je rýchla.

4. EVAL_EPISODES = 30  (späť k V10)
   52 epizód pri random_episode_split=False → deterministic order → agentov
   výkon je konzistentne meraný aj s kratším eval.

Výsledný stavový priestor
--------------------------
    24 hodín × 4 solar_bin × 2 peak_price × 6 soc_bin = 1 152 stavov / budovu

    Pokrytie: 300 ep × 168 krokov × 5 budov = 252 000 prechodov
              / (1 152 × 3 akcií) ≈ 73 návštev / stav·akcia  ✓
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


# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURÁCIA
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_NAME  = "citylearn_challenge_2022_phase_1"
RANDOM_SEED   = 42
OUTPUT_PREFIX = "q_learning1204_15"

FAST_MODE = True

if FAST_MODE:
    EPISODE_TIME_STEPS = 24 * 7   # 1 týždeň / epizóda
    TRAIN_EPISODES     = 450
    EVAL_EPISODES      = 52
else:
    EPISODE_TIME_STEPS = 24 * 14
    TRAIN_EPISODES     = 800
    EVAL_EPISODES      = 52

WARMUP_EPISODES = 8

ALPHA         = 0.25
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05

# 3 akcie — identické s V10
ACTION_LEVELS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
N_ACTIONS     = len(ACTION_LEVELS)

FIXED_BASELINE_ACTION = 0.0

SOC_EMPTY_THRESH = 0.05
SOC_FULL_THRESH  = 0.95

# Pevné SOC biny (fyzikálne hranice, rovnaké ako V10)
SOC_BINS = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32)  # → 6 intervalov

# Záložné solar biny (W/m² irradiance — rovnaká škála ako V10)
SOLAR_FALLBACK = np.array([50.0, 200.0, 500.0], dtype=np.float32)  # → 4 intervaly

# Prah pre peak_price (z datasetu: lacné ≤ 0.22, drahé > 0.22)
PEAK_PRICE_THRESHOLD = 0.22   # → 0 = lacné, 1 = drahé


# ═══════════════════════════════════════════════════════════════════════════════
# POMOCNÉ FUNKCIE  (takmer identické s V10)
# ═══════════════════════════════════════════════════════════════════════════════

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
    for k in candidates:
        if k in idx:
            return k
    return None


def build_constant_actions(env: CityLearnEnv, value: float) -> List[List[float]]:
    return [[float(value)] * int(sp.shape[0]) for sp in env.action_space]


def build_actions_from_ids(env: CityLearnEnv, ids: List[int]) -> List[List[float]]:
    return [[float(ACTION_LEVELS[ids[i]])] * int(sp.shape[0])
            for i, sp in enumerate(env.action_space)]


def moving_average(values: List[float], window: int = 20) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if window <= 1 or arr.size == 0:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")


def reward_to_array(reward: object, n: int) -> np.ndarray:
    arr = np.asarray(reward, dtype=np.float32).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full((n,), float(arr[0]) / max(1, n), dtype=np.float32)
    out = np.zeros((n,), dtype=np.float32)
    out[:min(n, arr.size)] = arr[:min(n, arr.size)]
    return out


def grid_import(env: CityLearnEnv) -> float:
    if not env.net_electricity_consumption:
        return 0.0
    return max(0.0, float(env.net_electricity_consumption[-1]))


def district_cost(env: CityLearnEnv) -> float:
    if hasattr(env, "net_electricity_consumption_cost") and env.net_electricity_consumption_cost:
        return float(env.net_electricity_consumption_cost[-1])
    return 0.0


def safe_pct(base: float, ql: float, cap: float = 500.0) -> Optional[float]:
    if base <= 1e-6:
        return None
    return float(np.clip(100.0 * (base - ql) / base, -cap, cap))


def dig(val: float, bins: np.ndarray) -> int:
    return int(np.digitize(val if np.isfinite(val) else 0.0, bins, right=False))


def quantile_bins(values: List[float], n: int, fallback: np.ndarray) -> np.ndarray:
    """n-1 hraníc → n intervalov (rovnaká logika ako V10)."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size < max(20, n * 3):
        return fallback
    qs   = np.linspace(0.0, 1.0, n + 1)[1:-1]
    bins = np.unique(np.quantile(arr, qs))
    return bins.astype(np.float32) if bins.size > 0 else fallback


# ═══════════════════════════════════════════════════════════════════════════════
# BINY STAVOVÉHO PRIESTORU
# ═══════════════════════════════════════════════════════════════════════════════

class BinSet:
    """
    Stavový priestor: (hodina=24, solar_bin=4, peak_price=2, soc_bin=6)

    solar_bin: quantile biny z IRRADIANCE sumy (W/m²) — rovnako ako V10
    peak_price: binárny (0=lacné, 1=drahé), prah = PEAK_PRICE_THRESHOLD
    soc_bin: pevné fyzikálne hranice SOC_BINS
    """

    def __init__(self, env: CityLearnEnv) -> None:
        idx = get_obs_index(env)

        self.k_hour       = first_key(idx, ["hour"])
        # Irradiance (W/m²) — suma difuznej + priamej, ako v V10
        self.k_solar_diff = first_key(idx, [
            "diffuse_solar_irradiance_predicted_6h",
            "diffuse_solar_irradiance_predicted_1",
            "diffuse_solar_irradiance",
        ])
        self.k_solar_dir  = first_key(idx, [
            "direct_solar_irradiance_predicted_6h",
            "direct_solar_irradiance_predicted_1",
            "direct_solar_irradiance",
        ])
        self.k_price = first_key(idx, [
            "electricity_pricing_predicted_6h",
            "electricity_pricing_predicted_1",
            "electricity_pricing",
        ])
        self.k_soc   = first_key(idx, ["electrical_storage_soc"])

        self.i_hour       = idx[self.k_hour]       if self.k_hour       else None
        self.i_solar_diff = idx[self.k_solar_diff] if self.k_solar_diff else None
        self.i_solar_dir  = idx[self.k_solar_dir]  if self.k_solar_dir  else None
        self.i_price      = idx[self.k_price]      if self.k_price      else None
        self.i_soc        = idx[self.k_soc]        if self.k_soc        else None

        self.solar_bins: np.ndarray = SOLAR_FALLBACK
        self.soc_bins:   np.ndarray = SOC_BINS

        self._solar_samples: List[float] = []
        self._price_samples: List[float] = []

    def fit(self, env: CityLearnEnv) -> None:
        """Warmup: zbiera irradiance vzorky → quantile solar_bins."""
        solar_v: List[float] = []
        price_v: List[float] = []

        for ep_i in range(WARMUP_EPISODES):
            act_val = 1.0 if ep_i % 2 == 0 else -1.0
            env.reset()
            for _ in range(EPISODE_TIME_STEPS):
                next_obs_list, _, done, _ = env.step(
                    build_constant_actions(env, act_val)
                )
                for obs in next_obs_list:
                    s = 0.0
                    if self.i_solar_diff is not None:
                        s += float(obs[self.i_solar_diff])
                    if self.i_solar_dir is not None:
                        s += float(obs[self.i_solar_dir])
                    solar_v.append(s)
                    if self.i_price is not None:
                        price_v.append(float(obs[self.i_price]))
                if done:
                    break

        self._solar_samples = solar_v
        self._price_samples = price_v
        # 3 kvantilové hrany → 4 intervaly (rovnaké ako V10 malo 5 hraníc → 6, tu 4 stačí)
        self.solar_bins = quantile_bins(solar_v, 4, SOLAR_FALLBACK)

    def state_space_size(self) -> int:
        return 24 * (len(self.solar_bins) + 1) * 2 * (len(self.soc_bins) + 1)

    def print_summary(self) -> None:
        def stats(v: List[float]) -> str:
            if not v:
                return "—"
            a = np.asarray(v, dtype=np.float32)
            return f"min={a.min():.3g}  μ={a.mean():.3g}  max={a.max():.3g}  (n={a.size})"

        def bstr(b: np.ndarray) -> str:
            return "[" + ", ".join(f"{x:.3g}" for x in b) + "]"

        solar_key = (
            f"{self.k_solar_diff or '(nenájd.)'} + {self.k_solar_dir or '(nenájd.)'}"
            f"  [W/m² irradiance suma]"
        )
        n_sol = len(self.solar_bins) + 1
        n_soc = len(self.soc_bins) + 1

        sep = "─" * 100
        print(f"\n    {sep}")
        print(f"    {'Dimenzia':<14}  {'Použitý kľúč / popis':<50}  Štatistiky warmup vzoriek")
        print(f"    {sep}")
        print(f"    {'hodina':<14}  {str(self.k_hour or '(nenájdený)'):<50}  0–23 (celá hodina, 24 tried)")
        print(f"    {'solar (W/m²)':<14}  {solar_key:<50}  {stats(self._solar_samples)}")
        print(f"    {'':14}    → biny: {bstr(self.solar_bins)}  ({n_sol} intervalov)")
        print(f"    {'peak_price':<14}  {str(self.k_price or '(nenájdený)'):<50}  {stats(self._price_samples)}")
        print(f"    {'':14}    → prah: {PEAK_PRICE_THRESHOLD}  (0=lacné ≤{PEAK_PRICE_THRESHOLD}, 1=drahé >{PEAK_PRICE_THRESHOLD})")
        print(f"    {'bat. SOC':<14}  {str(self.k_soc or '(nenájdený)'):<50}  PEVNÉ → biny: {bstr(self.soc_bins)}")
        print(f"    {sep}")
        n_st = self.state_space_size()
        exp  = TRAIN_EPISODES * EPISODE_TIME_STEPS * 5 // (n_st * N_ACTIONS)
        ok   = "✓ dostatočné" if exp >= 20 else "⚠ malo"
        print(f"    Stavový priestor: 24 × {n_sol} × 2 × {n_soc} = {n_st:,} stavov/budovu")
        print(f"    Pokrytie: ~{exp} návštev / stav·akcia  ({ok})\n")


# ═══════════════════════════════════════════════════════════════════════════════
# KÓDOVANIE STAVU
# ═══════════════════════════════════════════════════════════════════════════════

# (hodina, solar_bin, peak_price, soc_bin)
StateT = Tuple[int, int, int, int]


def encode_state(obs: List[float], bins: BinSet, soc: float) -> StateT:
    hour = int(np.clip(
        (int(round(float(obs[bins.i_hour]))) - 1) if bins.i_hour is not None else 0,
        0, 23
    ))

    irr = 0.0
    if bins.i_solar_diff is not None:
        irr += float(obs[bins.i_solar_diff])
    if bins.i_solar_dir is not None:
        irr += float(obs[bins.i_solar_dir])

    price = float(obs[bins.i_price]) if bins.i_price is not None else 0.21
    peak  = 1 if price > PEAK_PRICE_THRESHOLD else 0

    return (
        hour,
        dig(irr,  bins.solar_bins),
        peak,
        dig(soc,  bins.soc_bins),
    )


def get_socs(env: CityLearnEnv, obs_list: List[List[float]],
             bins: BinSet) -> List[float]:
    return [
        float(obs[bins.i_soc]) if (bins.i_soc is not None and bins.i_soc < len(obs)) else 0.0
        for obs in obs_list
    ]


def valid_actions(soc: float) -> List[int]:
    if soc <= SOC_EMPTY_THRESH:
        return [1, 2]   # hold alebo nabíjaj
    if soc >= SOC_FULL_THRESH:
        return [0, 1]   # vybíjaj alebo drž
    return list(range(N_ACTIONS))


# ═══════════════════════════════════════════════════════════════════════════════
# Q-AGENT  (rovnaká štruktúra ako V10)
# ═══════════════════════════════════════════════════════════════════════════════

class TabularQAgent:
    def __init__(self, n_buildings: int) -> None:
        self.n_buildings = n_buildings
        self.epsilon     = EPSILON_START
        self.q_tables: List[Dict[StateT, np.ndarray]] = [
            {} for _ in range(n_buildings)
        ]

    def _q(self, b: int, state: StateT) -> np.ndarray:
        q = self.q_tables[b].get(state)
        if q is None:
            q = np.zeros(N_ACTIONS, dtype=np.float32)
            self.q_tables[b][state] = q
        return q

    def act(self, states: List[StateT], socs: List[float],
            training: bool) -> List[int]:
        out: List[int] = []
        for b, (state, soc) in enumerate(zip(states, socs)):
            allowed = valid_actions(soc)
            if training and np.random.rand() < self.epsilon:
                out.append(int(np.random.choice(allowed)))
            else:
                q = self._q(b, state)
                out.append(int(max(allowed, key=lambda a: q[a])))
        return out

    def update(self, states: List[StateT], acts: List[int],
               rewards: np.ndarray, next_states: List[StateT],
               next_socs: List[float], done: bool) -> None:
        for b in range(self.n_buildings):
            q       = self._q(b, states[b])
            q_nxt   = self._q(b, next_states[b])
            allowed = valid_actions(next_socs[b])
            best    = float(max(q_nxt[a] for a in allowed))
            target  = float(rewards[b]) + (0.0 if done else GAMMA * best)
            a       = acts[b]
            q[a]    = (1.0 - ALPHA) * q[a] + ALPHA * target

    def set_epsilon(self, episode: int, total: int) -> None:
        progress     = float(episode) / max(1, total - 1)
        self.epsilon = float(EPSILON_END + (EPSILON_START - EPSILON_END)
                             * np.exp(-5.0 * progress))

    @property
    def total_states(self) -> int:
        return sum(len(d) for d in self.q_tables)


# ═══════════════════════════════════════════════════════════════════════════════
# TRÉNING / EVALUÁCIA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalMetrics:
    avg_energy_import: float
    avg_cost:          float
    episode_energies:  List[float]
    episode_costs:     List[float]


def run_training_episode(
    env: CityLearnEnv, agent: TabularQAgent, bins: BinSet
) -> Tuple[float, float, float]:
    obs_list = env.reset()
    socs     = get_socs(env, obs_list, bins)
    states   = [encode_state(obs, bins, soc) for obs, soc in zip(obs_list, socs)]

    ep_reward = ep_energy = ep_cost = 0.0

    for _ in range(EPISODE_TIME_STEPS):
        acts = agent.act(states, socs, training=True)
        next_obs_list, raw_rew, done, _ = env.step(build_actions_from_ids(env, acts))
        rewards     = reward_to_array(raw_rew, len(env.buildings))
        next_socs   = get_socs(env, next_obs_list, bins)
        next_states = [encode_state(obs, bins, soc)
                       for obs, soc in zip(next_obs_list, next_socs)]

        agent.update(states, acts, rewards, next_states, next_socs, done)

        ep_reward += float(np.sum(rewards))
        ep_energy += grid_import(env)
        ep_cost   += district_cost(env)

        obs_list = next_obs_list
        states, socs = next_states, next_socs
        if done:
            break

    return ep_reward, ep_energy, ep_cost


def evaluate_policy(
    env: CityLearnEnv,
    *,
    agent:       Optional[TabularQAgent] = None,
    bins:        Optional[BinSet]        = None,
    fixed_value: Optional[float]         = None,
) -> EvalMetrics:
    assert (agent is not None) != (fixed_value is not None)

    ep_energies: List[float] = []
    ep_costs:    List[float] = []

    for _ in range(EVAL_EPISODES):
        obs_list = env.reset()
        if agent is not None:
            socs   = get_socs(env, obs_list, bins)
            states = [encode_state(obs, bins, soc)
                      for obs, soc in zip(obs_list, socs)]

        ep_energy = ep_cost = 0.0
        for _ in range(EPISODE_TIME_STEPS):
            if agent is not None:
                acts    = agent.act(states, socs, training=False)
                actions = build_actions_from_ids(env, acts)
            else:
                actions = build_constant_actions(env, fixed_value)

            next_obs_list, _, done, _ = env.step(actions)
            ep_energy += grid_import(env)
            ep_cost   += district_cost(env)

            if agent is not None:
                socs   = get_socs(env, next_obs_list, bins)
                states = [encode_state(obs, bins, soc)
                          for obs, soc in zip(next_obs_list, socs)]
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


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training(rewards: List[float], energy: List[float],
                  cost: List[float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Q_learning1204_15 – Tréning\n"
        "stav: (hodina, solar_W/m², peak_price, SOC)  |  3 akcie  |  α=0.25 konštantný",
        fontsize=11, fontweight="bold",
    )
    specs = [
        (rewards, "Kumulatívna odmena",      "Odmena",       "steelblue",  axes[0]),
        (energy,  "Odber energie (tréning)", "Energia [kWh]","teal",       axes[1]),
        (cost,    "Náklady (tréning)",       "Náklady [$]",  "darkorange", axes[2]),
    ]
    for data, title, ylabel, color, ax in specs:
        ax.plot(data, alpha=0.35, color=color, linewidth=0.8)
        ax.plot(moving_average(data, 20), linewidth=2, color="navy", label="MA(20)")
        ax.set_title(title)
        ax.set_xlabel("Epizóda")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_training.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_eval_comparison(baseline: EvalMetrics, q_res: EvalMetrics) -> None:
    n   = min(len(baseline.episode_energies), len(q_res.episode_energies))
    eps = np.arange(1, n + 1)
    be  = np.array(baseline.episode_energies[:n])
    qe  = np.array(q_res.episode_energies[:n])
    bc  = np.array(baseline.episode_costs[:n])
    qc  = np.array(q_res.episode_costs[:n])

    e_pct = np.array([
        safe_pct(b, q) if safe_pct(b, q) is not None else float("nan")
        for b, q in zip(be, qe)
    ])
    c_pct = np.array([
        safe_pct(b, q) if safe_pct(b, q) is not None else float("nan")
        for b, q in zip(bc, qc)
    ])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Porovnanie: Fixná stratégia  vs  Q-learning 1204_15\n"
        "stav: (hodina, solar_W/m², peak_price, SOC)  |  3 akcie",
        fontsize=11, fontweight="bold",
    )
    w  = 0.38
    x  = np.arange(n)
    tk = x[::max(1, n // 12)]

    for ax, b_data, q_data, title, ylabel in [
        (axes[0, 0], be, qe, "Odber energie [kWh]", "kWh"),
        (axes[0, 1], bc, qc, "Náklady [$]",          "$"),
    ]:
        ax.bar(x - w / 2, b_data, w, label="Fixná stratégia", color="tomato",    alpha=0.85)
        ax.bar(x + w / 2, q_data, w, label="Q-learning",      color="steelblue", alpha=0.85)
        ax.set_xticks(tk); ax.set_xticklabels(eps[tk], fontsize=8)
        ax.set_xlabel("Eval epizóda (týždeň)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    for ax, pct, title in [
        (axes[1, 0], e_pct, "Úspora energie [%]  (zelená = QL lepšie)"),
        (axes[1, 1], c_pct, "Úspora nákladov [%]  (N/A kde baseline ≤ 0)"),
    ]:
        valid  = ~np.isnan(pct)
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in np.where(valid, pct, 0)]
        ax.bar(eps[valid], pct[valid],
               color=[colors[i] for i in np.where(valid)[0]], alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        if np.any(valid):
            mean_v = float(np.nanmean(pct[valid]))
            ax.axhline(mean_v, color="navy", linewidth=1.5, linestyle="--",
                       label=f"priemer = {mean_v:.1f}%")
        n_na = int(np.sum(~valid))
        ax.set_title(f"{title}  [{n_na}× N/A]" if n_na > 0 else title)
        ax.set_xlabel("Eval epizóda (týždeň)")
        ax.set_ylabel("%")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_eval_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# HLAVNÁ FUNKCIA
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    set_seed(RANDOM_SEED)
    SEP = "=" * 80

    print(SEP)
    print("  Q_learning1204_15 – V10 jadro + peak_price príznak + irradiance solar")
    print("  Stav: (hodina, solar_W/m², peak_price, SOC)  |  3 akcie  |  α=0.25")
    print(SEP)
    print(f"  citylearn          : {citylearn.__version__}")
    print(f"  dataset            : {DATASET_NAME}")
    print(f"  episode_time_steps : {EPISODE_TIME_STEPS}  ({EPISODE_TIME_STEPS // 24} dní)")
    print(f"  train_episodes     : {TRAIN_EPISODES}")
    print(f"  eval_episodes      : {EVAL_EPISODES}")
    print(f"  α (learning rate)  : {ALPHA}  (konštantný)")
    print(f"  peak_price_thresh  : {PEAK_PRICE_THRESHOLD} $/kWh  "
          f"(5 unikátnych cien: 0.21/0.22/0.40/0.50/0.54)")
    print(f"  akcie              : {ACTION_LEVELS.tolist()}")

    # ── [1] Prostredia ───────────────────────────────────────────────────────
    print(f"\n[1/5] Inicializácia prostredí …")
    train_env   = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    eval_env    = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    n_buildings = len(train_env.buildings)
    print(f"       budov: {n_buildings}")

    # ── [2] Warmup ───────────────────────────────────────────────────────────
    print(f"\n[2/5] Warmup – výpočet solar binov ({WARMUP_EPISODES} ep.) …")
    bins = BinSet(train_env)
    bins.fit(train_env)
    bins.print_summary()

    # ── [3] Baseline ─────────────────────────────────────────────────────────
    print(f"[3/5] Evaluácia fixnej stratégie (akcia = {FIXED_BASELINE_ACTION}) …")
    baseline = evaluate_policy(eval_env, fixed_value=FIXED_BASELINE_ACTION)
    print(f"       avg energy import : {baseline.avg_energy_import:.4f} kWh")
    print(f"       avg cost          : {baseline.avg_cost:.4f} $")

    # ── [4] Tréning ──────────────────────────────────────────────────────────
    print(f"\n[4/5] Trénovanie Q-learningu ({TRAIN_EPISODES} epizód, α={ALPHA}) …")
    agent        = TabularQAgent(n_buildings)
    train_rewards: List[float] = []
    train_energy:  List[float] = []
    train_cost:    List[float] = []

    for ep in tqdm(range(TRAIN_EPISODES), desc="Q-learning tréning"):
        agent.set_epsilon(ep, TRAIN_EPISODES)
        r, e, c = run_training_episode(train_env, agent, bins)
        train_rewards.append(r)
        train_energy.append(e)
        train_cost.append(c)

    print(f"       navštívených stavov        : {agent.total_states}")
    print(f"       posledných 20 ep. – priem. odmena : {np.mean(train_rewards[-20:]):.4f}")

    # ── [5] Evaluácia ────────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluácia Q-agenta (greedy, ε=0, {EVAL_EPISODES} epizód) …")
    q_res = evaluate_policy(eval_env, agent=agent, bins=bins)
    print(f"       avg energy import : {q_res.avg_energy_import:.4f} kWh")
    print(f"       avg cost          : {q_res.avg_cost:.4f} $")

    # ── Per-epizóda tabuľka ──────────────────────────────────────────────────
    n_ep  = min(len(baseline.episode_energies), len(q_res.episode_energies))
    rows: List[dict] = []
    for i in range(n_ep):
        be_i = baseline.episode_energies[i]
        qe_i = q_res.episode_energies[i]
        bc_i = baseline.episode_costs[i]
        qc_i = q_res.episode_costs[i]
        ep   = safe_pct(be_i, qe_i)
        cp   = safe_pct(bc_i, qc_i)
        rows.append({
            "episode":      i + 1,
            "fixed_energy": round(be_i, 3),
            "ql_energy":    round(qe_i, 3),
            "energy_sav_%": round(ep, 2) if ep is not None else "N/A",
            "fixed_cost":   round(bc_i, 4),
            "ql_cost":      round(qc_i, 4),
            "cost_sav_%":   round(cp, 2) if cp is not None else "N/A",
        })

    per_ep_df = pd.DataFrame(rows)
    per_ep_df.to_csv(f"{OUTPUT_PREFIX}_per_episode.csv", index=False)
    print(f"\n  Per-epizóda porovnanie ({n_ep} epizód):")
    print(per_ep_df.to_string(index=False))

    e_vals = [r["energy_sav_%"] for r in rows if isinstance(r["energy_sav_%"], float)]
    c_vals = [r["cost_sav_%"]   for r in rows if isinstance(r["cost_sav_%"],   float)]
    n_na_e = sum(1 for r in rows if r["energy_sav_%"] == "N/A")
    n_na_c = sum(1 for r in rows if r["cost_sav_%"]   == "N/A")

    e_avg = float(np.mean(e_vals)) if e_vals else 0.0
    c_avg = float(np.mean(c_vals)) if c_vals else 0.0

    print(f"\n  {'─' * 70}")
    print(f"  PRIEMER – energia  : {e_avg:+.2f}%  "
          f"({'lepšie' if e_avg > 0 else 'horšie'} ako baseline, N/A: {n_na_e})")
    print(f"  PRIEMER – náklady  : {c_avg:+.2f}%  "
          f"({'lepšie' if c_avg > 0 else 'horšie'} ako baseline, N/A: {n_na_c})")
    print(f"  {'─' * 70}")

    # ── Súhrnná tabuľka ──────────────────────────────────────────────────────
    e_sav = baseline.avg_energy_import - q_res.avg_energy_import
    e_pct = safe_pct(baseline.avg_energy_import, q_res.avg_energy_import)
    c_sav = baseline.avg_cost - q_res.avg_cost
    c_pct = safe_pct(baseline.avg_cost, q_res.avg_cost)

    summary_df = pd.DataFrame([
        {
            "policy":            "fixed_baseline",
            "avg_energy_import": baseline.avg_energy_import,
            "avg_cost":          baseline.avg_cost,
            "energy_sav":        0.0,
            "energy_sav_%":      0.0,
            "cost_sav":          0.0,
            "cost_sav_%":        0.0,
        },
        {
            "policy":            "q_learning_1204_15",
            "avg_energy_import": q_res.avg_energy_import,
            "avg_cost":          q_res.avg_cost,
            "energy_sav":        e_sav,
            "energy_sav_%":      e_pct if e_pct is not None else float("nan"),
            "cost_sav":          c_sav,
            "cost_sav_%":        c_pct if c_pct is not None else float("nan"),
        },
    ])
    summary_df.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)

    print(f"\n  Súhrnné výsledky ({EVAL_EPISODES} eval epizód):")
    print(f"  {'─' * 70}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"  {'─' * 70}")

    # ── Grafy ─────────────────────────────────────────────────────────────────
    plot_training(train_rewards, train_energy, train_cost)
    plot_eval_comparison(baseline, q_res)

    print(f"\n  Uložené súbory:")
    for fn in [
        f"{OUTPUT_PREFIX}_results.csv",
        f"{OUTPUT_PREFIX}_per_episode.csv",
        f"{OUTPUT_PREFIX}_training.png",
        f"{OUTPUT_PREFIX}_eval_comparison.png",
    ]:
        print(f"    {fn}")
    print(SEP)


if __name__ == "__main__":
    main()
