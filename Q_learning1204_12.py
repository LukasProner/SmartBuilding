#!/usr/bin/env python
"""
Q_learning1204_12.py
====================
Tabulárny Q-learning pre CityLearn Challenge 2022 Phase 1.

Prečo bolo viac epizód horšie (analýza problémov z verzie 11)
--------------------------------------------------------------
1. PREUČENIE (overtraining) – α=0.25 konštantné celý tréning.
   Pri epizóde 900 stále veľké aktualizácie prepíšu dobrú politiku
   naučenú pri epizóde 300.
   Riešenie: α lineárne klesá od 0.25 → 0.03.

2. CHÝBAJÚCI SEZÓNNY KONTEXT – stav (hodina, solar, teplota, load, SOC)
   nerozlišuje dostatočne leto (vysoké solar) od zimy (nízke solar).
   Ten istý stav môže mať v lete aj v zime opačnú optimálnu akciu
   → Q-tabuľka spriemeruje → zlé pre obe sezóny.
   Riešenie: explicitná dimenzia SEZÓNA (0=zima,1=jar,2=leto,3=jeseň).

3. ZAVÁDZAJÚCI EVAL pri 30 epizódach – eval bol epizódy 1-30,
   ktoré náhodou pokryli letné týždne. So 100 eval bol viditeľný
   celý rok vrátane zlých zimných týždňov.
   Riešenie: eval_episodes = 52 (= celý rok, bez duplikátov).

4. ZBYTOČNE ZLOŽITÝ SOLAR VSTUP – súčet difúznej+priamej iradiancie
   namiesto priamo dostupného solar_generation (kWh).
   Riešenie: použiť solar_generation priamo.

Nový stavový priestor
---------------------
    (sezóna, hodina, solar_bin, load_bin, soc_bin)

    sezóna  : 4 hodnoty (0=zima, 1=jar, 2=leto, 3=jeseň)
    hodina  : 24 hodnôt (0–23)
    solar   : 5 intervalov z solar_generation [kWh]  → počasie
    load    : 4 intervaly z non_shiftable_load [kWh] → obsadenosť
    SOC     : 6 intervalov (pevné fyzikálne hranice)

    Celkovo: 4 × 24 × 5 × 4 × 6 = 11 520 stavov/budovu
    Pri 1000 ep × 168 kr × 5 budov = 840 000 prechodov
    → priemerne ~73 návštev/stav – dostatočné pokrytie.
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
OUTPUT_PREFIX = "q_learning1204_12"

FAST_MODE = True

if FAST_MODE:
    EPISODE_TIME_STEPS = 24 * 7    # 1 týždeň / epizóda
    TRAIN_EPISODES     = 600
    # 52 epizód = 1 celý rok (8760 h / 168 h), žiadne duplikáty
    EVAL_EPISODES      = 52
else:
    EPISODE_TIME_STEPS = 24 * 14   # 2 týždne / epizóda
    TRAIN_EPISODES     = 1500
    EVAL_EPISODES      = 52

WARMUP_EPISODES = 8

# Klesajúca rýchlosť učenia – FIX #1
ALPHA_START = 0.25    # agresívne učenie na začiatku
ALPHA_END   = 0.03    # jemné opravy ku koncu (zabraňuje preučeniu)

GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05

ACTION_LEVELS         = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
N_ACTIONS             = len(ACTION_LEVELS)
FIXED_BASELINE_ACTION = 0.0

SOC_EMPTY_THRESH = 0.05
SOC_FULL_THRESH  = 0.95

# Pevné SOC biny
SOC_BINS = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32)

# Fallback biny pre solar a load
SOLAR_FALLBACK = np.array([0.2,  0.8,  1.8,  3.5],  dtype=np.float32)  # kWh
LOAD_FALLBACK  = np.array([0.5,  1.5,  3.0],         dtype=np.float32)  # kWh


# ═══════════════════════════════════════════════════════════════════════════════
# POMOCNÉ FUNKCIE
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
    """Percentuálna úspora. None ak baseline ≤ 0 (solárny export)."""
    if base <= 1e-6:
        return None
    return float(np.clip(100.0 * (base - ql) / base, -cap, cap))


def quantile_bins(values: List[float], n: int, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size < max(20, n * 3):
        return fallback
    qs   = np.linspace(0.0, 1.0, n + 1)[1:-1]
    bins = np.unique(np.quantile(arr, qs))
    return bins.astype(np.float32) if bins.size > 0 else fallback


def dig(val: float, bins: np.ndarray) -> int:
    return int(np.digitize(val if np.isfinite(val) else 0.0, bins, right=False))


def month_to_season(month: int) -> int:
    """
    Meteorologické sezóny (0-3):
      0 = zima  (dec=12, jan=1, feb=2)
      1 = jar   (mar=3, apr=4, máj=5)
      2 = leto  (jún=6, júl=7, aug=8)
      3 = jeseň (sep=9, okt=10, nov=11)
    """
    return int(month % 12) // 3


def get_alpha(episode: int, total: int) -> float:
    """Lineárne klesajúca rýchlosť učenia: ALPHA_START → ALPHA_END."""
    progress = float(episode) / max(1, total - 1)
    return ALPHA_END + (ALPHA_START - ALPHA_END) * (1.0 - progress)


# ═══════════════════════════════════════════════════════════════════════════════
# BINY STAVOVÉHO PRIESTORU
# ═══════════════════════════════════════════════════════════════════════════════

class BinSet:
    """
    Diskretizuje stavový priestor: (sezóna, hodina, solar_bin, load_bin, soc_bin).

    solar: solar_generation [kWh] – FIX #4 (priame meranie, nie iradianciová suma)
    load : non_shiftable_load [kWh] – proxy pre obsadenosť budovy
    soc  : vždy pevné físikálne hranice
    season: odvodená z mesačnej pozorovania – FIX #2
    """

    def __init__(self, env: CityLearnEnv) -> None:
        idx = get_obs_index(env)

        self.k_month  = first_key(idx, ["month"])
        self.k_hour   = first_key(idx, ["hour"])

        # FIX #4: solar_generation priamo, záloha = iradianciová suma
        self.k_solar  = first_key(idx, [
            "solar_generation",
            "diffuse_solar_irradiance_predicted_6h",
            "diffuse_solar_irradiance",
        ])
        self.k_solar_dir = first_key(idx, [
            "direct_solar_irradiance_predicted_6h",
            "direct_solar_irradiance",
        ]) if self.k_solar != "solar_generation" else None

        self.k_load   = first_key(idx, [
            "non_shiftable_load_predicted_6h",
            "non_shiftable_load",
        ])
        self.k_soc    = first_key(idx, ["electrical_storage_soc"])

        self.i_month     = idx[self.k_month]   if self.k_month   else None
        self.i_hour      = idx[self.k_hour]    if self.k_hour    else None
        self.i_solar     = idx[self.k_solar]   if self.k_solar   else None
        self.i_solar_dir = idx[self.k_solar_dir] if self.k_solar_dir else None
        self.i_load      = idx[self.k_load]    if self.k_load    else None
        self.i_soc       = idx[self.k_soc]     if self.k_soc     else None

        self.solar_bins: np.ndarray = SOLAR_FALLBACK
        self.load_bins:  np.ndarray = LOAD_FALLBACK
        self.soc_bins:   np.ndarray = SOC_BINS

        self._solar_samples: List[float] = []
        self._load_samples:  List[float] = []

    # ------------------------------------------------------------------
    def fit(self, env: CityLearnEnv) -> None:
        """Warmup – zozbiera solar_generation a load vzorky pre quantile biny."""
        solar_v: List[float] = []
        load_v:  List[float] = []

        for ep_i in range(WARMUP_EPISODES):
            act_val = 1.0 if ep_i % 2 == 0 else -1.0
            env.reset()
            for _ in range(EPISODE_TIME_STEPS):
                next_obs_list, _, done, _ = env.step(
                    build_constant_actions(env, act_val)
                )
                for obs in next_obs_list:
                    s = 0.0
                    if self.i_solar is not None:
                        s += float(obs[self.i_solar])
                    if self.i_solar_dir is not None:
                        s += float(obs[self.i_solar_dir])
                    solar_v.append(s)
                    if self.i_load is not None:
                        load_v.append(float(obs[self.i_load]))
                if done:
                    break

        self._solar_samples = solar_v
        self._load_samples  = load_v
        # 4 hraničné body → 5 intervalov pre solar
        # 3 hraničné body → 4 intervaly pre load
        self.solar_bins = quantile_bins(solar_v, 4, SOLAR_FALLBACK)
        self.load_bins  = quantile_bins(load_v,  3, LOAD_FALLBACK)

    # ------------------------------------------------------------------
    def state_space_size(self) -> int:
        n_solar = len(self.solar_bins) + 1
        n_load  = len(self.load_bins)  + 1
        n_soc   = len(self.soc_bins)   + 1
        return 4 * 24 * n_solar * n_load * n_soc   # 4 sezóny

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        def stats(v: List[float]) -> str:
            if not v:
                return "n/a"
            a = np.asarray(v, dtype=np.float32)
            return f"min={a.min():.3g}  μ={a.mean():.3g}  max={a.max():.3g}  (n={a.size})"

        def bstr(b: np.ndarray) -> str:
            return "[" + ", ".join(f"{x:.3g}" for x in b) + "]"

        NF = "(nenájdený)"
        solar_note = (
            f"{self.k_solar or NF}"
            + (f" + {self.k_solar_dir}" if self.k_solar_dir else "  [priame solar_generation]")
        )
        soc_note = f"PEVNÉ  →  biny: {bstr(self.soc_bins)}"

        rows = [
            ("sezóna",      "month → (month%12)//3",    None,                 None),
            ("hodina",      self.k_hour    or NF,        None,                 None),
            ("solar [kWh]", solar_note,                  self._solar_samples,  self.solar_bins),
            ("load [kWh]",  self.k_load    or NF,        self._load_samples,   self.load_bins),
            ("bat. SOC",    self.k_soc     or NF,        None,                 None),
        ]
        sep = "─" * 96
        print(f"\n    {sep}")
        print(f"    {'Dimenzia':<15}  {'Použitý kľúč':<54}  Štatistiky warmup vzoriek")
        print(f"    {sep}")
        for feat, key, samples, bins in rows:
            if feat == "bat. SOC":
                print(f"    {feat:<15}  {str(key):<54}  {soc_note}")
            elif feat == "sezóna":
                print(f"    {feat:<15}  {str(key):<54}  —  (0=zima 1=jar 2=leto 3=jeseň)")
            else:
                s = stats(samples) if samples is not None else "—"
                print(f"    {feat:<15}  {str(key):<54}  {s}")
                if bins is not None:
                    print(f"    {'':15}    → biny: {bstr(bins)}")
        print(f"    {sep}")
        print(f"    Stavový priestor: 4 × 24 × {len(self.solar_bins)+1} × {len(self.load_bins)+1} × {len(self.soc_bins)+1}"
              f" = {self.state_space_size():,} stavov/budovu\n")


# ═══════════════════════════════════════════════════════════════════════════════
# KÓDOVANIE STAVU
# ═══════════════════════════════════════════════════════════════════════════════

# 5-tica: (sezóna, hodina, solar_bin, load_bin, soc_bin)
StateT = Tuple[int, int, int, int, int]


def encode_state(obs: List[float], bins: BinSet, soc: float) -> StateT:
    """
    Zakóduje pozorovanie do diskrétneho 5D stavu.

    (sezóna, hodina, solar_bin, load_bin, soc_bin)
    """
    month  = int(round(float(obs[bins.i_month]))) if bins.i_month is not None else 1
    hour   = int(np.clip(
        (int(round(float(obs[bins.i_hour]))) - 1) if bins.i_hour is not None else 0,
        0, 23
    ))
    solar  = 0.0
    if bins.i_solar is not None:
        solar += float(obs[bins.i_solar])
    if bins.i_solar_dir is not None:
        solar += float(obs[bins.i_solar_dir])
    load   = float(obs[bins.i_load]) if bins.i_load is not None else 1.0

    return (
        month_to_season(month),
        hour,
        dig(solar, bins.solar_bins),
        dig(load,  bins.load_bins),
        dig(soc,   bins.soc_bins),
    )


def get_socs(env: CityLearnEnv, obs_list: List[List[float]], bins: BinSet) -> List[float]:
    return [
        float(obs[bins.i_soc]) if (bins.i_soc is not None and bins.i_soc < len(obs)) else 0.0
        for obs in obs_list
    ]


def valid_actions(soc: float) -> List[int]:
    if soc <= SOC_EMPTY_THRESH:
        return [1, 2]
    if soc >= SOC_FULL_THRESH:
        return [0, 1]
    return [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════════
# Q-AGENT
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

    def act(self, states: List[StateT], socs: List[float], training: bool) -> List[int]:
        out: List[int] = []
        for b, (state, soc) in enumerate(zip(states, socs)):
            allowed = valid_actions(soc)
            if training and np.random.rand() < self.epsilon:
                out.append(int(np.random.choice(allowed)))
            else:
                q = self._q(b, state)
                out.append(int(max(allowed, key=lambda a: q[a])))
        return out

    def update(
        self,
        states:      List[StateT],
        acts:        List[int],
        rewards:     np.ndarray,
        next_states: List[StateT],
        next_socs:   List[float],
        done:        bool,
        alpha:       float,          # klesajúca rýchlosť – FIX #1
    ) -> None:
        for b in range(self.n_buildings):
            q       = self._q(b, states[b])
            q_nxt   = self._q(b, next_states[b])
            allowed = valid_actions(next_socs[b])
            best    = float(max(q_nxt[a] for a in allowed))
            target  = float(rewards[b]) + (0.0 if done else GAMMA * best)
            a       = acts[b]
            q[a]    = (1.0 - alpha) * q[a] + alpha * target

    def set_epsilon(self, episode: int, total: int) -> None:
        progress     = float(episode) / max(1, total - 1)
        self.epsilon = float(EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-5.0 * progress))

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
    env:     CityLearnEnv,
    agent:   TabularQAgent,
    bins:    BinSet,
    episode: int,
) -> Tuple[float, float, float]:
    alpha    = get_alpha(episode, TRAIN_EPISODES)   # klesajúci α
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

        agent.update(states, acts, rewards, next_states, next_socs, done, alpha)

        ep_reward += float(np.sum(rewards))
        ep_energy += grid_import(env)
        ep_cost   += district_cost(env)
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
            states = [encode_state(obs, bins, soc) for obs, soc in zip(obs_list, socs)]

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

def plot_training(
    rewards: List[float],
    energy:  List[float],
    cost:    List[float],
    alphas:  List[float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Q_learning1204_12 – Tréning  (sezóna+hodina+solar+load+SOC,  α klesá)",
        fontsize=12, fontweight="bold"
    )
    specs = [
        (rewards, "Kumulatívna odmena",      "Odmena",      "steelblue", axes[0, 0]),
        (energy,  "Odber energie (tréning)", "Energia [kWh]","teal",     axes[0, 1]),
        (cost,    "Náklady (tréning)",       "Náklady [$]", "darkorange", axes[1, 0]),
        (alphas,  "Rýchlosť učenia α",       "α",           "crimson",   axes[1, 1]),
    ]
    for data, title, ylabel, color, ax in specs:
        ax.plot(data, alpha=0.35, color=color, linewidth=0.8)
        if title != "Rýchlosť učenia α":
            ax.plot(moving_average(data, 20), linewidth=2, color="navy", label="MA(20)")
            ax.legend(fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Epizóda")
        ax.set_ylabel(ylabel)
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

    e_pct = np.array([safe_pct(b, q) if safe_pct(b, q) is not None else float("nan")
                      for b, q in zip(be, qe)])
    c_pct = np.array([safe_pct(b, q) if safe_pct(b, q) is not None else float("nan")
                      for b, q in zip(bc, qc)])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Porovnanie per-epizóda: Fixná stratégia vs Q-learning (1204_12)\n"
        "stav: (sezóna, hodina, solar_generation, load/obsadenosť, SOC)",
        fontsize=12, fontweight="bold"
    )
    w  = 0.38
    x  = np.arange(n)
    tk = x[::max(1, n // 15)]

    for ax, b_data, q_data, title, ylabel in [
        (axes[0, 0], be, qe, "Odber energie [kWh]", "kWh"),
        (axes[0, 1], bc, qc, "Náklady [$]",          "$"),
    ]:
        ax.bar(x - w / 2, b_data, w, label="Fixná stratégia", color="tomato",    alpha=0.85)
        ax.bar(x + w / 2, q_data, w, label="Q-learning",      color="steelblue", alpha=0.85)
        ax.set_xticks(tk)
        ax.set_xticklabels(eps[tk], fontsize=8)
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
    print("  Q_learning1204_12 – CityLearn tabular Q-learning (sezónny stav + α decay)")
    print("  Stav: (SEZÓNA, hodina, solar_generation, load/obsadenosť, SOC)")
    print(SEP)
    print(f"  citylearn          : {citylearn.__version__}")
    print(f"  dataset            : {DATASET_NAME}")
    print(f"  episode_time_steps : {EPISODE_TIME_STEPS}  ({EPISODE_TIME_STEPS // 24} dní)")
    print(f"  train_episodes     : {TRAIN_EPISODES}")
    print(f"  eval_episodes      : {EVAL_EPISODES}  (= 1 celý rok = 52 týždňov, bez duplikátov)")
    print(f"  α (learning rate)  : {ALPHA_START} → {ALPHA_END}  (lineárny pokles)")
    print(f"  akcie              : {ACTION_LEVELS.tolist()}  (maskované SOC < {SOC_EMPTY_THRESH} / > {SOC_FULL_THRESH})")

    # ── [1] Prostredia ──────────────────────────────────────────────────────
    print(f"\n[1/5] Inicializácia prostredí …")
    train_env   = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    eval_env    = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    n_buildings = len(train_env.buildings)
    print(f"       budov: {n_buildings}")

    # ── [2] Warmup ──────────────────────────────────────────────────────────
    print(f"\n[2/5] Warmup – výpočet solar/load binov ({WARMUP_EPISODES} ep.) …")
    bins = BinSet(train_env)
    bins.fit(train_env)
    bins.print_summary()

    # ── [3] Baseline ────────────────────────────────────────────────────────
    print(f"[3/5] Evaluácia fixnej stratégie (akcia = {FIXED_BASELINE_ACTION}) …")
    baseline = evaluate_policy(eval_env, fixed_value=FIXED_BASELINE_ACTION)
    print(f"       avg energy import : {baseline.avg_energy_import:.4f} kWh")
    print(f"       avg cost          : {baseline.avg_cost:.4f} $")

    # ── [4] Tréning ─────────────────────────────────────────────────────────
    print(f"\n[4/5] Trénovanie Q-learningu ({TRAIN_EPISODES} epizód, α klesá {ALPHA_START}→{ALPHA_END}) …")
    agent        = TabularQAgent(n_buildings)
    train_rewards: List[float] = []
    train_energy:  List[float] = []
    train_cost:    List[float] = []
    train_alphas:  List[float] = []

    for ep in tqdm(range(TRAIN_EPISODES), desc="Q-learning tréning"):
        agent.set_epsilon(ep, TRAIN_EPISODES)
        train_alphas.append(get_alpha(ep, TRAIN_EPISODES))
        r, e, c = run_training_episode(train_env, agent, bins, ep)
        train_rewards.append(r)
        train_energy.append(e)
        train_cost.append(c)

    print(f"       navštívených stavov        : {agent.total_states}")
    print(f"       α pri poslednej epizóde    : {train_alphas[-1]:.4f}")
    print(f"       posledných 20 ep. – priem. odmena : {np.mean(train_rewards[-20:]):.4f}")

    # ── [5] Evaluácia ────────────────────────────────────────────────────────
    print(f"\n[5/5] Evaluácia Q-agenta (greedy ε=0, {EVAL_EPISODES} epizód = 1 rok) …")
    q_res = evaluate_policy(eval_env, agent=agent, bins=bins)
    print(f"       avg energy import : {q_res.avg_energy_import:.4f} kWh")
    print(f"       avg cost          : {q_res.avg_cost:.4f} $")

    # ── Per-epizóda tabuľka ─────────────────────────────────────────────────
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
    print(f"\n  Per-epizóda porovnanie (52 týždňov = 1 rok, bez duplikátov):")
    print(per_ep_df.to_string(index=False))

    e_vals = [r["energy_sav_%"] for r in rows if isinstance(r["energy_sav_%"], float)]
    c_vals = [r["cost_sav_%"]   for r in rows if isinstance(r["cost_sav_%"],   float)]
    n_na_e = sum(1 for r in rows if r["energy_sav_%"] == "N/A")
    n_na_c = sum(1 for r in rows if r["cost_sav_%"]   == "N/A")

    print(f"\n  {'─' * 70}")
    print(f"  PRIEMER – energia  : {np.mean(e_vals):+.2f}%  "
          f"({'lepšie' if np.mean(e_vals) > 0 else 'horšie'} ako baseline, N/A: {n_na_e})")
    print(f"  PRIEMER – náklady  : {np.mean(c_vals):+.2f}%  "
          f"({'lepšie' if np.mean(c_vals) > 0 else 'horšie'} ako baseline, N/A: {n_na_c})")
    print(f"  {'─' * 70}")

    # ── Súhrnná tabuľka ─────────────────────────────────────────────────────
    e_sav_g = baseline.avg_energy_import - q_res.avg_energy_import
    e_pct_g = safe_pct(baseline.avg_energy_import, q_res.avg_energy_import)
    c_sav_g = baseline.avg_cost - q_res.avg_cost
    c_pct_g = safe_pct(baseline.avg_cost, q_res.avg_cost)

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
            "policy":            "q_learning_1204_12",
            "avg_energy_import": q_res.avg_energy_import,
            "avg_cost":          q_res.avg_cost,
            "energy_sav":        e_sav_g,
            "energy_sav_%":      e_pct_g if e_pct_g is not None else float("nan"),
            "cost_sav":          c_sav_g,
            "cost_sav_%":        c_pct_g if c_pct_g is not None else float("nan"),
        },
    ])
    summary_df.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)

    print(f"\n  Súhrnné výsledky (52 eval epizód = 1 celý rok):")
    print(f"  {'─' * 70}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"  {'─' * 70}")

    # ── Grafy ────────────────────────────────────────────────────────────────
    plot_training(train_rewards, train_energy, train_cost, train_alphas)
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
