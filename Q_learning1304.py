#!/usr/bin/env python
"""
Q_learning1304.py
=================
Úloha: Naučiť RL agenta optimalizovať spotrebu energie v budove
        na základe predpovedí počasia a obsadenosti miestností.

Prečo predchádzajúce verzie nesplnili úlohu
-------------------------------------------
V15 (a všetky predošlé) mal v stave "peak_price" → agent sa naučil
CENOVÚ ARBITRÁŽ (nabiť za lacno, vybiť za draho), nie ENERGETICKÚ
OPTIMALIZÁCIU (nabiť zo solárnej energie, vybiť keď budova potrebuje).

Výsledok V15: energia +10% (horšie!), náklady -7% (lepšie)
→ agent kupoval lacnú sieťovú energiu v noci a "predával" večer

Správne riešenie
----------------
1. STATE: (hodina, solar_bin, load_bin, soc_bin)
   - hodina    : 0–23 → ČASOVÝ kontext
   - solar_bin : irradiance SUM (difúzna+priama, predpovedaná 6h) → PREDPOVEĎ POČASIA
   - load_bin  : non_shiftable_load → PROXY OBSADENOSTI (viac záťaže = viac ľudí)
   - soc_bin   : SOC batérie → STAV ZÁSOBNÍKA
   → 24 × 4 × 3 × 6 = 1 728 stavov / budovu  (bez ceny = bez arbitráže)

2. REWARD FUNKCIE — porovnanie (požiadavka zadania):
   Varianta A – DefaultReward:
       r = -(max(net_electricity_consumption, 0))
       Problém: keď solár > záťaž → r = 0 → ŽIADNY GRADIENT pre nabíjanie zo solára!
       Agent sa zabudne nabiť počas slnečného dňa.

   Varianta B – SolarPenaltyReward (CityLearn built-in):
       r = -(1 + sign(e) * soc) * |e|   kde e = net_elec_consumption
       + keď solární prebytok (e < 0) a SOC nízke → agent DOSTÁVA ODMENU za kapacitu
       + keď sieťový import (e > 0) a SOC vysoké → DVOJNÁSOBNÁ POKUTA (prečo si nevybil?!)
       → Agent sa naučí: nabiť počas solárnej energie, vybiť pred solárnou špičkou

3. STRATÉGIA KTORÚ AGENT MUSÍ OBJAVIŤ:
   - Ráno (6–9h), slabé solar → vybiť na minimum (urobiť priestor)
   - Dopoludnie (9–14h), silné solar → nabiť maximum zo solárnej energie
   - Cez deň peak záťaž → čiastočne vybiť (znížiť import zo siete)
   - Večer (18–22h), slabé solar, vysoká záťaž → vybiť zvyšok

Meranie úspechu
---------------
Obidve varianty sa TRÉNUJÚ s rôznymi odmenami, ale EVALUUJÚ sa identicky:
  - avg energy import [kWh] vs baseline
  - avg cost [$] vs baseline
→ Spravodlivé porovnanie na rovnakých metrikách
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
from citylearn.reward_function import RewardFunction, SolarPenaltyReward


# ═══════════════════════════════════════════════════════════════════════════════
# KONFIGURÁCIA
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_NAME  = "citylearn_challenge_2022_phase_1"
RANDOM_SEED   = 42
OUTPUT_PREFIX = "q_learning1304"

EPISODE_TIME_STEPS = 24 * 7     # 1 týždeň / epizóda (168 hodín)
WARMUP_EPISODES    = 8
TRAIN_EPISODES     = 400
EVAL_EPISODES      = 52         # celý rok (52 týždňov)

ALPHA         = 0.25            # learning rate (konštantný — overené V10)
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05

# 3 akcie
ACTION_LEVELS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
N_ACTIONS = len(ACTION_LEVELS)

# SOC ochranné prahy
SOC_MIN = 0.05
SOC_MAX = 0.95

# Pevné SOC biny (fyzikálne hranice, nezávislé od dát)
SOC_BINS = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32)  # → 6 intervalov

# Záložné solar biny (W/m² irradiance súčet)
SOLAR_FALLBACK = np.array([10.0, 150.0, 600.0], dtype=np.float32)      # → 4 intervaly

# Záložné load biny (kWh)
LOAD_FALLBACK = np.array([0.5, 1.5], dtype=np.float32)                  # → 3 intervaly


# ═══════════════════════════════════════════════════════════════════════════════
# POMOCNÉ FUNKCIE
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def resolve_schema(name: str) -> str:
    localapp = os.environ.get("LOCALAPPDATA", "")
    cached = (
        Path(localapp)
        / "intelligent-environments-lab" / "citylearn" / "Cache"
        / f"v{citylearn.__version__}" / "datasets" / name / "schema.json"
    )
    return str(cached) if cached.exists() else name


def make_env(seed: int, random_split: bool) -> CityLearnEnv:
    return CityLearnEnv(
        schema=resolve_schema(DATASET_NAME),
        central_agent=False,
        episode_time_steps=EPISODE_TIME_STEPS,
        random_episode_split=random_split,
        random_seed=seed,
    )


def obs_index(env: CityLearnEnv) -> Dict[str, int]:
    return {name: i for i, name in enumerate(env.observation_names[0])}


def first_key(idx: Dict[str, int], candidates: List[str]) -> Optional[str]:
    for k in candidates:
        if k in idx:
            return k
    return None


def const_actions(env: CityLearnEnv, val: float) -> List[List[float]]:
    return [[float(val)] * int(sp.shape[0]) for sp in env.action_space]


def id_actions(env: CityLearnEnv, ids: List[int]) -> List[List[float]]:
    return [[float(ACTION_LEVELS[ids[b]])] * int(sp.shape[0])
            for b, sp in enumerate(env.action_space)]


def reward_array(raw, n: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.full(n, float(arr[0]) / max(1, n), dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
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


def safe_pct(base: float, new: float) -> Optional[float]:
    if abs(base) < 1e-6:
        return None
    return float(np.clip(100.0 * (base - new) / abs(base), -500.0, 500.0))


def digitize(val: float, bins: np.ndarray) -> int:
    v = val if np.isfinite(val) else 0.0
    return int(np.digitize(v, bins, right=False))


def quantile_bins(values: List[float], n_bins: int, fallback: np.ndarray) -> np.ndarray:
    """Vráti n_bins-1 hraníc z kvantilov. Ak málo dát → fallback."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float32)
    if arr.size < max(30, n_bins * 5):
        return fallback
    qs   = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    bins = np.unique(np.quantile(arr, qs))
    return bins.astype(np.float32) if bins.size > 0 else fallback


def moving_avg(vals: List[float], w: int = 25) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    if w <= 1 or arr.size < 2:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


# ═══════════════════════════════════════════════════════════════════════════════
# BINY STAVOVÉHO PRIESTORU
# ═══════════════════════════════════════════════════════════════════════════════

class StateBins:
    """
    Stavový priestor: (hodina, solar_bin, load_bin, soc_bin)

    solar: diffuse + direct irradiance PREDICTED 6h (W/m²) — predpoveď počasia
    load : non_shiftable_load (kWh) — proxy obsadenosti budovy
    soc  : electrical_storage_soc — stav batérie

    Biny solar a load sa určujú z warmup epizód (kvantilovanie).
    """

    def __init__(self, env: CityLearnEnv) -> None:
        idx = obs_index(env)

        self.k_hour = first_key(idx, ["hour"])
        self.k_diff = first_key(idx, [
            "diffuse_solar_irradiance_predicted_6h",
            "diffuse_solar_irradiance_predicted_1",
            "diffuse_solar_irradiance",
        ])
        self.k_dir  = first_key(idx, [
            "direct_solar_irradiance_predicted_6h",
            "direct_solar_irradiance_predicted_1",
            "direct_solar_irradiance",
        ])
        self.k_load = first_key(idx, [
            "non_shiftable_load",
            "non_shiftable_load_predicted_6h",
        ])
        self.k_soc  = first_key(idx, ["electrical_storage_soc"])

        self.i_hour = idx[self.k_hour] if self.k_hour else None
        self.i_diff = idx[self.k_diff] if self.k_diff else None
        self.i_dir  = idx[self.k_dir]  if self.k_dir  else None
        self.i_load = idx[self.k_load] if self.k_load else None
        self.i_soc  = idx[self.k_soc]  if self.k_soc  else None

        self.solar_bins: np.ndarray = SOLAR_FALLBACK
        self.load_bins:  np.ndarray = LOAD_FALLBACK
        self.soc_bins:   np.ndarray = SOC_BINS

        self._solar_smp: List[float] = []
        self._load_smp:  List[float] = []

    def fit(self, env: CityLearnEnv) -> None:
        """Zbiera vzorky počas warmup epizód a vypočíta adaptívne biny."""
        solar_v: List[float] = []
        load_v:  List[float] = []

        for ep_i in range(WARMUP_EPISODES):
            env.reset()
            action_val = 1.0 if ep_i % 2 == 0 else -1.0
            for _ in range(EPISODE_TIME_STEPS):
                obs_list, _, done, _ = env.step(const_actions(env, action_val))
                for obs in obs_list:
                    s = 0.0
                    if self.i_diff is not None:
                        s += float(obs[self.i_diff])
                    if self.i_dir is not None:
                        s += float(obs[self.i_dir])
                    solar_v.append(s)
                    if self.i_load is not None:
                        load_v.append(float(obs[self.i_load]))
                if done:
                    break

        self._solar_smp = solar_v
        self._load_smp  = load_v

        # 3 hrany → 4 solar biny (noc / slabé / stredné / silné)
        self.solar_bins = quantile_bins(solar_v, 4, SOLAR_FALLBACK)
        # 2 hrany → 3 load biny (nízka / stredná / vysoká záťaž)
        self.load_bins  = quantile_bins(load_v,  3, LOAD_FALLBACK)

    @property
    def n_states(self) -> int:
        return 24 * (len(self.solar_bins) + 1) * (len(self.load_bins) + 1) * (len(self.soc_bins) + 1)

    def print_summary(self) -> None:
        def bstr(b: np.ndarray) -> str:
            return "[" + ", ".join(f"{x:.3g}" for x in b) + "]"

        def stats(v: List[float]) -> str:
            if not v:
                return "—"
            a = np.asarray(v, dtype=np.float32)
            return f"min={a.min():.3g}  μ={a.mean():.3g}  max={a.max():.3g}  (n={a.size})"

        sep = "─" * 100
        ns  = len(self.solar_bins) + 1
        nl  = len(self.load_bins)  + 1
        nsc = len(self.soc_bins)   + 1
        cov = TRAIN_EPISODES * EPISODE_TIME_STEPS * 5 // (self.n_states * N_ACTIONS)
        ok  = "✓ dostatočné" if cov >= 20 else "⚠ málo"
        solar_key = f"{self.k_diff or '?'} + {self.k_dir or '?'}  [W/m² suma]"

        print(f"\n    {sep}")
        print(f"    {'Dimenzia':<16}  {'Kľúč / popis':<55}  Štatistiky warmup")
        print(f"    {sep}")
        print(f"    {'hodina':<16}  {str(self.k_hour or '(nenájd.)'):<55}  0–23 (24 tried)")
        print(f"    {'solar W/m²':<16}  {solar_key:<55}  {stats(self._solar_smp)}")
        print(f"    {'':16}    biny: {bstr(self.solar_bins)}  ({ns} intervalov)")
        print(f"    {'záťaž kWh':<16}  {str(self.k_load or '(nenájd.)'):<55}  {stats(self._load_smp)}")
        print(f"    {'':16}    biny: {bstr(self.load_bins)}  ({nl} intervalov)")
        print(f"    {'bat. SOC':<16}  {str(self.k_soc or '(nenájd.)'):<55}  PEVNÉ biny: {bstr(self.soc_bins)}")
        print(f"    {sep}")
        print(f"    Stavový priestor: 24 × {ns} × {nl} × {nsc} = {self.n_states:,} stavov/budovu")
        print(f"    Pokrytie: ~{cov} návštev / stav·akcia  ({ok})\n")


# ═══════════════════════════════════════════════════════════════════════════════
# KÓDOVANIE STAVU
# ═══════════════════════════════════════════════════════════════════════════════

StateT = Tuple[int, int, int, int]  # (hodina, solar_bin, load_bin, soc_bin)


def encode(obs: List[float], bins: StateBins, soc: float) -> StateT:
    hour = int(np.clip(
        (int(round(float(obs[bins.i_hour]))) - 1) if bins.i_hour is not None else 0,
        0, 23,
    ))

    irr = 0.0
    if bins.i_diff is not None:
        irr += float(obs[bins.i_diff])
    if bins.i_dir is not None:
        irr += float(obs[bins.i_dir])

    load = float(obs[bins.i_load]) if bins.i_load is not None else 1.0

    return (
        hour,
        digitize(irr,  bins.solar_bins),
        digitize(load, bins.load_bins),
        digitize(soc,  bins.soc_bins),
    )


def get_socs(env: CityLearnEnv, obs_list: List[List[float]], bins: StateBins) -> List[float]:
    return [
        float(obs[bins.i_soc]) if (bins.i_soc is not None and bins.i_soc < len(obs)) else 0.0
        for obs in obs_list
    ]


def allowed_actions(soc: float) -> List[int]:
    """Zabraňuje nabíjaniu pri plnej/vybíjaniu pri prázdnej batérii."""
    if soc <= SOC_MIN:
        return [1, 2]   # iba hold alebo charge
    if soc >= SOC_MAX:
        return [0, 1]   # iba discharge alebo hold
    return [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════════
# Q-AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class QAgent:
    """
    Tabulárny Q-learning s epsilon-greedy exploráciou.

    Q-tabuľka je lazy dict: stav nie je inicializovaný kým nenavštívime.
    Inicializácia na 0 = neutrálna (agent sa nezaujíma o žiadnu akciu na začiatku).
    """

    def __init__(self, n_buildings: int) -> None:
        self.n_buildings = n_buildings
        self.epsilon     = EPSILON_START
        self.q: List[Dict[StateT, np.ndarray]] = [{} for _ in range(n_buildings)]

    def _q(self, b: int, state: StateT) -> np.ndarray:
        q = self.q[b].get(state)
        if q is None:
            q = np.zeros(N_ACTIONS, dtype=np.float32)
            self.q[b][state] = q
        return q

    def act(self, states: List[StateT], socs: List[float], training: bool) -> List[int]:
        out: List[int] = []
        for b, (s, soc) in enumerate(zip(states, socs)):
            avail = allowed_actions(soc)
            if training and np.random.rand() < self.epsilon:
                out.append(int(np.random.choice(avail)))
            else:
                q = self._q(b, s)
                out.append(int(max(avail, key=lambda a: q[a])))
        return out

    def update(self, states: List[StateT], acts: List[int], rewards: np.ndarray,
               next_states: List[StateT], next_socs: List[float], done: bool) -> None:
        for b in range(self.n_buildings):
            q     = self._q(b, states[b])
            q_nxt = self._q(b, next_states[b])
            avail = allowed_actions(next_socs[b])
            best  = float(max(q_nxt[a] for a in avail))
            tgt   = float(rewards[b]) + (0.0 if done else GAMMA * best)
            a     = acts[b]
            q[a]  = (1.0 - ALPHA) * q[a] + ALPHA * tgt

    def decay_epsilon(self, ep: int, total: int) -> None:
        t = float(ep) / max(1, total - 1)
        self.epsilon = float(EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-5.0 * t))

    @property
    def visited_states(self) -> int:
        return sum(len(d) for d in self.q)


# ═══════════════════════════════════════════════════════════════════════════════
# TRÉNOVANIE
# ═══════════════════════════════════════════════════════════════════════════════

def train_episode(env: CityLearnEnv, agent: QAgent,
                  bins: StateBins) -> Tuple[float, float, float]:
    obs  = env.reset()
    socs = get_socs(env, obs, bins)
    ss   = [encode(o, bins, soc) for o, soc in zip(obs, socs)]

    ep_rew = ep_en = ep_cost = 0.0
    for _ in range(EPISODE_TIME_STEPS):
        acts   = agent.act(ss, socs, training=True)
        n_obs, raw_rew, done, _ = env.step(id_actions(env, acts))
        rews   = reward_array(raw_rew, len(env.buildings))
        n_socs = get_socs(env, n_obs, bins)
        n_ss   = [encode(o, bins, soc) for o, soc in zip(n_obs, n_socs)]

        agent.update(ss, acts, rews, n_ss, n_socs, done)

        ep_rew  += float(np.sum(rews))
        ep_en   += grid_import(env)
        ep_cost += district_cost(env)

        obs, ss, socs = n_obs, n_ss, n_socs
        if done:
            break

    return ep_rew, ep_en, ep_cost


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUÁCIA
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    avg_energy: float
    avg_cost:   float
    ep_energy:  List[float]
    ep_cost:    List[float]


def evaluate(env: CityLearnEnv,
             agent: Optional[QAgent] = None,
             bins: Optional[StateBins] = None,
             fixed: Optional[float] = None) -> EvalResult:
    assert (agent is not None) != (fixed is not None), "Presne jeden z agent/fixed musí byť zadaný"

    ep_en:   List[float] = []
    ep_cost: List[float] = []

    for _ in range(EVAL_EPISODES):
        obs  = env.reset()
        ep_e = ep_c = 0.0

        if agent is not None:
            socs = get_socs(env, obs, bins)
            ss   = [encode(o, bins, soc) for o, soc in zip(obs, socs)]

        for _ in range(EPISODE_TIME_STEPS):
            if agent is not None:
                acts    = agent.act(ss, socs, training=False)
                actions = id_actions(env, acts)
            else:
                actions = const_actions(env, fixed)

            n_obs, _, done, _ = env.step(actions)
            ep_e += grid_import(env)
            ep_c += district_cost(env)

            if agent is not None:
                socs = get_socs(env, n_obs, bins)
                ss   = [encode(o, bins, soc) for o, soc in zip(n_obs, socs)]
            obs = n_obs
            if done:
                break

        ep_en.append(ep_e)
        ep_cost.append(ep_c)

    return EvalResult(
        avg_energy=float(np.mean(ep_en)),
        avg_cost=float(np.mean(ep_cost)),
        ep_energy=ep_en,
        ep_cost=ep_cost,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFY
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training(rewards_a: List[float], rewards_b: List[float]) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rewards_a, alpha=0.3, color="steelblue", linewidth=0.7, label="A (default reward)")
    ax.plot(moving_avg(rewards_a), linewidth=2, color="steelblue", label="A MA(25)")
    ax.plot(rewards_b, alpha=0.3, color="darkorange", linewidth=0.7, label="B (SolarPenalty)")
    ax.plot(moving_avg(rewards_b), linewidth=2, color="darkorange", label="B MA(25)")
    ax.set_title("Q_learning1304 – Kumulatívna odmena počas tréningu", fontweight="bold")
    ax.set_xlabel("Epizóda")
    ax.set_ylabel("Odmena")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_training.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_energy_comparison(baseline: EvalResult, res_a: EvalResult,
                           res_b: EvalResult) -> None:
    n   = min(len(baseline.ep_energy), len(res_a.ep_energy), len(res_b.ep_energy))
    eps = np.arange(1, n + 1)
    be  = np.array(baseline.ep_energy[:n])
    ae  = np.array(res_a.ep_energy[:n])
    xe  = np.array(res_b.ep_energy[:n])

    def pct_arr(base, new):
        return np.array([
            safe_pct(b, v) if safe_pct(b, v) is not None else float("nan")
            for b, v in zip(base, new)
        ])

    a_pct = pct_arr(be, ae)
    x_pct = pct_arr(be, xe)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(
        "Q_learning1304 — Porovnanie: Fixná / QL-A (default reward) / QL-B (SolarPenalty)\n"
        "Stav: (hodina, solar_irr_pred, non_shiftable_load, SOC)   |   3 akcie",
        fontsize=11, fontweight="bold",
    )

    # --- Energia absolútne ---
    ax = axes[0, 0]
    w  = 0.28
    x  = np.arange(n)
    tk = x[::max(1, n // 12)]
    ax.bar(x - w, be, w, label="Fixná",    color="tomato",    alpha=0.85)
    ax.bar(x,     ae, w, label="QL-A def", color="steelblue", alpha=0.85)
    ax.bar(x + w, xe, w, label="QL-B sol", color="darkorange",alpha=0.85)
    ax.set_xticks(tk); ax.set_xticklabels(eps[tk], fontsize=8)
    ax.set_title("Odber energie [kWh]")
    ax.set_xlabel("Eval epizóda (týždeň)")
    ax.set_ylabel("kWh")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Náklady absolútne ---
    bc = np.array(baseline.ep_cost[:n])
    ac = np.array(res_a.ep_cost[:n])
    xc = np.array(res_b.ep_cost[:n])
    ax = axes[0, 1]
    ax.bar(x - w, bc, w, label="Fixná",    color="tomato",    alpha=0.85)
    ax.bar(x,     ac, w, label="QL-A def", color="steelblue", alpha=0.85)
    ax.bar(x + w, xc, w, label="QL-B sol", color="darkorange",alpha=0.85)
    ax.set_xticks(tk); ax.set_xticklabels(eps[tk], fontsize=8)
    ax.set_title("Náklady [$]")
    ax.set_xlabel("Eval epizóda (týždeň)")
    ax.set_ylabel("$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Úspora energie % ---
    for ax_i, pct, label, color, ax_pos in [
        (0, a_pct, "QL-A def  úspora energie %", "steelblue",  axes[1, 0]),
        (1, x_pct, "QL-B sol  úspora energie %", "darkorange", axes[1, 1]),
    ]:
        ax = ax_pos
        valid = ~np.isnan(pct)
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in np.where(valid, pct, 0)]
        ax.bar(eps[valid], pct[valid],
               color=[colors[i] for i in np.where(valid)[0]], alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        if np.any(valid):
            mn = float(np.nanmean(pct[valid]))
            ax.axhline(mn, color="navy", linewidth=1.8, linestyle="--",
                       label=f"priemer = {mn:+.1f}%")
        ax.set_title(label)
        ax.set_xlabel("Eval epizóda (týždeň)")
        ax.set_ylabel("%  (kladné = lepšie ako fixná)")
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
    print("  Q_learning1304 – energia + obsadenosť + predpoveď počasia")
    print("  Stav: (hodina, solar_irr_pred, záťaž_load, SOC) | 3 akcie | α=0.25")
    print("  Porovnanie: DefaultReward vs SolarPenaltyReward")
    print(SEP)
    print(f"  citylearn          : {citylearn.__version__}")
    print(f"  train_episodes     : {TRAIN_EPISODES}")
    print(f"  eval_episodes      : {EVAL_EPISODES}")
    print(f"  episode_time_steps : {EPISODE_TIME_STEPS} h  ({EPISODE_TIME_STEPS // 24} dní)")

    # ── [1] Prostredia ───────────────────────────────────────────────────────
    print(f"\n[1/6] Inicializácia prostredí …")
    # Tréningové prostredia — jedno pre každú variantu rewarda
    train_env_a = make_env(RANDOM_SEED, random_split=True)   # default reward
    train_env_b = make_env(RANDOM_SEED, random_split=True)   # SolarPenaltyReward
    eval_env    = make_env(RANDOM_SEED, random_split=False)   # pouze evaluácia

    # Nastav SolarPenaltyReward na env_b po inicializácii
    train_env_b.reward_function = SolarPenaltyReward(train_env_b.get_metadata())

    n_bld = len(train_env_a.buildings)
    print(f"       budov: {n_bld}")
    print(f"       train_env_a reward : {train_env_a.reward_function.__class__.__name__}")
    print(f"       train_env_b reward : {train_env_b.reward_function.__class__.__name__}")

    # ── [2] Warmup + biny ───────────────────────────────────────────────────
    print(f"\n[2/6] Warmup – kvantilovanie solar a load binov ({WARMUP_EPISODES} ep.) …")
    bins = StateBins(train_env_a)
    bins.fit(train_env_a)
    bins.print_summary()

    # ── [3] Baseline ────────────────────────────────────────────────────────
    print(f"[3/6] Evaluácia fixnej stratégie (akcia = 0.0, batéria nečinná) …")
    baseline = evaluate(eval_env, fixed=0.0)
    print(f"       avg energy : {baseline.avg_energy:.4f} kWh")
    print(f"       avg cost   : {baseline.avg_cost:.4f} $")

    # ── [4] Tréning A – DefaultReward ────────────────────────────────────────
    print(f"\n[4/6] Tréning Varianta A – DefaultReward ({TRAIN_EPISODES} ep.) …")
    agent_a = QAgent(n_bld)
    rew_a: List[float] = []
    for ep in tqdm(range(TRAIN_EPISODES), desc="QL-A DefaultReward"):
        agent_a.decay_epsilon(ep, TRAIN_EPISODES)
        r, _, _ = train_episode(train_env_a, agent_a, bins)
        rew_a.append(r)
    print(f"       navštívených stavov : {agent_a.visited_states}")

    # ── [5] Tréning B – SolarPenaltyReward ───────────────────────────────────
    print(f"\n[5/6] Tréning Varianta B – SolarPenaltyReward ({TRAIN_EPISODES} ep.) …")
    agent_b = QAgent(n_bld)
    rew_b: List[float] = []
    for ep in tqdm(range(TRAIN_EPISODES), desc="QL-B SolarPenalty"):
        agent_b.decay_epsilon(ep, TRAIN_EPISODES)
        r, _, _ = train_episode(train_env_b, agent_b, bins)
        rew_b.append(r)
    print(f"       navštívených stavov : {agent_b.visited_states}")

    # ── [6] Evaluácia oboch agentov ──────────────────────────────────────────
    print(f"\n[6/6] Evaluácia agentov (greedy, ε=0, {EVAL_EPISODES} ep.) …")
    res_a = evaluate(eval_env, agent=agent_a, bins=bins)
    res_b = evaluate(eval_env, agent=agent_b, bins=bins)
    print(f"       QL-A: energy={res_a.avg_energy:.4f} kWh, cost={res_a.avg_cost:.4f} $")
    print(f"       QL-B: energy={res_b.avg_energy:.4f} kWh, cost={res_b.avg_cost:.4f} $")

    # ── Per-epizóda tabuľka ──────────────────────────────────────────────────
    n_ep = min(EVAL_EPISODES,
               len(baseline.ep_energy), len(res_a.ep_energy), len(res_b.ep_energy))
    rows = []
    for i in range(n_ep):
        be_i = baseline.ep_energy[i]
        ae_i = res_a.ep_energy[i]
        xe_i = res_b.ep_energy[i]
        bc_i = baseline.ep_cost[i]
        ac_i = res_a.ep_cost[i]
        xc_i = res_b.ep_cost[i]
        rows.append({
            "ep":         i + 1,
            "base_en":    round(be_i, 2),
            "ql_a_en":    round(ae_i, 2),
            "en_sav_a_%": round(safe_pct(be_i, ae_i), 2) if safe_pct(be_i, ae_i) is not None else "N/A",
            "ql_b_en":    round(xe_i, 2),
            "en_sav_b_%": round(safe_pct(be_i, xe_i), 2) if safe_pct(be_i, xe_i) is not None else "N/A",
            "base_cost":  round(bc_i, 3),
            "ql_a_cost":  round(ac_i, 3),
            "ql_b_cost":  round(xc_i, 3),
        })

    df_ep = pd.DataFrame(rows)
    df_ep.to_csv(f"{OUTPUT_PREFIX}_per_episode.csv", index=False)
    print(f"\n  Per-epizóda ({n_ep} týždňov):")
    print(df_ep.to_string(index=False))

    # ── Súhrnné štatistiky ───────────────────────────────────────────────────
    def avg_pct(ep_base, ep_new):
        vals = [safe_pct(b, n) for b, n in zip(ep_base[:n_ep], ep_new[:n_ep])]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    ea_energy = avg_pct(baseline.ep_energy, res_a.ep_energy)
    eb_energy = avg_pct(baseline.ep_energy, res_b.ep_energy)
    ea_cost   = avg_pct(baseline.ep_cost,   res_a.ep_cost)
    eb_cost   = avg_pct(baseline.ep_cost,   res_b.ep_cost)

    def status(v): return "✓ LEPŠIE" if v > 0 else "✗ horšie"

    print(f"\n  {'═' * 72}")
    print(f"  SÚHRNNÉ VÝSLEDKY ({n_ep} eval epizód = celý rok)")
    print(f"  {'─' * 72}")
    print(f"  {'Metrika':<35}  {'QL-A DefaultReward':>18}  {'QL-B SolarPenalty':>18}")
    print(f"  {'─' * 72}")
    print(f"  {'Priem. úspora energie %':<35}  {ea_energy:>+17.2f}%  {eb_energy:>+17.2f}%")
    print(f"  {'':35}  {status(ea_energy):>18}  {status(eb_energy):>18}")
    print(f"  {'Priem. úspora nákladov %':<35}  {ea_cost:>+17.2f}%  {eb_cost:>+17.2f}%")
    print(f"  {'':35}  {status(ea_cost):>18}  {status(eb_cost):>18}")
    print(f"  {'─' * 72}")
    print(f"  Absol. energia — baseline  : {baseline.avg_energy:.4f} kWh")
    print(f"  Absol. energia — QL-A      : {res_a.avg_energy:.4f} kWh")
    print(f"  Absol. energia — QL-B      : {res_b.avg_energy:.4f} kWh")
    print(f"  {'─' * 72}")
    print(f"  Absol. náklady — baseline  : {baseline.avg_cost:.4f} $")
    print(f"  Absol. náklady — QL-A      : {res_a.avg_cost:.4f} $")
    print(f"  Absol. náklady — QL-B      : {res_b.avg_cost:.4f} $")
    print(f"  {'═' * 72}")

    # ── Výsledková tabuľka + CSV ─────────────────────────────────────────────
    summary = pd.DataFrame([
        {"policy": "baseline_fixed_0",      "reward_fn": "DefaultReward",
         "avg_energy": baseline.avg_energy, "avg_cost": baseline.avg_cost,
         "energy_sav_%": 0.0, "cost_sav_%": 0.0},
        {"policy": "ql_a_default_reward",   "reward_fn": "DefaultReward",
         "avg_energy": res_a.avg_energy,    "avg_cost": res_a.avg_cost,
         "energy_sav_%": round(ea_energy, 4), "cost_sav_%": round(ea_cost, 4)},
        {"policy": "ql_b_solar_penalty",    "reward_fn": "SolarPenaltyReward",
         "avg_energy": res_b.avg_energy,    "avg_cost": res_b.avg_cost,
         "energy_sav_%": round(eb_energy, 4), "cost_sav_%": round(eb_cost, 4)},
    ])
    summary.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)

    # ── Grafy ─────────────────────────────────────────────────────────────────
    plot_training(rew_a, rew_b)
    plot_energy_comparison(baseline, res_a, res_b)

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
