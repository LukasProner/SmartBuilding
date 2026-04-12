#!/usr/bin/env python
"""
Q_learning1204_10.py
====================
Vylepšený tabulárny Q-learning pre CityLearn Challenge 2022 Phase 1.

Hlavné zmeny oproti Q_learning1204_09.py
-----------------------------------------
1. Redukovaný stavový priestor – len 4 dimenzie:
      (hodina, solar_bin, price_bin, soc_bin)
   → ~3 600 stavov/budovu namiesto >1 mil.; agent konverguje rýchlejšie.

2. Pevné SOC biny  [0.10, 0.30, 0.50, 0.70, 0.90]
   Warmup s nulovými akciami dáva SOC ≡ 0 → data-driven biny sú neplatné.

3. Maskovanie akcií – zakazuje:
      – vybíjanie (akcia –1) keď SOC < 0.05  (prázdna batéria),
      – nabíjanie (akcia +1) keď SOC > 0.95  (plná batéria).
   Greedy výber aj náhodný výber rešpektujú masku.

4. Bezpečný výpočet cost_sav_%
   Keď baseline cost ≤ 0 (episode so solárnym prebytkom = záporná cena),
   percentuálna metrika je nezmyselná → zobrazujeme "N/A".

5. Warmup so striedavým nabíjaním / vybíjaním
   Získame reálne solar a price distribúcie bez nulovej akcie.
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
OUTPUT_PREFIX = "q_learning1204_10"

FAST_MODE = True

if FAST_MODE:
    EPISODE_TIME_STEPS = 24 * 7    # 1 týždeň / epizóda
    TRAIN_EPISODES     = 300
    EVAL_EPISODES      = 30
else:
    EPISODE_TIME_STEPS = 24 * 14   # 2 týždne / epizóda
    TRAIN_EPISODES     = 1500
    EVAL_EPISODES      = 50

WARMUP_EPISODES = 8   # pre výpočet solar/price binov

ALPHA         = 0.25
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.05

# Akcie:  0 = vybíjanie (–1.0),  1 = drž (0.0),  2 = nabíjanie (+1.0)
ACTION_LEVELS         = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
N_ACTIONS             = len(ACTION_LEVELS)
FIXED_BASELINE_ACTION = 0.0   # baseline: nič nerob

# Masky SOC – hranice pod / nad ktorými je akcia neplatná
SOC_EMPTY_THRESH = 0.05
SOC_FULL_THRESH  = 0.95

# Pevné SOC biny (fyzikálne zmysluplné, nie z warmup dát)
SOC_BINS = np.array([0.10, 0.30, 0.50, 0.70, 0.90], dtype=np.float32)

# Fallback biny (ak warmup nemá dostatok vzoriek)
SOLAR_FALLBACK = np.array([50.0, 200.0, 500.0, 900.0], dtype=np.float32)
PRICE_FALLBACK = np.array([0.12, 0.20, 0.28, 0.38],   dtype=np.float32)


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
    """Odber zo siete v aktuálnom kroku (vždy ≥ 0)."""
    if not env.net_electricity_consumption:
        return 0.0
    return max(0.0, float(env.net_electricity_consumption[-1]))


def district_cost(env: CityLearnEnv) -> float:
    """Náklady na elektrinu v aktuálnom kroku (môže byť záporné = export)."""
    if hasattr(env, "net_electricity_consumption_cost") and env.net_electricity_consumption_cost:
        return float(env.net_electricity_consumption_cost[-1])
    return 0.0


def safe_pct(base: float, ql: float, cap: float = 500.0) -> Optional[float]:
    """
    Percentuálna úspora: 100*(base-ql)/base.
    Vracia None ak base ≤ 0 (neinterpretovateľné – solárny prebytok).
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# BINY STAVOVÉHO PRIESTORU + WARMUP
# ═══════════════════════════════════════════════════════════════════════════════

class BinSet:
    """
    Ukladá biny pre solar, price a SOC dimenzie stavového priestoru.
    SOC biny sú vždy pevné (SOC_BINS). Solar a price sa vypočítajú z warmup dát.
    """

    def __init__(self, env: CityLearnEnv) -> None:
        idx = get_obs_index(env)

        self.k_hour = first_key(idx, ["hour"])
        self.k_solar_diff = first_key(idx, [
            "diffuse_solar_irradiance_predicted_6h",
            "diffuse_solar_irradiance_predicted_1",
            "diffuse_solar_irradiance",
        ])
        self.k_solar_dir = first_key(idx, [
            "direct_solar_irradiance_predicted_6h",
            "direct_solar_irradiance_predicted_1",
            "direct_solar_irradiance",
        ])
        self.k_price = first_key(idx, [
            "electricity_pricing_predicted_6h",
            "electricity_pricing_predicted_1",
            "electricity_pricing",
        ])
        self.k_soc = first_key(idx, ["electrical_storage_soc"])

        self.i_hour      = idx[self.k_hour]      if self.k_hour      else None
        self.i_solar_diff = idx[self.k_solar_diff] if self.k_solar_diff else None
        self.i_solar_dir  = idx[self.k_solar_dir]  if self.k_solar_dir  else None
        self.i_price      = idx[self.k_price]     if self.k_price     else None
        self.i_soc        = idx[self.k_soc]       if self.k_soc       else None

        # Inicializácia fallback hodnotami; fit() prepíše solar a price
        self.solar_bins: np.ndarray = SOLAR_FALLBACK
        self.price_bins: np.ndarray = PRICE_FALLBACK
        self.soc_bins:   np.ndarray = SOC_BINS      # vždy pevné

        self._solar_samples: List[float] = []
        self._price_samples: List[float] = []

    # ------------------------------------------------------------------
    def fit(self, env: CityLearnEnv) -> None:
        """
        Spustí WARMUP_EPISODES epizód so striedavým nabíjaním / vybíjaním.
        Zbiera solar a price hodnoty → quantile biny.
        (SOC biny zostávajú pevné – neovplyvňuje ich warmup akcia.)
        """
        solar_v: List[float] = []
        price_v: List[float] = []

        for ep_i in range(WARMUP_EPISODES):
            # Striedanie +1 / –1 → SOC sa mení, solar/price dáta sú pestré
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
        self.solar_bins = quantile_bins(solar_v, 5, SOLAR_FALLBACK)
        self.price_bins = quantile_bins(price_v, 5, PRICE_FALLBACK)

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Vytlačí tabuľku: dimenzia, použitý kľúč, štatistiky, biny."""

        def stats(v: List[float]) -> str:
            if not v:
                return "n/a"
            a = np.asarray(v, dtype=np.float32)
            return f"min={a.min():.3g}  μ={a.mean():.3g}  max={a.max():.3g}  (n={a.size})"

        def bstr(b: np.ndarray) -> str:
            return "[" + ", ".join(f"{x:.3g}" for x in b) + "]"

        NF         = "(nenájdený)"
        solar_key  = f"{self.k_solar_diff or NF} + {self.k_solar_dir or NF}"
        soc_note   = f"PEVNÉ (fixné fyzikálne hranice)  →  biny: {bstr(self.soc_bins)}"

        rows = [
            ("hodina",     self.k_hour   or NF, None,                  None),
            ("solar (Σ)",  solar_key,            self._solar_samples,   self.solar_bins),
            ("cena",       self.k_price  or NF,  self._price_samples,   self.price_bins),
            ("bat. SOC",   self.k_soc    or NF,  None,                  None),
        ]
        sep = "─" * 92
        print(f"\n    {sep}")
        print(f"    {'Dimenzia':<14}  {'Použitý kľúč':<52}  Štatistiky warmup vzoriek")
        print(f"    {sep}")
        for feat, key, samples, bins in rows:
            if feat == "bat. SOC":
                print(f"    {feat:<14}  {str(key):<52}  {soc_note}")
            else:
                s = stats(samples) if samples is not None else "—"
                print(f"    {feat:<14}  {str(key):<52}  {s}")
                if bins is not None:
                    print(f"    {'':14}    → biny: {bstr(bins)}")
        print(f"    {sep}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# KÓDOVANIE STAVU
# ═══════════════════════════════════════════════════════════════════════════════

def encode_state(obs: List[float], bins: BinSet, soc: float) -> Tuple[int, int, int, int]:
    """
    Vráti diskrétny stav (hodina, solar_bin, price_bin, soc_bin).
    Hodina sa berie priamo z obs (0-23).
    soc sa preberá ako parameter (buildings[b].electrical_storage.soc).
    """
    hour = int(np.clip(
        (int(round(float(obs[bins.i_hour]))) - 1) if bins.i_hour is not None else 0,
        0, 23
    ))
    solar = 0.0
    if bins.i_solar_diff is not None:
        solar += float(obs[bins.i_solar_diff])
    if bins.i_solar_dir is not None:
        solar += float(obs[bins.i_solar_dir])
    price = float(obs[bins.i_price]) if bins.i_price is not None else 0.0

    return (
        hour,
        dig(solar, bins.solar_bins),
        dig(price, bins.price_bins),
        dig(soc,   bins.soc_bins),
    )


def get_socs(env: CityLearnEnv, obs_list: List[List[float]], bins: BinSet) -> List[float]:
    """SOC každej budovy z observations (index i_soc)."""
    return [
        float(obs[bins.i_soc]) if (bins.i_soc is not None and bins.i_soc < len(obs)) else 0.0
        for obs in obs_list
    ]


def valid_actions(soc: float) -> List[int]:
    """Povolené akcie pre danú hodnotu SOC (index do ACTION_LEVELS)."""
    if soc <= SOC_EMPTY_THRESH:
        return [1, 2]    # iba drž alebo nabíj
    if soc >= SOC_FULL_THRESH:
        return [0, 1]    # iba vybíj alebo drž
    return [0, 1, 2]


# ═══════════════════════════════════════════════════════════════════════════════
# Q-AGENT (tabulárny, per-budova, s maskovaním akcií)
# ═══════════════════════════════════════════════════════════════════════════════

class TabularQAgent:
    def __init__(self, n_buildings: int) -> None:
        self.n_buildings = n_buildings
        self.epsilon     = EPSILON_START
        self.q_tables: List[Dict[Tuple[int, int, int, int], np.ndarray]] = [
            {} for _ in range(n_buildings)
        ]

    def _q(self, b: int, state: Tuple) -> np.ndarray:
        q = self.q_tables[b].get(state)
        if q is None:
            q = np.zeros(N_ACTIONS, dtype=np.float32)
            self.q_tables[b][state] = q
        return q

    def act(
        self,
        states: List[Tuple],
        socs: List[float],
        training: bool,
    ) -> List[int]:
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
        states:      List[Tuple],
        acts:        List[int],
        rewards:     np.ndarray,
        next_states: List[Tuple],
        next_socs:   List[float],
        done:        bool,
    ) -> None:
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
    env: CityLearnEnv,
    agent: TabularQAgent,
    bins: BinSet,
) -> Tuple[float, float, float]:
    obs_list = env.reset()
    socs     = get_socs(env, obs_list, bins)
    states   = [encode_state(obs, bins, soc) for obs, soc in zip(obs_list, socs)]

    ep_reward = ep_energy = ep_cost = 0.0

    for _ in range(EPISODE_TIME_STEPS):
        acts            = agent.act(states, socs, training=True)
        next_obs_list, raw_rew, done, _ = env.step(build_actions_from_ids(env, acts))

        rewards     = reward_to_array(raw_rew, len(env.buildings))
        next_socs   = get_socs(env, next_obs_list, bins)
        next_states = [encode_state(obs, bins, soc)
                       for obs, soc in zip(next_obs_list, next_socs)]

        agent.update(states, acts, rewards, next_states, next_socs, done)

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
    """Evaluuje buď Q-agenta (agent+bins) alebo fixed stratégiu (fixed_value)."""
    assert (agent is not None) != (fixed_value is not None), \
        "Zadaj presne jeden z parametrov: agent alebo fixed_value."

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
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Q_learning1204_10 – Tréningové krivky", fontsize=13, fontweight="bold")
    specs = [
        (rewards, "Kumulatívna odmena",  "Odmena"),
        (energy,  "Odber energie (tréning)", "Energia [kWh]"),
        (cost,    "Náklady (tréning)",    "Náklady [$]"),
    ]
    for ax, (data, title, ylabel) in zip(axes, specs):
        ax.plot(data, alpha=0.35, color="steelblue", linewidth=0.8)
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

    # Percentuálna úspora – None kde baseline ≤ 0  →  NaN pre grafy
    e_pct = np.array([safe_pct(b, q) if safe_pct(b, q) is not None else float("nan")
                      for b, q in zip(be, qe)])
    c_pct = np.array([safe_pct(b, q) if safe_pct(b, q) is not None else float("nan")
                      for b, q in zip(bc, qc)])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Porovnanie per-epizóda: Fixná stratégia vs Q-learning (1204_10)",
        fontsize=13, fontweight="bold"
    )
    w  = 0.38
    x  = np.arange(n)
    tk = x[::max(1, n // 15)]

    # ── Energia a náklady (grouped bars) ──────────────────────────────────
    for ax, b_data, q_data, title, ylabel in [
        (axes[0, 0], be, qe, "Odber energie [kWh]", "kWh"),
        (axes[0, 1], bc, qc, "Náklady [$]",         "$"),
    ]:
        ax.bar(x - w / 2, b_data, w, label="Fixná stratégia", color="tomato",    alpha=0.85)
        ax.bar(x + w / 2, q_data, w, label="Q-learning",      color="steelblue", alpha=0.85)
        ax.set_xticks(tk)
        ax.set_xticklabels(eps[tk], fontsize=8)
        ax.set_xlabel("Eval epizóda")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    # ── Percentuálna úspora ────────────────────────────────────────────────
    for ax, pct, title in [
        (axes[1, 0], e_pct, "Úspora energie [%]  (zelená = QL lepšie)"),
        (axes[1, 1], c_pct, "Úspora nákladov [%]  (N/A kde baseline ≤ 0)"),
    ]:
        valid = ~np.isnan(pct)
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in np.where(valid, pct, 0)]
        ax.bar(eps[valid], pct[valid],
               color=[colors[i] for i in np.where(valid)[0]], alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        if np.any(valid):
            mean_v = float(np.nanmean(pct[valid]))
            ax.axhline(mean_v, color="navy", linewidth=1.5, linestyle="--",
                       label=f"priemer = {mean_v:.1f}%")
        n_na = int(np.sum(~valid))
        if n_na > 0:
            ax.set_title(f"{title}  [{n_na}× N/A]")
        else:
            ax.set_title(title)
        ax.set_xlabel("Eval epizóda")
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
    print("  Q_learning1204_10 – CityLearn tabular Q-learning (vylepšená verzia)")
    print(SEP)
    print(f"  citylearn          : {citylearn.__version__}")
    print(f"  dataset            : {DATASET_NAME}")
    print(f"  episode_time_steps : {EPISODE_TIME_STEPS}  ({EPISODE_TIME_STEPS // 24} dní)")
    print(f"  train_episodes     : {TRAIN_EPISODES}")
    print(f"  eval_episodes      : {EVAL_EPISODES}")
    print(f"  warmup_episodes    : {WARMUP_EPISODES}")
    print(f"  akcie              : {ACTION_LEVELS.tolist()}  (maskované pri prázdnej/plnej batérii)")

    # ── Prostredia ──────────────────────────────────────────────────────────
    print(f"\n[1/5] Inicializácia prostredí …")
    train_env   = make_env(random_seed=RANDOM_SEED, random_episode_split=True)
    eval_env    = make_env(random_seed=RANDOM_SEED, random_episode_split=False)
    n_buildings = len(train_env.buildings)
    print(f"       budov: {n_buildings}")

    # ── Warmup – biny ───────────────────────────────────────────────────────
    print(f"\n[2/5] Warmup – výpočet solar/price binov ({WARMUP_EPISODES} ep., striedavé nabíjanie) …")
    bins = BinSet(train_env)
    bins.fit(train_env)
    bins.print_summary()

    # ── Baseline (eval pred tréningom) ─────────────────────────────────────
    print(f"[3/5] Evaluácia fixnej stratégie (akcia = {FIXED_BASELINE_ACTION}) …")
    baseline = evaluate_policy(eval_env, fixed_value=FIXED_BASELINE_ACTION)
    print(f"       avg energy import : {baseline.avg_energy_import:.4f} kWh")
    print(f"       avg cost          : {baseline.avg_cost:.4f} $")

    # ── Tréning ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Trénovanie Q-learningu ({TRAIN_EPISODES} epizód) …")
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

    print(f"       navštívených stavov : {agent.total_states}")
    print(f"       posledných 20 ep. – priem. odmena : {np.mean(train_rewards[-20:]):.4f}")

    # ── Evaluácia Q-agenta ──────────────────────────────────────────────────
    print(f"\n[5/5] Evaluácia natrénovaneho Q-agenta (greedy, ε=0) …")
    q_res = evaluate_policy(eval_env, agent=agent, bins=bins)
    print(f"       avg energy import : {q_res.avg_energy_import:.4f} kWh")
    print(f"       avg cost          : {q_res.avg_cost:.4f} $")

    # ── Per-epizóda tabuľka ─────────────────────────────────────────────────
    n_ep = min(len(baseline.episode_energies), len(q_res.episode_energies))
    rows: List[dict] = []
    for i in range(n_ep):
        be_i = baseline.episode_energies[i]
        qe_i = q_res.episode_energies[i]
        bc_i = baseline.episode_costs[i]
        qc_i = q_res.episode_costs[i]
        ep = safe_pct(be_i, qe_i)
        cp = safe_pct(bc_i, qc_i)
        rows.append({
            "episode":      i + 1,
            "fixed_energy": round(be_i, 3),
            "ql_energy":    round(qe_i, 3),
            "energy_sav_%": round(ep, 2)  if ep is not None else "N/A",
            "fixed_cost":   round(bc_i, 4),
            "ql_cost":      round(qc_i, 4),
            "cost_sav_%":   round(cp, 2)  if cp is not None else "N/A",
        })

    per_ep_df = pd.DataFrame(rows)
    per_ep_df.to_csv(f"{OUTPUT_PREFIX}_per_episode.csv", index=False)

    print(f"\n  Per-epizóda porovnanie (Fixná stratégia vs Q-learning):")
    print(per_ep_df.to_string(index=False))

    # Priemery (len platné riadky)
    e_vals  = [r["energy_sav_%"] for r in rows if isinstance(r["energy_sav_%"], float)]
    c_vals  = [r["cost_sav_%"]   for r in rows if isinstance(r["cost_sav_%"],   float)]
    n_na_e  = sum(1 for r in rows if r["energy_sav_%"] == "N/A")
    n_na_c  = sum(1 for r in rows if r["cost_sav_%"]   == "N/A")

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
            "policy":            "q_learning_1204_10",
            "avg_energy_import": q_res.avg_energy_import,
            "avg_cost":          q_res.avg_cost,
            "energy_sav":        e_sav_g,
            "energy_sav_%":      e_pct_g if e_pct_g is not None else float("nan"),
            "cost_sav":          c_sav_g,
            "cost_sav_%":        c_pct_g if c_pct_g is not None else float("nan"),
        },
    ])
    summary_df.to_csv(f"{OUTPUT_PREFIX}_results.csv", index=False)

    print(f"\n  Súhrnné výsledky:")
    print(f"  {'─' * 70}")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"  {'─' * 70}")

    # ── Grafy ────────────────────────────────────────────────────────────────
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
