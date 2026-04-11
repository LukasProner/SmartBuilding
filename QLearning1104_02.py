from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_CITYLEARN_ROOT = PROJECT_ROOT / "CityLearn"
if LOCAL_CITYLEARN_ROOT.exists():
    sys.path.insert(0, str(LOCAL_CITYLEARN_ROOT))

try:
    import citylearn
    from citylearn.citylearn import CityLearnEnv
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "unknown")
    print("Chyba: chyba Python balicek potrebny pre CityLearn.")
    print(f"Missing module: {missing}")
    print("Nainstaluj zavislosti v aktivnom .venv:")
    print("  python -m pip install -r CityLearn\\requirements.txt")
    if missing == "yaml":
        print("Rychly fix pre tuto chybu:")
        print("  python -m pip install pyyaml")
    raise SystemExit(1) from exc


DATASET_NAME = "citylearn_challenge_2022_phase_1"
SEED = 0
DEFAULT_EPISODES = 40
DEFAULT_MAX_STEPS = 24 * 14
DEFAULT_BUILDINGS = 5
ACTION_LEVELS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

WEATHER_FEATURES = [
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_relative_humidity_predicted_1",
    "diffuse_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_1",
]
OCCUPANCY_FEATURES = ["occupant_count", "occupant_count_predicted_1", "occupancy"]
OCCUPANCY_PROXY_FEATURES = ["non_shiftable_load", "non_shiftable_load_predicted_1"]


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


def make_env(schema: str) -> CityLearnEnv:
    try:
        return CityLearnEnv(
            schema=schema,
            central_agent=False,
            random_episode_split=False,
            random_seed=SEED,
        )
    except TypeError:
        return CityLearnEnv(schema)


def reset_env(env: CityLearnEnv, seed: int):
    try:
        result = env.reset(seed=seed)
    except TypeError:
        result = env.reset()
    return result[0] if isinstance(result, tuple) else result


def step_env(env: CityLearnEnv, actions):
    result = env.step(actions)
    if len(result) == 5:
        obs, reward, terminated, truncated, _ = result
        return obs, reward, bool(terminated or truncated)
    if len(result) == 4:
        obs, reward, done, _ = result
        return obs, reward, bool(done)
    raise RuntimeError("Neocakavany format navratovej hodnoty env.step(...).")


def get_obs_index(env: CityLearnEnv) -> Dict[str, int]:
    return {name: i for i, name in enumerate(env.observation_names[0])}


def choose_features(env: CityLearnEnv) -> Tuple[List[str], str]:
    names = set(env.observation_names[0])

    weather = [name for name in WEATHER_FEATURES if name in names]
    if not weather:
        raise RuntimeError("Dataset nema potrebne predikcie pocasia.")

    occ = [name for name in OCCUPANCY_FEATURES if name in names]
    if occ:
        return weather + [occ[0]], "direct"

    proxy = [name for name in OCCUPANCY_PROXY_FEATURES if name in names]
    if proxy:
        return weather + [proxy[0]], "proxy"

    raise RuntimeError("Dataset nema obsadenost ani proxy obsadenosti.")


class StateEncoder:
    def __init__(self, env: CityLearnEnv, feature_names: Sequence[str], bins: int = 6):
        self.feature_names = list(feature_names)
        self.bins = int(bins)
        self.index = get_obs_index(env)
        self.space = env.observation_space[0]

    def _bin(self, feature_name: str, value: float) -> int:
        idx = self.index[feature_name]
        low = float(self.space.low[idx])
        high = float(self.space.high[idx])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return 0
        x = float(np.clip(value, low, high))
        b = int((x - low) / (high - low) * self.bins)
        return int(np.clip(b, 0, self.bins - 1))

    def encode(self, obs: Sequence[float]) -> Tuple[int, ...]:
        return tuple(self._bin(name, float(obs[self.index[name]])) for name in self.feature_names)


class TabularQAgent:
    def __init__(self, n_actions: int, alpha: float = 0.2, gamma: float = 0.99):
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q: Dict[Tuple[int, ...], np.ndarray] = {}

    def _values(self, state: Tuple[int, ...]) -> np.ndarray:
        values = self.q.get(state)
        if values is None:
            values = np.zeros((self.n_actions,), dtype=np.float32)
            self.q[state] = values
        return values

    def act(self, state: Tuple[int, ...], epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))
        return int(np.argmax(self._values(state)))

    def greedy(self, state: Tuple[int, ...]) -> int:
        return int(np.argmax(self._values(state)))

    def update(self, state: Tuple[int, ...], action: int, reward: float, next_state: Tuple[int, ...], done: bool):
        q = self._values(state)
        target = reward if done else reward + self.gamma * float(np.max(self._values(next_state)))
        q[action] = (1.0 - self.alpha) * q[action] + self.alpha * target


def build_actions(env: CityLearnEnv, action_ids: Sequence[int]) -> List[List[float]]:
    actions: List[List[float]] = []
    for i, space in enumerate(env.action_space):
        dim = int(space.shape[0])
        act = [0.0] * dim
        if dim > 0 and i < len(action_ids):
            act[0] = float(ACTION_LEVELS[int(action_ids[i])])
        actions.append(act)
    return actions


def zero_actions(env: CityLearnEnv) -> List[List[float]]:
    return [[0.0] * int(space.shape[0]) for space in env.action_space]


def last_net_import(env: CityLearnEnv, n_buildings: int) -> np.ndarray:
    step_idx = max(int(env.time_step) - 1, 0)
    values = [float(env.buildings[i].net_electricity_consumption[step_idx]) for i in range(n_buildings)]
    return np.maximum(np.asarray(values, dtype=np.float32), 0.0)


def run_baseline(env: CityLearnEnv, n_buildings: int, seed: int, max_steps: int) -> float:
    reset_env(env, seed)
    total_import = 0.0
    for _ in range(max_steps):
        _, _, done = step_env(env, zero_actions(env))
        total_import += float(np.sum(last_net_import(env, n_buildings)))
        if done:
            break
    return total_import


def train_q_learning(
    env: CityLearnEnv,
    encoder: StateEncoder,
    n_buildings: int,
    episodes: int,
    max_steps: int,
    seed: int,
) -> Tuple[List[TabularQAgent], List[float]]:
    agents = [TabularQAgent(n_actions=len(ACTION_LEVELS)) for _ in range(n_buildings)]
    episode_imports: List[float] = []

    for ep in range(episodes):
        obs_list = reset_env(env, seed + ep)
        epsilon = 0.05 + 0.95 * np.exp(-ep / max(1.0, episodes / 3.0))
        ep_import = 0.0

        for _ in range(max_steps):
            states = [encoder.encode(obs_list[i]) for i in range(n_buildings)]
            action_ids = [agents[i].act(states[i], epsilon) for i in range(n_buildings)]
            next_obs_list, _, done = step_env(env, build_actions(env, action_ids))

            imports = last_net_import(env, n_buildings)
            rewards = -imports

            for i in range(n_buildings):
                next_state = encoder.encode(next_obs_list[i])
                agents[i].update(states[i], action_ids[i], float(rewards[i]), next_state, done)

            ep_import += float(np.sum(imports))
            obs_list = next_obs_list
            if done:
                break

        episode_imports.append(ep_import)
        print(f"train {ep + 1:03d}/{episodes}: import={ep_import:.2f}, eps={epsilon:.3f}")

    return agents, episode_imports


def evaluate_q_learning(
    env: CityLearnEnv,
    encoder: StateEncoder,
    agents: List[TabularQAgent],
    n_buildings: int,
    seed: int,
    max_steps: int,
) -> float:
    obs_list = reset_env(env, seed)
    total_import = 0.0
    for _ in range(max_steps):
        states = [encoder.encode(obs_list[i]) for i in range(n_buildings)]
        action_ids = [agents[i].greedy(states[i]) for i in range(n_buildings)]
        obs_list, _, done = step_env(env, build_actions(env, action_ids))
        total_import += float(np.sum(last_net_import(env, n_buildings)))
        if done:
            break
    return total_import


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple and robust tabular Q-learning for CityLearn")
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--buildings", type=int, default=DEFAULT_BUILDINGS)
    parser.add_argument("--bins", type=int, default=6)
    args = parser.parse_args()

    np.random.seed(args.seed)
    schema = resolve_schema(args.dataset)

    print(f"citylearn={citylearn.__version__}")
    print(f"schema={schema}")

    env_probe = make_env(schema)
    feature_names, occ_source = choose_features(env_probe)
    n_buildings = min(args.buildings, len(env_probe.buildings))
    encoder = StateEncoder(env_probe, feature_names, bins=args.bins)

    print(f"buildings={n_buildings}")
    print(f"features={feature_names}")
    if occ_source == "proxy":
        print("obsadenost: pouzity proxy signal non_shiftable_load")

    baseline_env = make_env(schema)
    baseline_import = run_baseline(baseline_env, n_buildings, args.seed, args.max_steps)
    print(f"baseline_import={baseline_import:.2f}")

    train_env = make_env(schema)
    agents, _ = train_q_learning(
        train_env,
        encoder,
        n_buildings,
        args.episodes,
        args.max_steps,
        args.seed,
    )

    eval_env = make_env(schema)
    q_import = evaluate_q_learning(
        eval_env,
        encoder,
        agents,
        n_buildings,
        args.seed,
        args.max_steps,
    )

    saving_pct = 100.0 * (baseline_import - q_import) / max(1e-6, baseline_import)

    print("\n=== Porovnanie oproti fixnej strategii ===")
    print(f"baseline import:   {baseline_import:.2f}")
    print(f"q-learning import: {q_import:.2f}")
    print(f"uspora energie:    {saving_pct:.2f}%")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
