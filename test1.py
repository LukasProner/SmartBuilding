"""Simple Q-learning starter for CityLearn.

What this script does:
1) Trains a tabular Q-learning agent (with state discretization).
2) Compares the learned policy against a fixed baseline strategy.
3) Compares multiple reward functions.
4) Estimates how fast each reward setup reaches a stable solution.

This is intentionally educational and compact, not fully optimized.
"""

import time
import argparse
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from citylearn.citylearn import CityLearnEnv

try:
    import torch
except ImportError:
    torch = None


# -----------------------------
# Experiment configuration
# -----------------------------
SCENARIO = "citylearn_challenge_2020_climate_zone_1"
TRAIN_EPISODES = 8  # Increase to 30+ for stronger results.
EVAL_EPISODES = 2
MAX_STEPS_PER_EPISODE = 2000  # Faster feedback; increase for full-year simulation.
ALPHA = 0.15
GAMMA = 0.98
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.85

# We only control the first actuator of the first building to keep tabular Q-learning feasible.
# All other actuators/buildings use 0.0 action.
DISCRETE_ACTIONS = np.array([-0.5, -0.25, 0.0, 0.25, 0.5], dtype=np.float32)

# Reward variants for comparison.
REWARD_MODES = ["default", "penalize_peak", "smooth_operation"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tabular Q-learning starter for CityLearn with optional CUDA backend."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device for Q-table operations. Default: auto.",
    )
    parser.add_argument(
        "--no-device-info",
        action="store_true",
        help="Disable startup print of selected compute device.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str = "auto", print_info: bool = True) -> str:
    cuda_available = torch is not None and torch.cuda.is_available()

    if device_arg == "cuda" and not cuda_available:
        if torch is None:
            raise RuntimeError(
                "Requested --device cuda, but PyTorch is not installed in this environment."
            )
        raise RuntimeError(
            "Requested --device cuda, but CUDA is not available. "
            "Check NVIDIA driver and CUDA-enabled PyTorch installation."
        )

    if device_arg == "cpu":
        device = "cpu"
    elif device_arg == "cuda":
        device = "cuda"
    else:
        device = "cuda" if cuda_available else "cpu"

    if print_info:
        print("=== Compute backend ===")
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0) if torch is not None else "Unknown GPU"
            print(f"Device: CUDA | GPU: {gpu_name}")
        else:
            if torch is None:
                reason = "PyTorch is not installed"
            else:
                reason = "CUDA not available"
            print(f"Device: CPU | Reason: {reason}")
        print(
            "Note: CityLearn environment simulation is mostly CPU-bound, "
            "so end-to-end speedup from CUDA may be limited."
        )
        print()

    return device


def make_env() -> CityLearnEnv:
    return CityLearnEnv(SCENARIO)


def get_obs_index_map(env: CityLearnEnv) -> Dict[str, int]:
    names = env.observation_names[0]
    return {name: i for i, name in enumerate(names)}


def get_state_features_and_bins() -> Dict[str, np.ndarray]:
    """Selected features: weather forecasts + occupancy proxy.

    Note: This dataset has no direct occupancy signal, so we use:
    - hour and day_type (behavior pattern proxy)
    - non_shiftable_load (usage proxy)
    """

    return {
        "hour": np.array([6, 12, 18]),
        "day_type": np.array([2.5, 5.5]),
        "outdoor_dry_bulb_temperature": np.array([0, 10, 20, 30]),
        "outdoor_dry_bulb_temperature_predicted_6h": np.array([0, 10, 20, 30]),
        "outdoor_relative_humidity": np.array([30, 50, 70, 90]),
        "direct_solar_irradiance_predicted_6h": np.array([100, 300, 600]),
        "non_shiftable_load": np.array([0.5, 1.5, 3.0, 5.0]),
        "electrical_storage_soc": np.array([0.2, 0.4, 0.6, 0.8]),
    }


def discretize_state(
    building_obs: Sequence[float],
    obs_index: Dict[str, int],
    feature_bins: Dict[str, np.ndarray],
) -> Tuple[int, ...]:
    bins = []

    for feature_name, thresholds in feature_bins.items():
        raw_value = float(building_obs[obs_index[feature_name]])
        # np.digitize returns which bin the value belongs to.
        bin_index = int(np.digitize(raw_value, thresholds))
        bins.append(bin_index)

    return tuple(bins)


def choose_action(q_values, epsilon: float, rng: np.random.Generator, device: str) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(q_values)))

    if device == "cuda":
        return int(torch.argmax(q_values).item())

    return int(np.argmax(q_values))


def build_actions(env: CityLearnEnv, discrete_action_idx: int) -> List[np.ndarray]:
    all_actions: List[np.ndarray] = []

    for building_id, action_space in enumerate(env.action_space):
        action = np.zeros(action_space.shape, dtype=np.float32)

        if building_id == 0:
            low = float(action_space.low[0])
            high = float(action_space.high[0])
            action[0] = np.clip(DISCRETE_ACTIONS[discrete_action_idx], low, high)

        all_actions.append(action)

    return all_actions


def fixed_baseline_actions(env: CityLearnEnv) -> List[np.ndarray]:
    return [np.zeros(space.shape, dtype=np.float32) for space in env.action_space]


def reward_transform(
    reward_list: Sequence[float],
    next_obs: Sequence[Sequence[float]],
    obs_index: Dict[str, int],
    mode: str,
) -> float:
    env_reward = float(np.sum(reward_list))
    # Buildings may expose different observation lengths. The last value is net electricity.
    total_net = float(np.sum([o[-1] for o in next_obs if len(o) > 0]))

    if mode == "default":
        return env_reward

    if mode == "penalize_peak":
        # Extra penalty for high positive net demand.
        return env_reward - 0.2 * max(total_net, 0.0)

    if mode == "smooth_operation":
        # Penalize aggressive peaks stronger to favor smoother operation.
        return env_reward - 0.05 * (max(total_net, 0.0) ** 2)

    raise ValueError(f"Unknown reward mode: {mode}")


def estimate_stabilization_episode(
    episode_returns: Sequence[float],
    window: int = 3,
    tolerance: float = 0.03,
    consecutive: int = 2,
) -> int:
    """Returns the first episode index where moving average changes become small.

    If stabilization is not detected, returns len(episode_returns).
    """

    if len(episode_returns) < (window + consecutive):
        return len(episode_returns)

    moving_avg = np.convolve(episode_returns, np.ones(window) / window, mode="valid")
    stable_count = 0

    for i in range(1, len(moving_avg)):
        prev = moving_avg[i - 1]
        curr = moving_avg[i]
        scale = max(abs(prev), 1e-6)
        relative_change = abs(curr - prev) / scale

        if relative_change < tolerance:
            stable_count += 1
            if stable_count >= consecutive:
                # +window maps moving-average index back to episode count.
                return i + window
        else:
            stable_count = 0

    return len(episode_returns)


def train_q_learning(
    reward_mode: str,
    device: str,
    seed: int = 42,
) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    env = make_env()

    obs_index = get_obs_index_map(env)
    feature_bins = get_state_features_and_bins()
    if device == "cuda":
        q_table = defaultdict(
            lambda: torch.zeros(len(DISCRETE_ACTIONS), dtype=torch.float32, device="cuda")
        )
    else:
        q_table = defaultdict(lambda: np.zeros(len(DISCRETE_ACTIONS), dtype=np.float32))

    epsilon = EPSILON_START
    training_returns = []

    start_time = time.time()

    for episode in range(TRAIN_EPISODES):
        obs = env.reset()
        state = discretize_state(obs[0], obs_index, feature_bins)
        done = False
        episode_return = 0.0
        step_count = 0

        while (not done) and (step_count < MAX_STEPS_PER_EPISODE):
            action_idx = choose_action(q_table[state], epsilon, rng, device)
            next_obs, reward, done, _ = env.step(build_actions(env, action_idx))

            shaped_reward = reward_transform(reward, next_obs, obs_index, reward_mode)
            next_state = discretize_state(next_obs[0], obs_index, feature_bins)

            if device == "cuda":
                best_next = float(torch.max(q_table[next_state]).item())
                td_target = shaped_reward + GAMMA * best_next
                q_table[state][action_idx] = q_table[state][action_idx] + ALPHA * (
                    td_target - q_table[state][action_idx]
                )
            else:
                best_next = float(np.max(q_table[next_state]))
                td_target = shaped_reward + GAMMA * best_next
                td_error = td_target - q_table[state][action_idx]
                q_table[state][action_idx] += ALPHA * td_error

            state = next_state
            episode_return += shaped_reward
            step_count += 1

        training_returns.append(episode_return)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        print(
            f"[{reward_mode}] Episode {episode + 1}/{TRAIN_EPISODES} | "
            f"Return: {episode_return:.2f} | Epsilon: {epsilon:.3f}"
        )

    train_seconds = time.time() - start_time
    stabilization_episode = estimate_stabilization_episode(training_returns)

    return {
        "q_table": q_table,
        "obs_index": obs_index,
        "feature_bins": feature_bins,
        "training_returns": training_returns,
        "train_seconds": train_seconds,
        "stabilization_episode": stabilization_episode,
    }


def evaluate_policy(
    q_table,
    obs_index: Dict[str, int],
    feature_bins: Dict[str, np.ndarray],
    device: str,
) -> Dict[str, float]:
    env = make_env()
    env_rewards = []
    total_positive_net = []

    for _ in range(EVAL_EPISODES):
        obs = env.reset()
        done = False
        episode_env_reward = 0.0
        episode_energy = 0.0
        step_count = 0

        while (not done) and (step_count < MAX_STEPS_PER_EPISODE):
            state = discretize_state(obs[0], obs_index, feature_bins)
            if device == "cuda":
                action_idx = int(torch.argmax(q_table[state]).item())
            else:
                action_idx = int(np.argmax(q_table[state]))
            obs, reward, done, _ = env.step(build_actions(env, action_idx))

            episode_env_reward += float(np.sum(reward))
            episode_energy += float(np.sum([max(o[-1], 0.0) for o in obs if len(o) > 0]))
            step_count += 1

        env_rewards.append(episode_env_reward)
        total_positive_net.append(episode_energy)

    return {
        "mean_env_reward": float(np.mean(env_rewards)),
        "mean_positive_net": float(np.mean(total_positive_net)),
    }


def evaluate_fixed_baseline(obs_index: Dict[str, int]) -> Dict[str, float]:
    env = make_env()
    env_rewards = []
    total_positive_net = []

    for _ in range(EVAL_EPISODES):
        obs = env.reset()
        done = False
        episode_env_reward = 0.0
        episode_energy = 0.0
        step_count = 0

        while (not done) and (step_count < MAX_STEPS_PER_EPISODE):
            obs, reward, done, _ = env.step(fixed_baseline_actions(env))
            episode_env_reward += float(np.sum(reward))
            episode_energy += float(np.sum([max(o[-1], 0.0) for o in obs if len(o) > 0]))
            step_count += 1

        env_rewards.append(episode_env_reward)
        total_positive_net.append(episode_energy)

    return {
        "mean_env_reward": float(np.mean(env_rewards)),
        "mean_positive_net": float(np.mean(total_positive_net)),
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device, print_info=not args.no_device_info)

    print("=== CityLearn Q-learning starter ===")
    print(f"Scenario: {SCENARIO}")
    print(f"Train episodes: {TRAIN_EPISODES} | Eval episodes: {EVAL_EPISODES}")
    print(f"Selected device for Q-table ops: {device}")
    print()

    all_results = []

    # Train and evaluate one model per reward variant.
    for mode in REWARD_MODES:
        trained = train_q_learning(mode, device=device)
        eval_metrics = evaluate_policy(
            trained["q_table"], trained["obs_index"], trained["feature_bins"], device
        )

        baseline_metrics = evaluate_fixed_baseline(trained["obs_index"])
        baseline_energy = baseline_metrics["mean_positive_net"]
        learned_energy = eval_metrics["mean_positive_net"]
        energy_savings_pct = 100.0 * (baseline_energy - learned_energy) / max(
            baseline_energy, 1e-6
        )

        all_results.append(
            {
                "reward_mode": mode,
                "train_seconds": trained["train_seconds"],
                "stabilization_episode": trained["stabilization_episode"],
                "baseline_reward": baseline_metrics["mean_env_reward"],
                "learned_reward": eval_metrics["mean_env_reward"],
                "baseline_energy": baseline_energy,
                "learned_energy": learned_energy,
                "energy_savings_pct": energy_savings_pct,
            }
        )

    print("\n=== Comparison summary ===")
    for row in all_results:
        print(f"Reward mode: {row['reward_mode']}")
        print(
            f"  Energy savings vs fixed strategy: {row['energy_savings_pct']:.2f}% "
            f"(baseline {row['baseline_energy']:.2f} -> learned {row['learned_energy']:.2f})"
        )
        print(
            f"  Env reward (mean): baseline {row['baseline_reward']:.2f}, "
            f"learned {row['learned_reward']:.2f}"
        )
        print(
            f"  Time to train: {row['train_seconds']:.2f}s | "
            f"Stabilization episode estimate: {row['stabilization_episode']}"
        )
        print()


if __name__ == "__main__":
    main()