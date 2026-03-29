"""Simple step-by-step Q-learning demo for energy optimization.

This file is intentionally simple and educational:
1) A tiny custom environment with 24-hour demand, prices, and one battery.
2) Tabular Q-learning (no neural network) to learn charge/discharge behavior.
3) Comparison against a naive baseline policy (battery idle).

Run:
    ./Scripts/python.exe test2.py
"""

import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


# -----------------------------
# 1) Environment (simple model)
# -----------------------------
class SimpleEnergyEnv:
    """One-day simulation with fixed demand and price profiles.

    State:
        - hour in day: 0..23
        - battery state of charge (SOC): 0..capacity

    Actions:
        - 0: discharge battery
        - 1: idle
        - 2: charge battery
    """

    def __init__(self):
        self.hours = 24
        self.capacity_kwh = 10.0
        self.max_power_kwh = 2.0
        self.charge_eff = 0.95
        self.discharge_eff = 0.95

        # A small synthetic demand profile [kWh/h].
        self.demand_profile = np.array(
            [
                1.2,
                1.1,
                1.0,
                1.0,
                1.1,
                1.4,
                1.8,
                2.2,
                2.5,
                2.2,
                2.0,
                1.9,
                1.8,
                1.9,
                2.1,
                2.4,
                2.8,
                3.1,
                3.4,
                3.0,
                2.5,
                2.1,
                1.8,
                1.5,
            ],
            dtype=np.float32,
        )

        # Time-of-use price profile [EUR/kWh].
        self.price_profile = np.array(
            [
                0.10,
                0.10,
                0.10,
                0.10,
                0.10,
                0.12,
                0.15,
                0.18,
                0.20,
                0.22,
                0.20,
                0.18,
                0.16,
                0.16,
                0.18,
                0.22,
                0.28,
                0.35,
                0.40,
                0.32,
                0.24,
                0.18,
                0.14,
                0.12,
            ],
            dtype=np.float32,
        )

        self.reset()

    def reset(self):
        self.hour = 0
        self.soc = 0.5 * self.capacity_kwh
        return self._state()

    def _state(self):
        return self.hour, self.soc

    def step(self, action_idx):
        demand = float(self.demand_profile[self.hour])
        price = float(self.price_profile[self.hour])

        # Action mapping to battery power request.
        if action_idx == 0:
            requested_discharge = self.max_power_kwh
            available_discharge = min(
                requested_discharge, self.soc * self.discharge_eff
            )
            discharge = available_discharge
            charge = 0.0
        elif action_idx == 2:
            requested_charge = self.max_power_kwh
            free_space = self.capacity_kwh - self.soc
            charge = min(requested_charge, free_space / self.charge_eff)
            discharge = 0.0
        else:
            charge = 0.0
            discharge = 0.0

        # Battery dynamics.
        self.soc = self.soc + charge * self.charge_eff - discharge / self.discharge_eff
        self.soc = float(np.clip(self.soc, 0.0, self.capacity_kwh))

        # Grid import: demand + battery charging - battery discharging.
        grid_import = max(0.0, demand + charge - discharge)
        cost = grid_import * price
        reward = -cost

        self.hour += 1
        done = self.hour >= self.hours

        next_state = (self.hour if not done else self.hours - 1, self.soc)
        info = {
            "cost": cost,
            "grid_import": grid_import,
            "demand": demand,
            "price": price,
            "soc": self.soc,
        }

        return next_state, reward, done, info


# --------------------------------
# 2) Q-learning helper functions
# --------------------------------
def resolve_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Device: CUDA | GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("Device: CPU | CUDA not available")

    return device


def soc_to_bin(soc, capacity_kwh, num_bins):
    ratio = soc / max(capacity_kwh, 1e-6)
    idx = int(np.clip(ratio * (num_bins - 1), 0, num_bins - 1))
    return idx


def choose_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        return int(np.random.randint(0, q_values.shape[0]))

    return int(torch.argmax(q_values).item())


def run_baseline_episode(env):
    state = env.reset()
    done = False
    total_cost = 0.0

    while not done:
        # Baseline: battery always idle.
        _, _, done, info = env.step(1)
        total_cost += info["cost"]

    return total_cost


def run_greedy_episode(env, q_table, num_soc_bins):
    hour, soc = env.reset()
    done = False
    total_cost = 0.0

    while not done:
        soc_bin = soc_to_bin(soc, env.capacity_kwh, num_soc_bins)
        action = int(torch.argmax(q_table[hour, soc_bin]).item())

        (hour, soc), _, done, info = env.step(action)
        total_cost += info["cost"]

    return total_cost


# -----------------------------
# 3) Training configuration
# -----------------------------
NUM_EPISODES = 800
NUM_SOC_BINS = 20
NUM_ACTIONS = 3
ALPHA = 0.1
GAMMA = 0.98
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
SEED = 42


# -----------------------------
# 4) Train tabular Q-learning
# -----------------------------
def main():
    start_time = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = resolve_device()

    env = SimpleEnergyEnv()

    # Q table shape: [hour, soc_bin, action]
    q_table = torch.zeros((24, NUM_SOC_BINS, NUM_ACTIONS), device=device)

    epsilon = EPSILON_START
    episode_returns = []

    print(f"Training episodes: {NUM_EPISODES}")

    for episode in tqdm(range(NUM_EPISODES), desc="Training", ncols=100):
        hour, soc = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            soc_bin = soc_to_bin(soc, env.capacity_kwh, NUM_SOC_BINS)
            action = choose_action(q_table[hour, soc_bin], epsilon)

            (next_hour, next_soc), reward, done, _ = env.step(action)

            next_soc_bin = soc_to_bin(next_soc, env.capacity_kwh, NUM_SOC_BINS)

            if done:
                td_target = reward
            else:
                best_next = torch.max(q_table[next_hour, next_soc_bin]).item()
                td_target = reward + GAMMA * best_next
            td_error = td_target - q_table[hour, soc_bin, action].item()

            q_table[hour, soc_bin, action] += ALPHA * td_error

            hour, soc = next_hour, next_soc
            episode_return += reward

        episode_returns.append(episode_return)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if (episode + 1) % 50 == 0:
            print(
                f"Episode {episode + 1:3d}/{NUM_EPISODES} | "
                f"Return: {episode_return:.3f} | Epsilon: {epsilon:.3f}"
            )

    # -----------------------------
    # 5) Evaluate vs baseline
    # -----------------------------
    baseline_cost = run_baseline_episode(SimpleEnergyEnv())
    learned_cost = run_greedy_episode(SimpleEnergyEnv(), q_table, NUM_SOC_BINS)

    savings_pct = 100.0 * (baseline_cost - learned_cost) / max(baseline_cost, 1e-6)

    elapsed = timedelta(seconds=int(time.time() - start_time))

    print("\n=== Results ===")
    print(f"Baseline cost (idle battery): {baseline_cost:.3f} EUR/day")
    print(f"Learned policy cost:          {learned_cost:.3f} EUR/day")
    print(f"Savings:                      {savings_pct:.2f}%")
    print(f"Training time:                {elapsed}")

    # -----------------------------
    # 6) Simple plot
    # -----------------------------
    returns_np = np.array(episode_returns, dtype=np.float32)
    window = 20

    if len(returns_np) >= window:
        kernel = np.ones(window, dtype=np.float32) / window
        moving_avg = np.convolve(returns_np, kernel, mode="valid")
        moving_x = np.arange(window - 1, len(returns_np))
    else:
        moving_avg = returns_np
        moving_x = np.arange(len(returns_np))

    plt.figure(figsize=(10, 4))
    plt.plot(returns_np, alpha=0.35, label="Episode return")
    plt.plot(moving_x, moving_avg, linewidth=2, label=f"Moving avg ({window})")
    plt.title("Q-learning training progress")
    plt.xlabel("Episode")
    plt.ylabel("Return (negative cost)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
