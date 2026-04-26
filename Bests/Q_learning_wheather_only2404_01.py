from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product as iterproduct
from pathlib import Path

import numpy as np
import pandas as pd

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_3_3' / 'schema.json'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning_wheather_only2404_01'

TRAIN_BUILDINGS = ['Building_1', 'Building_2']
EVAL_BUILDINGS = ['Building_3']

ACTIVE_OBSERVATIONS = [
    'outdoor_dry_bulb_temperature_predicted_1',
    'outdoor_dry_bulb_temperature_predicted_2',
    'diffuse_solar_irradiance_predicted_1',
    'diffuse_solar_irradiance_predicted_2',
    'direct_solar_irradiance_predicted_1',
    'direct_solar_irradiance_predicted_2',
    'occupant_count',
]

ACTIVE_ACTIONS = ['cooling_device']

OBS_BINS = {
    'temp_pred_1': 3,
    'temp_trend_12': 3,
    'solar_pred_1_total': 3,
    'solar_trend_12_total': 3,
    'occupancy_present': 2,
}

ACTION_BINS = [3]


class WeatherOccupancyReward(RewardFunction):
    def __init__(self, env_metadata, occupancy_weight: float = 1.5, hot_weather_weight: float = 0.05, **kwargs):
        super().__init__(env_metadata, **kwargs)
        self.occupancy_weight = float(occupancy_weight)
        self.hot_weather_weight = float(hot_weather_weight)

    def calculate(self, observations: list[dict]) -> list[float]:
        reward_list: list[float] = []

        for obs in observations:
            grid_import = max(obs['net_electricity_consumption'], 0.0)
            occ_present = 1.0 if max(obs.get('occupant_count', 0.0), 0.0) > 0.0 else 0.0
            mean_forecast = float(np.mean([
                obs.get('outdoor_dry_bulb_temperature_predicted_1', 0.0),
                obs.get('outdoor_dry_bulb_temperature_predicted_2', 0.0),
                obs.get('outdoor_dry_bulb_temperature_predicted_3', 0.0),
            ]))
            occ_factor = 1.0 + self.occupancy_weight * occ_present
            weather_factor = 1.0 + self.hot_weather_weight * max(mean_forecast - 24.0, 0.0)
            reward_list.append(-(grid_import * occ_factor * weather_factor))

        return [sum(reward_list)] if self.central_agent else reward_list


class ObservationDiscretizer:
    #
    def __init__(self, env: CityLearnEnv):
        self.observation_names = env.observation_names[0]#zoberiem mena
        self.index = {name: i for i, name in enumerate(self.observation_names)}#priradim indexy
        self.feature_names = list(OBS_BINS.keys())
        self.bin_counts = [int(OBS_BINS[name]) for name in self.feature_names]
        self.state_shape = tuple(self.bin_counts)
        self.state_count = int(np.prod(self.state_shape))#vynasoby vsetky dokopy

        low_by_name = {n: float(v) for n, v in zip(self.observation_names, env.observation_space[0].low)}
        high_by_name = {n: float(v) for n, v in zip(self.observation_names, env.observation_space[0].high)}

        t1_lo, t1_hi = low_by_name['outdoor_dry_bulb_temperature_predicted_1'], high_by_name['outdoor_dry_bulb_temperature_predicted_1']
        t2_lo, t2_hi = low_by_name['outdoor_dry_bulb_temperature_predicted_2'], high_by_name['outdoor_dry_bulb_temperature_predicted_2']
        d1_lo, d1_hi = low_by_name['diffuse_solar_irradiance_predicted_1'], high_by_name['diffuse_solar_irradiance_predicted_1']
        d2_lo, d2_hi = low_by_name['diffuse_solar_irradiance_predicted_2'], high_by_name['diffuse_solar_irradiance_predicted_2']
        r1_lo, r1_hi = low_by_name['direct_solar_irradiance_predicted_1'], high_by_name['direct_solar_irradiance_predicted_1']
        r2_lo, r2_hi = low_by_name['direct_solar_irradiance_predicted_2'], high_by_name['direct_solar_irradiance_predicted_2']

        s1_lo, s1_hi = d1_lo + r1_lo, d1_hi + r1_hi
        s2_lo, s2_hi = d2_lo + r2_lo, d2_hi + r2_hi

        feature_lows = [t1_lo, t2_lo - t1_hi, s1_lo, s2_lo - s1_hi, 0.0]
        feature_highs = [t1_hi, t2_hi - t1_lo, s1_hi, s2_hi - s1_lo, 1.0]

        self.edges = [
            np.linspace(float(low), float(high), count + 1)[1:-1]#[1:-1] odstrani prvy a posledny
            for low, high, count in zip(feature_lows, feature_highs, self.bin_counts)
        ]
#
    def encode(self, observation: list[float]) -> int:
        idx = self.index

        t1 = float(observation[idx['outdoor_dry_bulb_temperature_predicted_1']])
        t2 = float(observation[idx['outdoor_dry_bulb_temperature_predicted_2']])
        d1 = float(observation[idx['diffuse_solar_irradiance_predicted_1']])
        d2 = float(observation[idx['diffuse_solar_irradiance_predicted_2']])
        r1 = float(observation[idx['direct_solar_irradiance_predicted_1']])
        r2 = float(observation[idx['direct_solar_irradiance_predicted_2']])
        occ = 1.0 if float(observation[idx['occupant_count']]) > 0.0 else 0.0

        features = [t1, t2 - t1, d1 + r1, (d2 + r2) - (d1 + r1), occ]
        bins = [int(np.digitize(v, e, right=False)) for v, e in zip(features, self.edges)]#do ktorého intervalu patrí
        return int(np.ravel_multi_index(tuple(bins), self.state_shape))


class ActionDiscretizer:
    #
    def __init__(self, env: CityLearnEnv):
        n_dims = env.action_space[0].shape[0]
        if len(ACTION_BINS) != n_dims:
            raise ValueError(f'Expected {n_dims} action bins, got {len(ACTION_BINS)}.')

        lows = env.action_space[0].low.tolist()
        highs = env.action_space[0].high.tolist()
        grids = [
            np.linspace(float(low), float(high), int(count), dtype=float)
            for low, high, count in zip(lows, highs, ACTION_BINS)
        ]
        self.joint_actions = list(iterproduct(*grids))

    @property
    def action_count(self) -> int:
        return len(self.joint_actions)

    def decode(self, action_index: int) -> list[float]:
        return list(self.joint_actions[action_index])


class TabularQLearning:
    #
    def __init__(
        self,
        env: CityLearnEnv,
        learning_rate: float,
        discount_factor: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        random_seed: int,
    ):
        self.obs_disc = ObservationDiscretizer(env)
        self.act_disc = ActionDiscretizer(env)

        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.episode_index = 0

        self.random_state = np.random.RandomState(random_seed)
        self.q_table = np.zeros((self.obs_disc.state_count, self.act_disc.action_count), dtype=np.float32)

        self._last_states: list[int] = []
        self._last_actions: list[int] = []
#
    def reset(self) -> None:
        self._last_states = []
        self._last_actions = []

    def act_train(self, observations: list[list[float]]) -> list[list[float]]:
        self._last_states = []
        self._last_actions = []
        actions: list[list[float]] = []

        for observation in observations:
            state = self.obs_disc.encode(observation)
            if self.random_state.rand() < self.epsilon:
                action_index = int(self.random_state.randint(self.act_disc.action_count))
            else:
                action_index = int(np.argmax(self.q_table[state]))

            self._last_states.append(state)
            self._last_actions.append(action_index)
            actions.append(self.act_disc.decode(action_index))

        return actions

    def act_eval(self, observations: list[list[float]]) -> list[list[float]]:
        actions: list[list[float]] = []

        for observation in observations:
            state = self.obs_disc.encode(observation)
            action_index = int(np.argmax(self.q_table[state]))
            actions.append(self.act_disc.decode(action_index))

        return actions
#POZRIET ESTE RAZ
    def update(self, rewards: list[float], next_observations: list[list[float]], terminated: bool) -> None:
        for state, action, reward, next_obs in zip(self._last_states, self._last_actions, rewards, next_observations):
            next_state = self.obs_disc.encode(next_obs)
            best_next = 0.0 if terminated else float(np.max(self.q_table[next_state]))
            td_target = float(reward) + self.discount_factor * best_next
            self.q_table[state, action] += self.learning_rate * (td_target - float(self.q_table[state, action]))

    def finish_episode(self) -> None:
        self.episode_index += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-self.epsilon_decay))

# 
class FixedPolicy:
    def __init__(self, cooling_action: float):
        self.cooling_action = float(cooling_action)

    # ODSTRANIT
    def reset(self) -> None:
        pass

    def act(self, observations: list[list[float]]) -> list[list[float]]:
        return [[self.cooling_action] for _ in observations]


@dataclass
class RunOutput:
    total_grid_import_kwh: float
    total_net_consumption_kwh: float
    discomfort_proportion: float
    cumulative_reward: float
    trajectory: pd.DataFrame
    kpis: pd.DataFrame

# 
def make_env(schema_path: Path, building_names: list[str], random_seed: int) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=False,
        buildings=building_names,
        active_observations=ACTIVE_OBSERVATIONS,
        active_actions=ACTIVE_ACTIONS,
        reward_function=WeatherOccupancyReward,
        random_seed=random_seed,
    )


def train(agent: TabularQLearning, env: CityLearnEnv, episodes: int) -> list[float]:
    rewards_per_episode: list[float] = []

    for _ in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        terminated = False
        episode_reward = 0.0

        while not terminated:
            actions = agent.act_train(observations)
            next_obs, rewards, terminated, _, _ = env.step(actions)
            agent.update(rewards, next_obs, terminated)
            observations = next_obs
            episode_reward += float(np.sum(rewards))

        agent.finish_episode()
        rewards_per_episode.append(episode_reward)

    return rewards_per_episode

# 
def evaluate_policy(env: CityLearnEnv, act_fn) -> RunOutput:
    observations, _ = env.reset()
    terminated = False
    cumulative_reward = 0.0
    reward_trace: list[float] = []

    while not terminated:
        actions = act_fn(observations)
        observations, rewards, terminated, _, _ = env.step(actions)
        step_reward = float(np.sum(rewards))
        reward_trace.append(step_reward)
        cumulative_reward += step_reward

    base_env = env.unwrapped
    buildings = base_env.buildings
    kpis = base_env.evaluate()

    building_names = [b.name for b in buildings]
    discomfort_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_proportion')]
    discomfort = float(discomfort_rows['value'].mean()) if not discomfort_rows.empty else 0.0

    # je tam kvôli porovnaniu kvality politík (energia vs. komfort), či agent síce šetrí energiu, ale nekazí komfort

    aggregate_net = np.zeros(len(reward_trace), dtype=float)
    trajectory_data: dict[str, np.ndarray | list[float]] = {
        'time_step': np.arange(len(reward_trace)),
        'reward': reward_trace,
        'cumulative_reward': np.cumsum(reward_trace),
    }

    for building in buildings:
        net = np.asarray(building.net_electricity_consumption, dtype=float)[:len(reward_trace)]
        aggregate_net += net
        trajectory_data[f'grid_import_{building.name}_kwh'] = np.clip(net, 0.0, None)

# všetko < 0 → nastav na 0
# všetko ≥ 0 → nechaj
# berie len odber zo siete (import)
# ignoruje export (napr. solár)

    trajectory_data['grid_import_kwh'] = np.clip(aggregate_net, 0.0, None)
    trajectory_data['cumulative_grid_import_kwh'] = np.cumsum(np.clip(aggregate_net, 0.0, None))

    return RunOutput(
        total_grid_import_kwh=float(np.sum(np.clip(aggregate_net, 0.0, None))),
        total_net_consumption_kwh=float(np.sum(aggregate_net)),
        discomfort_proportion=discomfort,
        cumulative_reward=cumulative_reward,
        trajectory=pd.DataFrame(trajectory_data),
        kpis=kpis,
    )


def run_experiment(
    schema_path: Path,
    episodes: int,
    baseline_cooling: float,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_env = make_env(schema_path, EVAL_BUILDINGS, seed)
    fixed_policy = FixedPolicy(cooling_action=baseline_cooling)
    fixed_run = evaluate_policy(fixed_env, fixed_policy.act)

    train_env = make_env(schema_path, TRAIN_BUILDINGS, seed)
    agent = TabularQLearning(
        train_env,
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.03,
        random_seed=seed,
    )
    episode_rewards = train(agent, train_env, episodes)

    eval_env = make_env(schema_path, EVAL_BUILDINGS, seed)
    learned_run = evaluate_policy(eval_env, agent.act_eval)

    savings_vs_fixed_pct = None
    if fixed_run.total_grid_import_kwh > 0.0:
        savings_vs_fixed_pct = 100.0 * (fixed_run.total_grid_import_kwh - learned_run.total_grid_import_kwh) / fixed_run.total_grid_import_kwh

    summary = pd.DataFrame([
        {
            'policy': f'Fixed(cool={baseline_cooling:.2f})',
            'seed': seed,
            'grid_import_kwh': round(fixed_run.total_grid_import_kwh, 3),
            'net_consumption_kwh': round(fixed_run.total_net_consumption_kwh, 3),
            'discomfort_proportion': round(fixed_run.discomfort_proportion, 4),
            'cumulative_reward': round(fixed_run.cumulative_reward, 3),
            'savings_vs_fixed_pct': 0.0,
        },
        {
            'policy': 'Q-learning (weather)',
            'seed': seed,
            'grid_import_kwh': round(learned_run.total_grid_import_kwh, 3),
            'net_consumption_kwh': round(learned_run.total_net_consumption_kwh, 3),
            'discomfort_proportion': round(learned_run.discomfort_proportion, 4),
            'cumulative_reward': round(learned_run.cumulative_reward, 3),
            'savings_vs_fixed_pct': None if savings_vs_fixed_pct is None else round(savings_vs_fixed_pct, 3),
            'last_10_episode_reward_mean': round(float(np.mean(episode_rewards[-10:])), 3) if episode_rewards else None,
        },
    ])

    summary.to_csv(output_dir / 'summary_results.csv', index=False)
    fixed_run.trajectory.to_csv(output_dir / 'trajectory_fixed.csv', index=False)
    learned_run.trajectory.to_csv(output_dir / 'trajectory_q_learning.csv', index=False)
    fixed_run.kpis.to_csv(output_dir / 'kpis_fixed.csv', index=False)
    learned_run.kpis.to_csv(output_dir / 'kpis_q_learning.csv', index=False)

    pd.DataFrame({
        'episode': np.arange(1, len(episode_rewards) + 1),
        'episode_reward': episode_rewards,
    }).to_csv(output_dir / 'learning_trace.csv', index=False)

    np.save(output_dir / 'q_table.npy', agent.q_table)

    return summary

# 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simple weather+occupancy Q-learning experiment in CityLearn.')
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument('--episodes', type=int, default=400)
    parser.add_argument('--baseline-cooling', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_experiment(
        schema_path=args.schema,
        episodes=args.episodes,
        baseline_cooling=args.baseline_cooling,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
