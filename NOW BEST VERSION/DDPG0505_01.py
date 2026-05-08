from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction

from Q_learning0505 import (
    ACTIVE_ACTIONS,
    ACTIVE_OBSERVATIONS,
    CombinedMultiObjectiveReward,
    ComfortAwareReward,
    GridImportOnlyReward,
    SmartTouReward,
    PeakShavingReward,
    PricingAwareReward,
    RampingPenaltyReward,
    REWARD_CONFIGS,
    SelfSufficiencyReward,
    SolarAlignmentReward,
    StorageManagementReward,
    TimeOfUseReward,
    WeatherOccupancyReward,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1' / 'schema.json'
DEFAULT_BUILDINGS = ['Building_1', 'Building_2', 'Building_3']
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_ddpg0505_01'
DEFAULT_EPISODES = 40
DEFAULT_BASELINE_COOLING = 0.5
DEFAULT_RANDOM_SEED = 7
DEFAULT_COMPARISON_HORIZON = 719

#
class DecentralFixedPolicy:
    def __init__(self, cooling_action: float = 0.25):
        self.cooling_action = float(cooling_action)
        self.base_pattern = np.array([0.0, 0.0, self.cooling_action], dtype=np.float32)

    def reset(self) -> None:
        pass

    def predict(self, observations: list[list[float]], deterministic: bool = True) -> list[list[float]]:
        return [self.base_pattern.astype(float).tolist() for _ in observations]


class Buffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = action
        self.reward_buffer[self.position] = reward
        self.next_state_buffer[self.position] = next_state
        self.done_buffer[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.as_tensor(self.state_buffer[idx], dtype=torch.float32, device=device)
        actions = torch.as_tensor(self.action_buffer[idx], dtype=torch.float32, device=device)
        rewards = torch.as_tensor(self.reward_buffer[idx], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(self.next_state_buffer[idx], dtype=torch.float32, device=device)
        dones = torch.as_tensor(self.done_buffer[idx], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))

#modely fungujú lepšie, keď vstupy:
# nemajú extrémne veľké hodnoty
# sú približne v rovnakom rozsahu
class ObsNormalizer:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.scale = np.maximum(self.high - self.low, 0.000001)
    #nastavi hodnotu v rozmedzi -1 1
    def normalize(self, observation: Sequence[float]) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        n = 2.0 * (obs - self.low) / self.scale - 1.0
        return np.clip(n, -5.0, 5.0).astype(np.float32)


class DDPGAgent:
    def __init__(
        self,
        env: CityLearnEnv,
        learning_rate_actor: float = 0.0001,
        learning_rate_critic: float = 0.001,
        gamma: float = 0.95,
        tau: float = 0.005,
        batch_size: int = 128,
        replay_size: int = 100000,
        warmup_steps: int = 1000,#nepotrebna hodnota - nahrad cislom - viacero ich je pri refaktorizacii zmenit
        # update_every: int = 1,
        policy_noise: float = 0.20,
        noise_decay: float = 0.995,
        min_noise: float = 0.03,
        hidden_dim: int = 256,
        random_seed: int = 7,
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.random_state = np.random.RandomState(random_seed)
        torch.manual_seed(random_seed)
        #nastaví seed pre PyTorch
        #ovplyvňuje inicializáciu váh sietí a ďalšie náhodné operácie v Torch

        self.state_dim = int(env.observation_space[0].shape[0])#pocet stavov na vstupe
        self.action_dim = int(env.action_space[0].shape[0])
        observation_low = np.asarray(env.observation_space[0].low, dtype=np.float32)
        observation_high = np.asarray(env.observation_space[0].high, dtype=np.float32)
        self.action_low = np.asarray(env.action_space[0].low, dtype=np.float32)
        self.action_high = np.asarray(env.action_space[0].high, dtype=np.float32)#dolne a horne hranice pre oba zoznamy

        #lineárne škálovanie
        self.action_x = (self.action_high - self.action_low) / 2
        self.action_y = (self.action_high + self.action_low) / 2
        
        self.normalizer = ObsNormalizer(observation_low, observation_high)
        self.action_low_tensor = torch.as_tensor(self.action_low, dtype=torch.float32, device=self.device)
        self.action_high_tensor = torch.as_tensor(self.action_high, dtype=torch.float32, device=self.device)
        self.action_x_tensor = torch.as_tensor(self.action_x, dtype=torch.float32, device=self.device)
        self.action_y_tensor = torch.as_tensor(self.action_y, dtype=torch.float32, device=self.device)

        self.actor = Actor(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        #vytvorí presnú kópiu siete:
# rovnaká architektúra
# rovnaké váhy
# ale nezávislá inštancia
        self.critic = Critic(self.state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_actor)#aktualizuje váhy Actor neurónky
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate_critic)

        self.gamma = float(gamma)
        self.tau = float(tau)#kontroluje, ako rýchlo sa kopírujú váhy do target sietí
        self.batch_size = int(batch_size)
        self.warmup_steps = int(warmup_steps)
        # self.update_every = int(update_every)
        self.noise_decay = float(noise_decay)
        self.min_noise = float(min_noise)
        self.current_noise = float(policy_noise)
        self.buffer = Buffer(replay_size, self.state_dim, self.action_dim)
        self.total_steps = 0
#
    def reset(self) -> None:
        pass
#
    def normalize_state(self, observation: Sequence[float]) -> np.ndarray:
        return self.normalizer.normalize(observation)
#prevedie normalizovanú akciu z Actora na reálne hodnoty v prostredí
# np.clip([-2, 0, 3, 10], 0, 5) -> [0, 0, 3, 5]
    def scale_action(self, raw_action: np.ndarray) -> np.ndarray:
        return np.clip(self.action_y + raw_action * self.action_x, self.action_low, self.action_high)

    def predict(self, observations: list[list[float]], deterministic: bool = False) -> list[list[float]]:
        actions: list[list[float]] = []

        for observation in observations:
            state = self.normalize_state(observation)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

# Neurónky v PyTorch očakávajú batch:
# bez unsqueeze	s unsqueeze
# [state_dim]	[1, state_dim]

            with torch.no_grad():
                raw_action = self.actor(state_tensor).cpu().numpy()[0]
# Actor dostane stav (state_tensor) a prejde ho neurónovou sieťou bez učenia (no_grad).
# Sieť vypočíta akciu na základe naučených váh.
# Výstup je v rozsahu [-1, 1] vďaka Tanh a reprezentuje navrhnutú akciu pre daný stav.
            if not deterministic:
                raw_action = raw_action + self.random_state.normal(0.0, self.current_noise, size=self.action_dim)
                raw_action = np.clip(raw_action, -1.0, 1.0)

            action = self.scale_action(raw_action.astype(np.float32))
            actions.append(action.tolist())

        return actions
#
    def remember(
        self,
        observation: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_observation: Sequence[float],
        done: bool,
    ) -> None:
        state = self.normalize_state(observation)
        next_state = self.normalize_state(next_observation)
        self.buffer.push(
            state=state,
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward),
            next_state=next_state,
            done=done,
        )
        self.total_steps += 1

    def update(self) -> tuple[float | None, float | None]:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None, None
        # if self.total_steps % self.update_every != 0: #ODSTRANIT
        #     return None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size, self.device)

        with torch.no_grad():
            next_actions = self._scale_action_tensor(self.actor_target(next_states))
            target_q = self.critic_target(next_states, next_actions) #aká dobrá je táto budúca akcia
            q_target = rewards + (1.0 - dones) * self.gamma * target_q
#odmena teraz + odhad budúcej odmeny
        q_current = self.critic(states, actions)#koľko si myslím, že táto akcia stojí
        critic_loss = F.mse_loss(q_current, q_target)#ako veľmi sa líšim od správnej odpovede
        self.critic_optimizer.zero_grad()#vymaze gradienty
        critic_loss.backward()#ako mám zmeniť váhy, aby som bol presnejší
        self.critic_optimizer.step()#upraví váhy neurónky

        actor_actions = self._scale_action_tensor(self.actor(states))
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()#ako zmeniť Actor, aby robil lepšie akcie
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return float(actor_loss.item()), float(critic_loss.item())
    # DDPG update krok (logika):
#
# 1. Z replay bufferu sa náhodne vyberie batch skúseností (state, action, reward, next_state, done)
#
# 2. VÝPOČET TARGETU (bez učenia):
#    - použije sa actor_target na výpočet akcie v next_state
#    - použije sa critic_target na ohodnotenie tejto budúcej akcie
#    - target = reward + gamma * Q(next_state, actor_target(next_state))
#    - ak done = True, budúcnosť sa ignoruje
#
#    -> toto je len výpočet "správnej odpovede" pre Critic (NEUČÍ SA TU NIČ)
#
# 3. CRITIC UČENIE:
#    - q_current = critic(state, action)
#    - critic sa učí minimalizovať rozdiel medzi q_current a q_target
#    - teda učí sa presne odhadovať hodnotu akcie
#
# 4. ACTOR UČENIE:
#    - actor navrhne akcie pre states
#    - critic ich ohodnotí
#    - actor sa učí maximalizovať toto ohodnotenie (vyššie Q = lepšie akcie)
#
# 5. TARGET SIEŤ:
#    - actor_target a critic_target sa pomaly kopírujú z aktuálnych sietí
#    - zabezpečujú stabilné učenie (aby sa targety nehýbali príliš rýchlo)
#
# => výsledok:
#    critic sa učí hodnotiť akcie
#    actor sa učí robiť lepšie akcie
#    target siete stabilizujú celý proces





    def finish_episode(self) -> None:
        self.current_noise = max(self.min_noise, self.current_noise * self.noise_decay)
    #scale action ale pre ucenie clamp je funkcia v PyTorch, ktorá obmedzí hodnoty na daný rozsah
    def _scale_action_tensor(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            self.action_y_tensor + raw_action * self.action_x_tensor,
            min=self.action_low_tensor,
            max=self.action_high_tensor,
        )
    #target sieť sa pomaly približuje k source sieti
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


@dataclass
class ExperimentResult:
    policy: str
    total_grid_import_kwh: float
    total_export_kwh: float
    total_net_consumption_kwh: float #čistá spotreba
    discomfort_proportion: float
    discomfort_cold_proportion: float
    discomfort_hot_proportion: float
    all_rewards: float
    cost_total_ratio: float | None = None
    carbon_emissions_total_ratio: float | None = None
    daily_peak_average_ratio: float | None = None
    ramping_average_ratio: float | None = None
    savings_vs_fixed_pct: float | None = None
    train_sec: float | None = None
    stability_episode: int | None = None
    last_10_episode_reward_mean: float | None = None


@dataclass
class PolicyRun:
    result: ExperimentResult
    trajectory: pd.DataFrame
    kpis: pd.DataFrame


@dataclass
class TrainingTrace:
    episode_rewards: list[float]
    actor_losses: list[float]
    critic_losses: list[float]
    exploration_noise: list[float]
    train_sec: float
    stability_episode: int | None

#vytvorí a vráti prostredie pre simuláciu v CityLearn
def make_env(schema_path: Path, building_names: list[str], random_seed: int, reward_function: type[RewardFunction] = WeatherOccupancyReward) -> CityLearnEnv:
    return CityLearnEnv(
        str(schema_path),
        central_agent=False,
        buildings=building_names,
        active_observations=ACTIVE_OBSERVATIONS,
        active_actions=ACTIVE_ACTIONS,
        reward_function=reward_function,
        random_seed=random_seed,
    )


def estimate_stability_episode(rewards: list[float], window: int = 10, tolerance: float = 0.03) -> int | None:
    if len(rewards) < window * 2:
        return None

    arr = np.asarray(rewards, dtype=float)
    for index in range((window * 2) - 1, len(arr)):
        prev = arr[index - (2 * window) + 1:index - window + 1]
        curr = arr[index - window + 1:index + 1]
        prev_mean = float(np.mean(prev))
        curr_mean = float(np.mean(curr))
        scale = max(1.0, abs(prev_mean))
        stable_mean = abs(curr_mean - prev_mean) / scale <= tolerance
        stable_std = float(np.std(curr)) / max(1.0, abs(curr_mean)) <= tolerance * 2
        if stable_mean and stable_std:
            return index + 1

    return None


def train_ddpg(agent: DDPGAgent, env: CityLearnEnv, episodes: int) -> TrainingTrace:
    episode_rewards: list[float] = [] #celková odmena za každú epizódu
    actor_losses: list[float] = []#priemerná chyba (loss) Actor siete za epizódu
    critic_losses: list[float] = []#chyba Critic siete
    exploration_noise: list[float] = []#veľkosť náhodného šumu
    start_time = time.perf_counter()#uloží čas začiatku tréningu
    progress_every = 30

    for episode in range(episodes):
        observations, _ = env.reset()
        agent.reset()
        terminated = False
        episode_reward = 0.0
        ep_actor_losses: list[float] = []
        ep_critic_losses: list[float] = []

        while not terminated:
            actions = agent.predict(observations, deterministic=False)
            next_observations, rewards, terminated, _, _ = env.step(actions)

            for observation, action, reward, next_observation in zip(observations, actions, rewards, next_observations):
                agent.remember(observation, action, float(reward), next_observation, terminated)

            actor_loss, critic_loss = agent.update()
            if actor_loss is not None and critic_loss is not None:
                ep_actor_losses.append(actor_loss)
                ep_critic_losses.append(critic_loss)
            observations = next_observations
            episode_reward += float(np.sum(rewards))

        agent.finish_episode()
        episode_rewards.append(episode_reward)
        actor_losses.append(float(np.mean(ep_actor_losses)) if ep_actor_losses else np.nan)
        critic_losses.append(float(np.mean(ep_critic_losses)) if ep_critic_losses else np.nan)
        exploration_noise.append(agent.current_noise)

        current_episode = episode + 1
        if current_episode == 1 or current_episode % progress_every == 0 or current_episode == episodes:
            rolling_reward = float(np.mean(episode_rewards[-10:])) if episode_rewards else episode_reward
            elapsed = time.perf_counter() - start_time
            print(
                f'    Episode {current_episode}/{episodes} | reward={episode_reward:.2f} '
                f'| rolling10={rolling_reward:.2f} | noise={agent.current_noise:.3f} | elapsed={elapsed:.1f}s',
                flush=True,
            )

    return TrainingTrace(
        episode_rewards=episode_rewards,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        exploration_noise=exploration_noise,
        train_sec=time.perf_counter() - start_time,
        stability_episode=estimate_stability_episode(episode_rewards),
    )

def eval_agent(env: CityLearnEnv, agent) -> PolicyRun:
    obs, _ = env.reset()
    agent.reset()
    terminated = False
    rew_list: list[float] = []
    all_rewards = 0.0

    while not terminated:
        act = agent.predict(obs, deterministic=True)
        obs, rewrds, terminated, _, _ = env.step(act)
        step_reward = float(np.sum(rewrds))
        rew_list.append(step_reward)
        all_rewards += step_reward

    base_env = env.unwrapped
    buildings = base_env.buildings
    kpis = base_env.evaluate()
    building_names = [building.name for building in buildings]
    discomfort_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_proportion')]
    discomfort_cold_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_cold_proportion')]
    discomfort_hot_rows = kpis[(kpis['name'].isin(building_names)) & (kpis['cost_function'] == 'discomfort_hot_proportion')]
    discomfort = float(discomfort_rows['value'].mean()) if not discomfort_rows.empty else 0.0
    discomfort_cold = float(discomfort_cold_rows['value'].mean()) if not discomfort_cold_rows.empty else 0.0
    discomfort_hot = float(discomfort_hot_rows['value'].mean()) if not discomfort_hot_rows.empty else 0.0
    district_kpis = kpis[(kpis['name'] == 'District') & (kpis['level'] == 'district')].copy()
    district_metric_map = district_kpis.set_index('cost_function')['value'].to_dict()

    aggregate_net = np.zeros(len(rew_list), dtype=float)
    trajectory_data: dict[str, np.ndarray | list[float]] = {
        'time_step': np.arange(len(rew_list)),
        'reward': rew_list,
        'all_rewards': np.cumsum(rew_list),
    }
    for building in buildings:
        net_consumption = np.asarray(building.net_electricity_consumption, dtype=float)[:len(rew_list)]
        aggregate_net += net_consumption
        trajectory_data[f'grid_import_{building.name}_kwh'] = np.clip(net_consumption, 0.0, None)
        trajectory_data[f'export_{building.name}_kwh'] = np.clip(-net_consumption, 0.0, None)

    grid_import = np.clip(aggregate_net, 0.0, None)
    export = np.clip(-aggregate_net, 0.0, None)
    trajectory_data['grid_import_kwh'] = grid_import
    trajectory_data['export_kwh'] = export
    trajectory_data['cumulative_grid_import_kwh'] = np.cumsum(grid_import)
    trajectory_data['cumulative_export_kwh'] = np.cumsum(export)

    return PolicyRun(
        result=ExperimentResult(
            policy='',
            total_grid_import_kwh=float(np.sum(grid_import)),
            total_export_kwh=float(np.sum(export)),
            total_net_consumption_kwh=float(np.sum(aggregate_net)),
            discomfort_proportion=discomfort,
            discomfort_cold_proportion=discomfort_cold,
            discomfort_hot_proportion=discomfort_hot,
            all_rewards=all_rewards,
            cost_total_ratio=float(district_metric_map['cost_total']) if 'cost_total' in district_metric_map and pd.notna(district_metric_map['cost_total']) else None,
            carbon_emissions_total_ratio=float(district_metric_map['carbon_emissions_total']) if 'carbon_emissions_total' in district_metric_map and pd.notna(district_metric_map['carbon_emissions_total']) else None,
            daily_peak_average_ratio=float(district_metric_map['daily_peak_average']) if 'daily_peak_average' in district_metric_map and pd.notna(district_metric_map['daily_peak_average']) else None,
            ramping_average_ratio=float(district_metric_map['ramping_average']) if 'ramping_average' in district_metric_map and pd.notna(district_metric_map['ramping_average']) else None,
        ),
        trajectory=pd.DataFrame(trajectory_data),
        kpis=kpis,
    )


def build_results_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'policy': result.policy,
            'grid_import_kwh': round(result.total_grid_import_kwh, 3),
            'export_kwh': round(result.total_export_kwh, 3),
            'net_consumption_kwh': round(result.total_net_consumption_kwh, 3),
            'discomfort_proportion': round(result.discomfort_proportion, 4),
            'discomfort_cold_proportion': round(result.discomfort_cold_proportion, 4),
            'discomfort_hot_proportion': round(result.discomfort_hot_proportion, 4),
            'all_rewards': round(result.all_rewards, 3),
            'cost_total_ratio': None if result.cost_total_ratio is None else round(result.cost_total_ratio, 4),
            'carbon_emissions_total_ratio': None if result.carbon_emissions_total_ratio is None else round(result.carbon_emissions_total_ratio, 4),
            'daily_peak_average_ratio': None if result.daily_peak_average_ratio is None else round(result.daily_peak_average_ratio, 4),
            'ramping_average_ratio': None if result.ramping_average_ratio is None else round(result.ramping_average_ratio, 4),
            'savings_vs_fixed_pct': None if result.savings_vs_fixed_pct is None else round(result.savings_vs_fixed_pct, 3),
            'train_sec': None if result.train_sec is None else round(result.train_sec, 2),
            'stability_episode': result.stability_episode,
            'last_10_episode_reward_mean': None if result.last_10_episode_reward_mean is None else round(result.last_10_episode_reward_mean, 3),
        }
        for result in results
    ])


def run_experiment(
    schema_path: Path,
    building_names: list[str],
    episodes: int,
    baseline_cooling: float,
    random_seed: int,
    output_dir: Path,
    comparison_horizon: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    fixed_env = make_env(schema_path, building_names, random_seed)
    fixed_policy = DecentralFixedPolicy(cooling_action=baseline_cooling)
    fixed_run = eval_agent(fixed_env, fixed_policy)
    fixed_run.result.policy = f'Fixed(cool={baseline_cooling:.2f})'
    fixed_run.result.savings_vs_fixed_pct = 0.0
    results.append(fixed_run.result)

    learned_runs: list[tuple[str, str, PolicyRun, TrainingTrace, DDPGAgent]] = []
    for key, display_name, reward_cls in REWARD_CONFIGS:
        print(f'reward: {display_name}')
        train_env = make_env(schema_path, building_names, random_seed, reward_function=reward_cls)
        agent = DDPGAgent(train_env, random_seed=random_seed)
        trace = train_ddpg(agent, train_env, episodes)

        eval_env = make_env(schema_path, building_names, random_seed, reward_function=reward_cls)
        run = eval_agent(eval_env, agent)
        run.result.policy = f'DDPG decentral ({display_name})'
        run.result.train_sec = trace.train_sec
        run.result.stability_episode = trace.stability_episode
        run.result.last_10_episode_reward_mean = float(np.mean(trace.episode_rewards[-10:])) if trace.episode_rewards else None
        if fixed_run.result.total_grid_import_kwh > 0.0:
            run.result.savings_vs_fixed_pct = 100.0 * (fixed_run.result.total_grid_import_kwh - run.result.total_grid_import_kwh) / fixed_run.result.total_grid_import_kwh

        results.append(run.result)
        learned_runs.append((key, display_name, run, trace, agent))

    results_frame = build_results_frame(results).reset_index(drop=True)
    results_frame.to_csv(output_dir / 'summary_results.csv', index=False)

    for key, _display_name, run, trace, agent in learned_runs:
        torch.save(agent.actor.state_dict(), output_dir / f'ddpg_actor_{key}.pt')
        torch.save(agent.critic.state_dict(), output_dir / f'ddpg_critic_{key}.pt')
        pd.DataFrame({
            'episode': np.arange(1, len(trace.episode_rewards) + 1),
            'episode_reward': trace.episode_rewards,
            'actor_loss': trace.actor_losses,
            'critic_loss': trace.critic_losses,
            'exploration_noise': trace.exploration_noise,
        }).to_csv(output_dir / f'learning_trace_{key}.csv', index=False)
        run.trajectory.to_csv(output_dir / f'trajectory_ddpg_{key}.csv', index=False)
        run.kpis.to_csv(output_dir / f'kpis_{key}.csv', index=False)

    fixed_run.trajectory.to_csv(output_dir / 'trajectory_fixed_strategy.csv', index=False)
    fixed_run.kpis.to_csv(output_dir / 'kpis_fixed.csv', index=False)

    return results_frame

#
def main() -> None:
    results = run_experiment(
        schema_path=DEFAULT_SCHEMA,
        building_names=DEFAULT_BUILDINGS,
        episodes=DEFAULT_EPISODES,
        baseline_cooling=DEFAULT_BASELINE_COOLING,
        random_seed=DEFAULT_RANDOM_SEED,
        output_dir=DEFAULT_OUTPUT_DIR,
        comparison_horizon=DEFAULT_COMPARISON_HORIZON,
    )
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()