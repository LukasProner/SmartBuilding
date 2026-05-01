from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import Q_learning0105_04 as base
from citylearn.reward_function import RewardFunction


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1' / 'schema.json'
DEFAULT_BUILDINGS = ['Building_1']
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning0105_07'


class SolarPriceArbitrageReward(RewardFunction):
    """Shape storage behavior toward low-price/high-solar charging and high-price discharge.

    The goal is not to optimize one-step import only, but to make the storage state useful
    for later expensive hours while still penalizing import spikes.
    """

    def __init__(
        self,
        env_metadata,
        import_weight: float = 1.0,
        price_weight: float = 2.2,
        peak_weight: float = 0.35,
        soc_weight: float = 1.25,
        export_penalty: float = 0.15,
        **kwargs,
    ):
        super().__init__(env_metadata, **kwargs)
        self.import_weight = float(import_weight)
        self.price_weight = float(price_weight)
        self.peak_weight = float(peak_weight)
        self.soc_weight = float(soc_weight)
        self.export_penalty = float(export_penalty)

    def calculate(self, observations: list[dict]) -> list[float]:
        rewards: list[float] = []
        for obs in observations:
            net = float(obs['net_electricity_consumption'])
            grid_import = max(net, 0.0)
            export = max(-net, 0.0)

            price_now = max(float(obs.get('electricity_pricing', 0.0)), 0.0)
            price_next = max(float(obs.get('electricity_pricing_predicted_1', price_now)), 0.0)
            price_signal = np.clip((price_now + price_next) / 2.0, 0.0, 1.0)

            diffuse = max(float(obs.get('diffuse_solar_irradiance_predicted_1', 0.0)), 0.0)
            direct = max(float(obs.get('direct_solar_irradiance_predicted_1', 0.0)), 0.0)
            solar_signal = np.clip((diffuse + direct) / 900.0, 0.0, 1.0)

            avg_soc = (
                float(obs.get('dhw_storage_soc', 0.5))
                + float(obs.get('electrical_storage_soc', 0.5))
            ) / 2.0

            # Charge preference when power is expected to be cheap or solar-rich.
            charge_window = max(solar_signal, 1.0 - price_signal)
            # Discharge preference when power is expensive and import is positive.
            discharge_window = price_signal
            target_soc = np.clip(0.2 + 0.65 * charge_window - 0.45 * discharge_window, 0.15, 0.9)
            soc_penalty = abs(avg_soc - target_soc)

            import_penalty = grid_import * (self.import_weight + self.price_weight * price_signal)
            peak_penalty = self.peak_weight * (grid_import ** 2)
            export_penalty = self.export_penalty * export * price_signal

            reward = -(import_penalty + peak_penalty + self.soc_weight * soc_penalty + export_penalty)
            rewards.append(reward)

        return [sum(rewards)] if self.central_agent else rewards


base.REWARD_CONFIGS = [
    ('arbitrage', 'Solar-price arbitrage', SolarPriceArbitrageReward),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Q-learning experiment pre 1 budovu s jednou reward funkciou zameranou na solar-price arbitrage.'
    )
    parser.add_argument('--schema', type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument('--buildings', nargs='+', default=DEFAULT_BUILDINGS)
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--baseline-cooling', type=float, default=0.0)
    parser.add_argument('--random-seed', type=int, default=7)
    parser.add_argument('--comparison-horizon', type=int, default=719)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = base.run_experiment(
        schema_path=args.schema,
        building_names=args.buildings,
        episodes=args.episodes,
        baseline_cooling=args.baseline_cooling,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
        comparison_horizon=args.comparison_horizon,
    )
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()