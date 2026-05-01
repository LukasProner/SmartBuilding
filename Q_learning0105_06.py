from __future__ import annotations

import argparse
from pathlib import Path

import Q_learning0105_04 as base


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCHEMA = PROJECT_ROOT / 'data' / 'datasets' / 'citylearn_challenge_2023_phase_1' / 'schema.json'
DEFAULT_BUILDINGS = ['Building_1']
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning0105_06'


base.REWARD_CONFIGS = [
    ('grid', 'Grid import only', base.GridImportOnlyReward),
    ('peak', 'Peak shaving', base.PeakShavingReward),
    ('tou', 'TimeOfUse reward', base.TimeOfUseReward),
    ('solar', 'Solar alignment', base.SolarAlignmentReward),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Q-learning experiment pre 1 budovu s rewardmi GridImportOnly, PeakShaving, TimeOfUse a SolarAlignment.'
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
