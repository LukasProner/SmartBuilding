from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / 'outputs_q_learning0305_02'
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / 'combined_q_learning_results.csv'
DEFAULT_PIVOT_PATH = DEFAULT_OUTPUT_DIR / 'combined_q_learning_results_pivot.csv'
DEFAULT_TABLE_PATH = DEFAULT_OUTPUT_DIR / 'combined_q_learning_results_table.csv'
DEFAULT_MARKDOWN_PATH = DEFAULT_OUTPUT_DIR / 'combined_q_learning_results_table.md'
DEFAULT_RATIO_PLOT_PATH = DEFAULT_OUTPUT_DIR / 'combined_q_learning_results_ratios.png'


def discover_source_dirs(project_root: Path) -> list[Path]:
    candidates = []
    for summary_path in sorted(project_root.glob('outputs_q_learning*/summary_results.csv')):
        candidates.append(summary_path.parent)
    return candidates


def extract_algorithm_and_reward(policy: str) -> tuple[str, str]:
    policy = str(policy).strip()
    if policy.startswith('Fixed('):
        return 'Fixed', 'fixed'

    match = re.match(r'^(?P<algorithm>[^()]+?)\s*\((?P<reward>.+)\)$', policy)
    if match is None:
        return policy, 'unknown'

    algorithm = match.group('algorithm').strip()
    reward = match.group('reward').strip()
    return algorithm, reward


def normalize_reward_key(reward: str) -> str:
    return (
        str(reward).strip().lower()
        .replace(' reward', '')
        .replace('-only', '')
        .replace(' ', '_')
        .replace('-', '_')
        .replace('__', '_')
    )


def infer_episodes(source_dir: Path, algorithm: str, reward: str) -> int | None:
    if algorithm == 'Fixed':
        return None

    reward_key = normalize_reward_key(reward)
    trace_path = source_dir / f'learning_trace_{reward_key}.csv'
    if not trace_path.exists():
        return None

    trace = pd.read_csv(trace_path)
    if 'episode' in trace.columns and not trace.empty:
        return int(trace['episode'].max())
    return int(len(trace)) if not trace.empty else None


def infer_reward_key(_source_dir: Path, algorithm: str, reward: str) -> str:
    if algorithm == 'Fixed':
        return 'fixed'
    return normalize_reward_key(reward)


def load_result_folder(source_dir: Path) -> pd.DataFrame:
    summary_path = source_dir / 'summary_results.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary_results.csv in {source_dir}')

    frame = pd.read_csv(summary_path)
    rows: list[dict] = []
    for _, row in frame.iterrows():
        algorithm, reward = extract_algorithm_and_reward(row['policy'])
        if algorithm not in {'Fixed', 'Q-learning'}:
            continue

        rows.append(
            {
                'source_dir': source_dir.name,
                'algorithm': algorithm,
                'reward_name': reward,
                'reward_key': infer_reward_key(source_dir, algorithm, reward),
                'episodes': infer_episodes(source_dir, algorithm, reward),
                'policy': row['policy'],
                'grid_import_kwh': row.get('grid_import_kwh'),
                'export_kwh': row.get('export_kwh'),
                'net_consumption_kwh': row.get('net_consumption_kwh'),
                'discomfort_proportion': row.get('discomfort_proportion'),
                'discomfort_cold_proportion': row.get('discomfort_cold_proportion'),
                'discomfort_hot_proportion': row.get('discomfort_hot_proportion'),
                'all_rewards': row.get('all_rewards', row.get('cumulative_reward')),
                'cost_total_ratio': row.get('cost_total_ratio'),
                'carbon_emissions_total_ratio': row.get('carbon_emissions_total_ratio'),
                'daily_peak_average_ratio': row.get('daily_peak_average_ratio'),
                'ramping_average_ratio': row.get('ramping_average_ratio'),
                'savings_vs_fixed_pct': row.get('savings_vs_fixed_pct'),
                'train_sec': row.get('train_sec', row.get('training_seconds')),
                'stability_episode': row.get('stability_episode'),
                'last_10_episode_reward_mean': row.get('last_10_episode_reward_mean'),
            }
        )

    return pd.DataFrame(rows)


def build_combined_table(source_dirs: list[Path]) -> pd.DataFrame:
    frames = [frame for frame in (load_result_folder(path) for path in source_dirs) if not frame.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        by=['algorithm', 'reward_key', 'episodes', 'source_dir'],
        ascending=[True, True, True, True],
        na_position='first',
    ).reset_index(drop=True)
    return combined


def build_pivot_table(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame[frame['algorithm'] == 'Q-learning'].copy()
    if filtered.empty:
        return pd.DataFrame()

    pivot = filtered.pivot_table(
        index=['algorithm', 'reward_key', 'episodes'],
        values=['grid_import_kwh', 'savings_vs_fixed_pct', 'train_sec', 'stability_episode', 'discomfort_proportion'],
        aggfunc='first',
    ).reset_index()
    return pivot.sort_values(by=['reward_key', 'episodes', 'source_dir'] if 'source_dir' in pivot.columns else ['reward_key', 'episodes']).reset_index(drop=True)


def build_summary_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    fixed_lookup = (
        table[table['algorithm'] == 'Fixed']
        .drop_duplicates(subset=['source_dir'])
        .set_index('source_dir')
    )

    def ratio_to_fixed(row: pd.Series, metric: str) -> float | None:
        if row['algorithm'] == 'Fixed' or row['source_dir'] not in fixed_lookup.index:
            return None
        fixed_value = fixed_lookup.loc[row['source_dir'], metric]
        current_value = row.get(metric)
        if pd.isna(fixed_value) or pd.isna(current_value) or float(fixed_value) == 0.0:
            return None
        return float(current_value) / float(fixed_value)

    table['grid_import_ratio_vs_fixed'] = table.apply(lambda row: ratio_to_fixed(row, 'grid_import_kwh'), axis=1)
    table['discomfort_ratio_vs_fixed'] = table.apply(lambda row: ratio_to_fixed(row, 'discomfort_proportion'), axis=1)
    table['cost_ratio_vs_fixed'] = table.apply(lambda row: ratio_to_fixed(row, 'cost_total_ratio'), axis=1)

    preferred_columns = [
        'source_dir',
        'algorithm',
        'reward_key',
        'episodes',
        'grid_import_kwh',
        'grid_import_ratio_vs_fixed',
        'savings_vs_fixed_pct',
        'discomfort_proportion',
        'discomfort_ratio_vs_fixed',
        'cost_total_ratio',
        'cost_ratio_vs_fixed',
        'carbon_emissions_total_ratio',
        'daily_peak_average_ratio',
        'ramping_average_ratio',
        'train_sec',
        'stability_episode',
        'last_10_episode_reward_mean',
    ]
    table = table[preferred_columns].copy()
    table = table.sort_values(
        by=['algorithm', 'reward_key', 'episodes', 'source_dir'],
        ascending=[True, True, True, True],
        na_position='first',
    ).reset_index(drop=True)
    return table


def save_ratio_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plot_frame = frame[frame['algorithm'] == 'Q-learning'].copy()
    if plot_frame.empty:
        return

    plot_frame['label'] = plot_frame.apply(
        lambda row: f"{row['reward_key']}\n{'' if pd.isna(row['episodes']) else int(row['episodes'])}",
        axis=1,
    )
    plot_frame = plot_frame.sort_values(by=['reward_key', 'episodes'], na_position='last').reset_index(drop=True)
    x_axis = range(len(plot_frame))
    colors = plt.get_cmap('tab20')(range(len(plot_frame)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_specs = [
        ('grid_import_ratio_vs_fixed', 'Grid Import / Fixed', 'Ratio'),
        ('discomfort_ratio_vs_fixed', 'Discomfort / Fixed', 'Ratio'),
        ('cost_ratio_vs_fixed', 'Cost / Fixed', 'Ratio'),
    ]

    for axis, (metric, title, ylabel) in zip(axes, metric_specs):
        values = plot_frame[metric].fillna(0.0)
        axis.bar(x_axis, values, color=colors)
        axis.axhline(1.0, color='#666666', linestyle='--', linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(x_axis), plot_frame['label'], rotation=45, ha='right', fontsize=8)
        axis.grid(axis='y', alpha=0.25)

    fig.suptitle('Q-learning reward comparison vs fixed strategy', fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_markdown_table(frame: pd.DataFrame, output_path: Path) -> None:
    display_frame = frame.copy()
    for column in display_frame.columns:
        if pd.api.types.is_float_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].map(lambda value: '' if pd.isna(value) else f'{value:.3f}')
        else:
            display_frame[column] = display_frame[column].fillna('')

    headers = [str(column) for column in display_frame.columns]
    rows = [[str(value) for value in row] for row in display_frame.to_numpy().tolist()]
    widths = []
    for index, header in enumerate(headers):
        column_values = [row[index] for row in rows]
        widths.append(max([len(header)] + [len(value) for value in column_values]))

    def format_row(values: list[str]) -> str:
        return '| ' + ' | '.join(value.ljust(width) for value, width in zip(values, widths)) + ' |'

    separator = '| ' + ' | '.join('-' * width for width in widths) + ' |'
    markdown_lines = [format_row(headers), separator]
    markdown_lines.extend(format_row(row) for row in rows)
    output_path.write_text('\n'.join(markdown_lines) + '\n', encoding='utf-8')


def run_folder_report(source_dir: Path, output_dir: Path) -> None:
    frame = build_combined_table([source_dir])
    if frame.empty:
        raise FileNotFoundError(f'No Q-learning results found in {source_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / DEFAULT_OUTPUT_PATH.name, index=False)

    pivot = build_pivot_table(frame)
    pivot.to_csv(output_dir / DEFAULT_PIVOT_PATH.name, index=False)

    summary_table = build_summary_table(frame)
    summary_table.to_csv(output_dir / DEFAULT_TABLE_PATH.name, index=False)
    write_markdown_table(summary_table, output_dir / DEFAULT_MARKDOWN_PATH.name)
    save_ratio_plot(summary_table, output_dir / DEFAULT_RATIO_PLOT_PATH.name)

    print('Q-learning combined results saved to:', output_dir / DEFAULT_OUTPUT_PATH.name)
    print('Q-learning pivot results saved to:', output_dir / DEFAULT_PIVOT_PATH.name)
    print('Q-learning summary table saved to:', output_dir / DEFAULT_TABLE_PATH.name)
    print('Q-learning markdown table saved to:', output_dir / DEFAULT_MARKDOWN_PATH.name)
    print('Q-learning ratio plot saved to:', output_dir / DEFAULT_RATIO_PLOT_PATH.name)
    print('\nQ-learning table preview:')
    print(summary_table.to_string(index=False))


def run_combined_report(project_root: Path, output_dir: Path) -> None:
    source_dirs = discover_source_dirs(project_root)
    if not source_dirs:
        raise FileNotFoundError('No Q-learning output directories found.')

    frame = build_combined_table(source_dirs)
    if frame.empty:
        raise FileNotFoundError('No Q-learning rows found in discovered summary files.')

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / DEFAULT_OUTPUT_PATH.name, index=False)

    pivot = build_pivot_table(frame)
    pivot.to_csv(output_dir / DEFAULT_PIVOT_PATH.name, index=False)

    summary_table = build_summary_table(frame)
    summary_table.to_csv(output_dir / DEFAULT_TABLE_PATH.name, index=False)
    write_markdown_table(summary_table, output_dir / DEFAULT_MARKDOWN_PATH.name)
    save_ratio_plot(summary_table, output_dir / DEFAULT_RATIO_PLOT_PATH.name)

    print('Q-learning combined results saved to:', output_dir / DEFAULT_OUTPUT_PATH.name)
    print('Q-learning pivot results saved to:', output_dir / DEFAULT_PIVOT_PATH.name)
    print('Q-learning summary table saved to:', output_dir / DEFAULT_TABLE_PATH.name)
    print('Q-learning markdown table saved to:', output_dir / DEFAULT_MARKDOWN_PATH.name)
    print('Q-learning ratio plot saved to:', output_dir / DEFAULT_RATIO_PLOT_PATH.name)
    print('\nQ-learning table preview:')
    print(summary_table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate Q-learning result CSV files only.')
    parser.add_argument('--mode', choices=['folder', 'combined'], default='folder')
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.mode == 'folder':
        run_folder_report(args.source_dir, args.output_dir)
        return

    run_combined_report(PROJECT_ROOT, args.output_dir)


if __name__ == '__main__':
    main()