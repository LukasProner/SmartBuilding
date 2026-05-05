from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_ddpg0305_02'
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / 'combined_rl_results.csv'
DEFAULT_PIVOT_PATH = DEFAULT_OUTPUT_DIR / 'combined_rl_results_pivot.csv'
DEFAULT_TABLE_PATH = DEFAULT_OUTPUT_DIR / 'combined_rl_results_table.csv'
DEFAULT_MARKDOWN_PATH = DEFAULT_OUTPUT_DIR / 'combined_rl_results_table.md'
DEFAULT_RATIO_PLOT_PATH = DEFAULT_OUTPUT_DIR / 'combined_rl_results_ratios.png'
DEFAULT_DDPG_SOURCE_DIR = PROJECT_ROOT / 'outputs_ddpg0305_02'
DEFAULT_DDPG_REBUILT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / 'ddpg_reward_summary_from_csv.csv'
DEFAULT_DDPG_POLICY_PLOT_PATH = DEFAULT_OUTPUT_DIR / 'ddpg_reward_policy_comparison_from_csv.png'
EXCLUDED_SOURCE_DIR_NAMES = {
    'outputs_ddpg0305_01_smoke',
}

REWARD_DISPLAY_NAMES = {
    'weather': 'Weather reward',
    'energy': 'Energy-only reward',
    'pricing': 'Pricing reward',
    'comfort': 'Comfort reward',
    'peak': 'Peak-shaving reward',
    'solar': 'Solar reward',
    'storage': 'Storage reward',
    'ramping': 'Ramping reward',
    'tou': 'TimeOfUse reward',
    'selfsuff': 'SelfSufficiency reward',
    'combined': 'Combined reward',
    'nightpre': 'NightPrecharge reward',
}

REWARD_ORDER = {
    key: index for index, key in enumerate(REWARD_DISPLAY_NAMES.keys())
}


def discover_source_dirs(project_root: Path) -> list[Path]:
    candidates = []
    for summary_path in sorted(project_root.glob('outputs_*/summary_results.csv')):
        source_dir = summary_path.parent
        if source_dir.name in EXCLUDED_SOURCE_DIR_NAMES:
            continue
        candidates.append(source_dir)
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


def infer_episodes(source_dir: Path, algorithm: str, reward: str) -> int | None:
    if algorithm == 'Fixed':
        return None

    reward_slug = (
        reward.lower()
        .replace(' reward', '')
        .replace('-only', '')
        .replace(' ', '_')
        .replace('-', '_')
        .replace('__', '_')
    )
    trace_candidates = list(source_dir.glob(f'learning_trace_{reward_slug}*.csv'))
    if not trace_candidates:
        algorithm_slug = algorithm.lower().replace('-', '').replace(' ', '_')
        trace_candidates = list(source_dir.glob(f'learning_trace_{algorithm_slug}*.csv'))
    if not trace_candidates:
        trace_candidates = list(source_dir.glob('learning_trace*.csv'))
    if not trace_candidates:
        return None

    trace = pd.read_csv(trace_candidates[0])
    if 'episode' in trace.columns and not trace.empty:
        return int(trace['episode'].max())
    return int(len(trace)) if not trace.empty else None


def infer_reward_key(source_dir: Path, algorithm: str, reward: str) -> str:
    if algorithm == 'Fixed':
        return 'fixed'

    normalized = (
        reward.lower()
        .replace(' reward', '')
        .replace('-only', '')
        .replace(' ', '_')
        .replace('-', '_')
        .replace('__', '_')
    )
    trace_candidates = list(source_dir.glob(f'learning_trace_{normalized}*.csv'))
    if trace_candidates:
        suffix = trace_candidates[0].stem.replace('learning_trace_', '', 1)
        return suffix or normalized
    return normalized


def load_result_folder(source_dir: Path) -> pd.DataFrame:
    summary_path = source_dir / 'summary_results.csv'
    if not summary_path.exists():
        raise FileNotFoundError(f'Missing summary_results.csv in {source_dir}')

    frame = pd.read_csv(summary_path)
    combined_rows: list[dict] = []
    for _, row in frame.iterrows():
        algorithm, reward = extract_algorithm_and_reward(row['policy'])
        combined_rows.append(
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

    return pd.DataFrame(combined_rows)


def build_combined_table(source_dirs: list[Path]) -> pd.DataFrame:
    frames = [frame for frame in (load_result_folder(path) for path in source_dirs) if not frame.empty]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(
        by=['algorithm', 'reward_key', 'episodes', 'source_dir'],
        ascending=[True, True, True, True],
        na_position='first',
    ).reset_index(drop=True)
    return combined


def build_pivot_table(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = frame[frame['algorithm'] != 'Fixed'].copy()
    if filtered.empty:
        return pd.DataFrame()

    pivot = filtered.pivot_table(
        index=['algorithm', 'reward_key', 'episodes'],
        values=['grid_import_kwh', 'savings_vs_fixed_pct', 'train_sec', 'stability_episode', 'discomfort_proportion'],
        aggfunc='first',
    ).reset_index()
    return pivot.sort_values(by=['algorithm', 'reward_key', 'episodes']).reset_index(drop=True)


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
    plot_frame = frame[frame['algorithm'] != 'Fixed'].copy()
    if plot_frame.empty:
        return

    plot_frame['label'] = plot_frame.apply(
        lambda row: f"{row['algorithm']}\n{row['reward_key']}\n{'' if pd.isna(row['episodes']) else int(row['episodes'])}",
        axis=1,
    )
    plot_frame = plot_frame.sort_values(by=['algorithm', 'reward_key', 'episodes'], na_position='last').reset_index(drop=True)
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

    fig.suptitle('Porovnanie pomerov voci fixnej strategii', fontsize=13)
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


def normalize_reward_key(value: str) -> str:
    return (
        str(value).strip().lower()
        .replace(' reward', '')
        .replace('-only', '')
        .replace(' ', '_')
        .replace('-', '_')
        .replace('__', '_')
    )


def display_name_from_reward_key(reward_key: str) -> str:
    normalized = normalize_reward_key(reward_key)
    return REWARD_DISPLAY_NAMES.get(normalized, normalized.replace('_', ' ').title())


def estimate_stability_episode(rewards: list[float], window: int = 10, tolerance: float = 0.03) -> int | None:
    if len(rewards) < window * 2:
        return None

    arr = np.asarray(rewards, dtype=float)
    for index in range((window * 2) - 1, len(arr)):
        previous = arr[index - (2 * window) + 1:index - window + 1]
        current = arr[index - window + 1:index + 1]
        previous_mean = float(np.mean(previous))
        current_mean = float(np.mean(current))
        scale = max(1.0, abs(previous_mean))
        if abs(current_mean - previous_mean) / scale <= tolerance:
            current_scale = max(1.0, abs(current_mean))
            if float(np.std(current)) / current_scale <= tolerance * 1.5:
                return index + 1

    return None


def extract_kpi_value(kpis: pd.DataFrame, cost_function: str, level: str) -> float | None:
    rows = kpis[(kpis['cost_function'] == cost_function) & (kpis['level'] == level)].copy()
    if rows.empty:
        return None
    values = pd.to_numeric(rows['value'], errors='coerce').dropna()
    if values.empty:
        return None
    return float(values.mean())


def build_fixed_row_from_csv(source_dir: Path, summary_frame: pd.DataFrame) -> dict:
    fixed_summary = summary_frame[summary_frame['policy'].astype(str).str.startswith('Fixed(')].copy()
    trajectory_path = source_dir / 'trajectory_fixed_strategy.csv'
    trajectory = pd.read_csv(trajectory_path)
    kpis_path = source_dir / 'kpis_fixed.csv'
    kpis = pd.read_csv(kpis_path) if kpis_path.exists() else pd.DataFrame(columns=['cost_function', 'value', 'level'])

    grid_import = float(pd.to_numeric(trajectory.get('grid_import_kwh'), errors='coerce').fillna(0.0).sum())
    export = float(pd.to_numeric(trajectory.get('export_kwh'), errors='coerce').fillna(0.0).sum())
    net_consumption = grid_import - export
    reward_series = pd.to_numeric(trajectory.get('all_rewards', trajectory.get('cumulative_reward')), errors='coerce').fillna(0.0)
    all_rewards = float(reward_series.iloc[-1]) if not reward_series.empty else 0.0
    policy = str(fixed_summary.iloc[0]['policy']) if not fixed_summary.empty else 'Fixed'

    return {
        'policy': policy,
        'grid_import_kwh': fixed_summary.iloc[0].get('grid_import_kwh', grid_import) if not fixed_summary.empty else grid_import,
        'export_kwh': fixed_summary.iloc[0].get('export_kwh', export) if not fixed_summary.empty else export,
        'net_consumption_kwh': fixed_summary.iloc[0].get('net_consumption_kwh', net_consumption) if not fixed_summary.empty else net_consumption,
        'discomfort_proportion': fixed_summary.iloc[0].get('discomfort_proportion', extract_kpi_value(kpis, 'discomfort_proportion', 'district')) if not fixed_summary.empty else extract_kpi_value(kpis, 'discomfort_proportion', 'district'),
        'discomfort_cold_proportion': fixed_summary.iloc[0].get('discomfort_cold_proportion', extract_kpi_value(kpis, 'discomfort_cold_proportion', 'district')) if not fixed_summary.empty else extract_kpi_value(kpis, 'discomfort_cold_proportion', 'district'),
        'discomfort_hot_proportion': fixed_summary.iloc[0].get('discomfort_hot_proportion', extract_kpi_value(kpis, 'discomfort_hot_proportion', 'district')) if not fixed_summary.empty else extract_kpi_value(kpis, 'discomfort_hot_proportion', 'district'),
        'all_rewards': fixed_summary.iloc[0].get('all_rewards', fixed_summary.iloc[0].get('cumulative_reward', all_rewards)) if not fixed_summary.empty else all_rewards,
        'cost_total_ratio': fixed_summary.iloc[0].get('cost_total_ratio', extract_kpi_value(kpis, 'cost_total', 'district')) if not fixed_summary.empty else extract_kpi_value(kpis, 'cost_total', 'district'),
        'carbon_emissions_total_ratio': fixed_summary.iloc[0].get('carbon_emissions_total_ratio', extract_kpi_value(kpis, 'carbon_emissions_total', 'district')) if not fixed_summary.empty else extract_kpi_value(kpis, 'carbon_emissions_total', 'district'),
        'daily_peak_average_ratio': fixed_summary.iloc[0].get('daily_peak_average_ratio', extract_kpi_value(kpis, 'daily_peak_average', 'district')) if not fixed_summary.empty else extract_kpi_value(kpis, 'daily_peak_average', 'district'),
        'ramping_average_ratio': fixed_summary.iloc[0].get('ramping_average_ratio', extract_kpi_value(kpis, 'ramping_average', 'district')) if not fixed_summary.empty else extract_kpi_value(kpis, 'ramping_average', 'district'),
        'savings_vs_fixed_pct': 0.0,
        'train_sec': fixed_summary.iloc[0].get('train_sec', fixed_summary.iloc[0].get('training_seconds')) if not fixed_summary.empty else None,
        'stability_episode': fixed_summary.iloc[0].get('stability_episode') if not fixed_summary.empty else None,
        'last_10_episode_reward_mean': fixed_summary.iloc[0].get('last_10_episode_reward_mean') if not fixed_summary.empty else None,
    }


def build_reward_row_from_csv(source_dir: Path, reward_key: str, summary_frame: pd.DataFrame, fixed_grid_import: float) -> dict:
    trajectory_path = source_dir / f'trajectory_ddpg_{reward_key}.csv'
    trajectory = pd.read_csv(trajectory_path)
    trace_path = source_dir / f'learning_trace_{reward_key}.csv'
    trace = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
    kpis_path = source_dir / f'kpis_{reward_key}.csv'
    kpis = pd.read_csv(kpis_path) if kpis_path.exists() else pd.DataFrame(columns=['cost_function', 'value', 'level'])

    reward_display = display_name_from_reward_key(reward_key)
    reward_summary = summary_frame[
        summary_frame['policy'].astype(str).map(lambda value: normalize_reward_key(extract_algorithm_and_reward(value)[1]) == reward_key)
    ].copy()

    grid_import = float(pd.to_numeric(trajectory.get('grid_import_kwh'), errors='coerce').fillna(0.0).sum())
    export = float(pd.to_numeric(trajectory.get('export_kwh'), errors='coerce').fillna(0.0).sum())
    net_consumption = grid_import - export
    reward_series = pd.to_numeric(trajectory.get('all_rewards', trajectory.get('cumulative_reward')), errors='coerce').fillna(0.0)
    all_rewards = float(reward_series.iloc[-1]) if not reward_series.empty else 0.0
    episode_rewards = pd.to_numeric(trace.get('episode_reward'), errors='coerce').dropna().tolist()
    stability_episode = estimate_stability_episode(episode_rewards) if episode_rewards else None
    last_10_episode_reward_mean = float(np.mean(episode_rewards[-10:])) if episode_rewards else None
    savings_vs_fixed_pct = None if fixed_grid_import == 0.0 else 100.0 * (fixed_grid_import - grid_import) / fixed_grid_import

    return {
        'policy': str(reward_summary.iloc[0]['policy']) if not reward_summary.empty else f'DDPG ({reward_display})',
        'grid_import_kwh': reward_summary.iloc[0].get('grid_import_kwh', grid_import) if not reward_summary.empty else grid_import,
        'export_kwh': reward_summary.iloc[0].get('export_kwh', export) if not reward_summary.empty else export,
        'net_consumption_kwh': reward_summary.iloc[0].get('net_consumption_kwh', net_consumption) if not reward_summary.empty else net_consumption,
        'discomfort_proportion': reward_summary.iloc[0].get('discomfort_proportion', extract_kpi_value(kpis, 'discomfort_proportion', 'district')) if not reward_summary.empty else extract_kpi_value(kpis, 'discomfort_proportion', 'district'),
        'discomfort_cold_proportion': reward_summary.iloc[0].get('discomfort_cold_proportion', extract_kpi_value(kpis, 'discomfort_cold_proportion', 'district')) if not reward_summary.empty else extract_kpi_value(kpis, 'discomfort_cold_proportion', 'district'),
        'discomfort_hot_proportion': reward_summary.iloc[0].get('discomfort_hot_proportion', extract_kpi_value(kpis, 'discomfort_hot_proportion', 'district')) if not reward_summary.empty else extract_kpi_value(kpis, 'discomfort_hot_proportion', 'district'),
        'all_rewards': reward_summary.iloc[0].get('all_rewards', reward_summary.iloc[0].get('cumulative_reward', all_rewards)) if not reward_summary.empty else all_rewards,
        'cost_total_ratio': reward_summary.iloc[0].get('cost_total_ratio', extract_kpi_value(kpis, 'cost_total', 'district')) if not reward_summary.empty else extract_kpi_value(kpis, 'cost_total', 'district'),
        'carbon_emissions_total_ratio': reward_summary.iloc[0].get('carbon_emissions_total_ratio', extract_kpi_value(kpis, 'carbon_emissions_total', 'district')) if not reward_summary.empty else extract_kpi_value(kpis, 'carbon_emissions_total', 'district'),
        'daily_peak_average_ratio': reward_summary.iloc[0].get('daily_peak_average_ratio', extract_kpi_value(kpis, 'daily_peak_average', 'district')) if not reward_summary.empty else extract_kpi_value(kpis, 'daily_peak_average', 'district'),
        'ramping_average_ratio': reward_summary.iloc[0].get('ramping_average_ratio', extract_kpi_value(kpis, 'ramping_average', 'district')) if not reward_summary.empty else extract_kpi_value(kpis, 'ramping_average', 'district'),
        'savings_vs_fixed_pct': reward_summary.iloc[0].get('savings_vs_fixed_pct', savings_vs_fixed_pct) if not reward_summary.empty else savings_vs_fixed_pct,
        'train_sec': reward_summary.iloc[0].get('train_sec', reward_summary.iloc[0].get('training_seconds')) if not reward_summary.empty else None,
        'stability_episode': reward_summary.iloc[0].get('stability_episode', stability_episode) if not reward_summary.empty else stability_episode,
        'last_10_episode_reward_mean': reward_summary.iloc[0].get('last_10_episode_reward_mean', last_10_episode_reward_mean) if not reward_summary.empty else last_10_episode_reward_mean,
    }


def build_ddpg_results_from_csv(source_dir: Path) -> tuple[pd.DataFrame, list[tuple[str, pd.DataFrame]]]:
    summary_path = source_dir / 'summary_results.csv'
    summary_frame = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(columns=['policy'])
    fixed_row = build_fixed_row_from_csv(source_dir, summary_frame)

    reward_keys = sorted(
        {
            path.stem.replace('trajectory_ddpg_', '', 1)
            for path in source_dir.glob('trajectory_ddpg_*.csv')
        },
        key=lambda key: (REWARD_ORDER.get(key, 10_000), key),
    )
    reward_rows = [build_reward_row_from_csv(source_dir, reward_key, summary_frame, float(fixed_row['grid_import_kwh'])) for reward_key in reward_keys]
    results_frame = pd.DataFrame([fixed_row] + reward_rows)

    trajectory_runs = [(str(fixed_row['policy']), pd.read_csv(source_dir / 'trajectory_fixed_strategy.csv'))]
    trajectory_runs.extend(
        (str(reward_row['policy']), pd.read_csv(source_dir / f'trajectory_ddpg_{reward_key}.csv'))
        for reward_key, reward_row in zip(reward_keys, reward_rows)
    )
    return results_frame, trajectory_runs


def save_ddpg_reward_comparison_plot(results_frame: pd.DataFrame, trajectory_runs: list[tuple[str, pd.DataFrame]], output_path: Path) -> None:
    labels = results_frame['policy'].tolist()
    x_axis = np.arange(len(labels))
    colors = ['#9aa0a6'] + [plt.get_cmap('tab20')(index) for index in range(max(0, len(labels) - 1))]

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))

    axes[0, 0].bar(x_axis, pd.to_numeric(results_frame['grid_import_kwh'], errors='coerce').fillna(0.0), color=colors[:len(labels)])
    axes[0, 0].set_title('Total grid import', fontsize=12)
    axes[0, 0].set_ylabel('kWh')
    axes[0, 0].set_xticks(x_axis, labels, rotation=35, ha='right', fontsize=8)
    axes[0, 0].grid(axis='y', alpha=0.25)

    axes[0, 1].bar(x_axis, pd.to_numeric(results_frame['savings_vs_fixed_pct'], errors='coerce').fillna(0.0), color=colors[:len(labels)])
    axes[0, 1].set_title('Savings vs fixed strategy', fontsize=12)
    axes[0, 1].set_ylabel('%')
    axes[0, 1].set_xticks(x_axis, labels, rotation=35, ha='right', fontsize=8)
    axes[0, 1].grid(axis='y', alpha=0.25)

    axes[1, 0].bar(x_axis, pd.to_numeric(results_frame['discomfort_proportion'], errors='coerce').fillna(0.0), color=colors[:len(labels)])
    axes[1, 0].set_title('Discomfort proportion', fontsize=12)
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].set_xticks(x_axis, labels, rotation=35, ha='right', fontsize=8)
    axes[1, 0].grid(axis='y', alpha=0.25)

    profile_hours = min(14 * 24, min(len(trajectory) for _, trajectory in trajectory_runs))
    profile_index = np.arange(profile_hours)
    for color, (policy, trajectory) in zip(colors[:len(trajectory_runs)], trajectory_runs):
        grouped = trajectory.groupby(trajectory['time_step'] % profile_hours)['grid_import_kwh'].mean()
        linestyle = '--' if str(policy).startswith('Fixed(') else '-'
        axes[1, 1].plot(
            profile_index,
            grouped.reindex(profile_index, fill_value=np.nan).to_numpy(),
            linestyle=linestyle,
            linewidth=1.8,
            color=color,
            label=policy,
        )
    axes[1, 1].set_title('Average grid import over 14-day profile', fontsize=12)
    axes[1, 1].set_xlabel('Hour in 14-day cycle')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].set_xticks(range(0, profile_hours + 1, 24))
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(loc='best', fontsize=7, ncol=2)

    fig.suptitle(f'3 budovy: Fixed vs {len(labels) - 1} reward variantov DDPG', fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_ddpg_folder_report(source_dir: Path, summary_output_path: Path, figure_output_path: Path) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f'Source directory not found: {source_dir}')

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    results_frame, trajectory_runs = build_ddpg_results_from_csv(source_dir)
    results_frame.to_csv(summary_output_path, index=False)
    save_ddpg_reward_comparison_plot(results_frame, trajectory_runs, figure_output_path)

    print('DDPG summary rebuilt from CSV files:', summary_output_path)
    print('DDPG policy comparison plot saved to:', figure_output_path)
    print('\nDDPG table preview:')
    print(results_frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate RL result CSV files.')
    parser.add_argument('--mode', choices=['ddpg-folder', 'combined'], default='ddpg-folder')
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_DDPG_SOURCE_DIR)
    parser.add_argument('--summary-output', type=Path, default=DEFAULT_DDPG_REBUILT_SUMMARY_PATH)
    parser.add_argument('--figure-output', type=Path, default=DEFAULT_DDPG_POLICY_PLOT_PATH)
    args = parser.parse_args()

    if args.mode == 'ddpg-folder':
        run_ddpg_folder_report(args.source_dir, args.summary_output, args.figure_output)
        return

    existing_dirs = discover_source_dirs(PROJECT_ROOT)
    if not existing_dirs:
        raise FileNotFoundError('No default output directories found.')

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined = build_combined_table(existing_dirs)
    combined.to_csv(DEFAULT_OUTPUT_PATH, index=False)

    pivot = build_pivot_table(combined)
    pivot.to_csv(DEFAULT_PIVOT_PATH, index=False)

    summary_table = build_summary_table(combined)
    summary_table.to_csv(DEFAULT_TABLE_PATH, index=False)
    write_markdown_table(summary_table, DEFAULT_MARKDOWN_PATH)
    save_ratio_plot(summary_table, DEFAULT_RATIO_PLOT_PATH)

    print('Combined results saved to:', DEFAULT_OUTPUT_PATH)
    print('Pivot results saved to:', DEFAULT_PIVOT_PATH)
    print('Summary table saved to:', DEFAULT_TABLE_PATH)
    print('Markdown table saved to:', DEFAULT_MARKDOWN_PATH)
    print('Ratio plot saved to:', DEFAULT_RATIO_PLOT_PATH)
    print('\nCombined table preview:')
    print(summary_table.to_string(index=False))


if __name__ == '__main__':
    main()