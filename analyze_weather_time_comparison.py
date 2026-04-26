from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'outputs_q_learning2104_weather_only'
DEFAULT_FIXED = DEFAULT_OUTPUT_DIR / 'trajectory_fixed_strategy.csv'
DEFAULT_LEARNED = DEFAULT_OUTPUT_DIR / 'trajectory_q_learning_weather_seed_7.csv'
DEFAULT_LEARNING_TRACE = DEFAULT_OUTPUT_DIR / 'learning_trace_weather_seed_7.csv'
DEFAULT_MERGED = DEFAULT_OUTPUT_DIR / 'time_comparison_fixed_vs_q_learning_weather_seed_7.csv'
DEFAULT_FIGURE = DEFAULT_OUTPUT_DIR / 'time_comparison_fixed_vs_q_learning_weather_seed_7.png'
DEFAULT_ZOOM_FIGURE = DEFAULT_OUTPUT_DIR / 'time_comparison_zoom_first_96_steps_weather_seed_7.png'
DEFAULT_COMBINED_FIGURE = DEFAULT_OUTPUT_DIR / 'learning_progress_and_step_reward_first_96_steps_weather_seed_7.png'

FIXED_COLOR = '#9aa0a6'
LEARNED_COLOR = '#1d3557'
DELTA_COLOR = '#2a9d8f'


def load_trajectory(path: Path, prefix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {
        'time_step',
        'reward',
        'cumulative_reward',
        'grid_import_kwh',
        'cumulative_grid_import_kwh',
    }
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')
    return frame.rename(columns={column: f'{prefix}_{column}' for column in frame.columns if column != 'time_step'})


def load_learning_trace(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {'episode', 'episode_reward', 'epsilon'}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f'{path} is missing columns: {sorted(missing)}')
    return frame.sort_values('episode').reset_index(drop=True)


def build_comparison_frame(fixed_path: Path, learned_path: Path) -> pd.DataFrame:
    fixed = load_trajectory(fixed_path, 'fixed')
    learned = load_trajectory(learned_path, 'q_learning')
    merged = fixed.merge(learned, on='time_step', how='inner').sort_values('time_step').reset_index(drop=True)
    merged['reward_delta_q_minus_fixed'] = merged['q_learning_reward'] - merged['fixed_reward']
    merged['grid_import_delta_fixed_minus_q'] = merged['fixed_grid_import_kwh'] - merged['q_learning_grid_import_kwh']
    merged['cumulative_reward_delta_q_minus_fixed'] = (
        merged['q_learning_cumulative_reward'] - merged['fixed_cumulative_reward']
    )
    merged['cumulative_grid_import_delta_fixed_minus_q'] = (
        merged['fixed_cumulative_grid_import_kwh'] - merged['q_learning_cumulative_grid_import_kwh']
    )
    return merged


def save_figure(comparison: pd.DataFrame, output_path: Path) -> None:
    time_step = comparison['time_step']

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex='col')

    axes[0, 0].plot(time_step, comparison['fixed_reward'], '--', lw=1.4, color=FIXED_COLOR, label='Fixed')
    axes[0, 0].plot(time_step, comparison['q_learning_reward'], lw=1.6, color=LEARNED_COLOR, label='Q-learning')
    axes[0, 0].set_title('Step reward over time')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=9)

    axes[0, 1].plot(time_step, comparison['reward_delta_q_minus_fixed'], lw=1.6, color=DELTA_COLOR)
    axes[0, 1].axhline(0.0, color='black', lw=1.0, alpha=0.4)
    axes[0, 1].set_title('Reward difference over time')
    axes[0, 1].set_ylabel('Q-learning - Fixed')
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(time_step, comparison['fixed_cumulative_reward'], '--', lw=1.4, color=FIXED_COLOR, label='Fixed')
    axes[1, 0].plot(time_step, comparison['q_learning_cumulative_reward'], lw=1.6, color=LEARNED_COLOR, label='Q-learning')
    axes[1, 0].set_title('Cumulative reward over time')
    axes[1, 0].set_ylabel('Cumulative reward')
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=9)

    axes[1, 1].plot(time_step, comparison['fixed_grid_import_kwh'], '--', lw=1.4, color=FIXED_COLOR, label='Fixed')
    axes[1, 1].plot(time_step, comparison['q_learning_grid_import_kwh'], lw=1.6, color=LEARNED_COLOR, label='Q-learning')
    axes[1, 1].set_title('Grid import per step')
    axes[1, 1].set_ylabel('kWh')
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(fontsize=9)

    axes[2, 0].plot(time_step, comparison['grid_import_delta_fixed_minus_q'], lw=1.6, color=DELTA_COLOR)
    axes[2, 0].axhline(0.0, color='black', lw=1.0, alpha=0.4)
    axes[2, 0].set_title('Grid import savings per step')
    axes[2, 0].set_xlabel('Time step')
    axes[2, 0].set_ylabel('Fixed - Q-learning (kWh)')
    axes[2, 0].grid(alpha=0.25)

    axes[2, 1].plot(time_step, comparison['fixed_cumulative_grid_import_kwh'], '--', lw=1.4, color=FIXED_COLOR, label='Fixed')
    axes[2, 1].plot(time_step, comparison['q_learning_cumulative_grid_import_kwh'], lw=1.6, color=LEARNED_COLOR, label='Q-learning')
    axes[2, 1].set_title('Cumulative grid import over time')
    axes[2, 1].set_xlabel('Time step')
    axes[2, 1].set_ylabel('kWh')
    axes[2, 1].grid(alpha=0.25)
    axes[2, 1].legend(fontsize=9)

    fig.suptitle('Fixed vs Q-learning time comparison')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_zoom_figure(comparison: pd.DataFrame, output_path: Path, start_step: int, end_step: int) -> None:
    zoom = comparison[(comparison['time_step'] >= start_step) & (comparison['time_step'] <= end_step)].copy()
    if zoom.empty:
        raise ValueError(f'No rows found between time_step={start_step} and time_step={end_step}.')

    time_step = zoom['time_step']
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(time_step, zoom['fixed_reward'], '--', lw=1.6, color=FIXED_COLOR, label='Fixed')
    axes[0].plot(time_step, zoom['q_learning_reward'], lw=1.8, color=LEARNED_COLOR, label='Q-learning')
    axes[0].set_title(f'Step reward from {start_step} to {end_step}')
    axes[0].set_ylabel('Reward')
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].plot(time_step, zoom['fixed_grid_import_kwh'], '--', lw=1.6, color=FIXED_COLOR, label='Fixed')
    axes[1].plot(time_step, zoom['q_learning_grid_import_kwh'], lw=1.8, color=LEARNED_COLOR, label='Q-learning')
    axes[1].set_title(f'Grid import from {start_step} to {end_step}')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('kWh')
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.suptitle('Zoomed fixed vs Q-learning comparison')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_learning_and_reward_figure(
    comparison: pd.DataFrame,
    learning_trace: pd.DataFrame,
    output_path: Path,
    start_step: int,
    end_step: int,
) -> None:
    zoom = comparison[(comparison['time_step'] >= start_step) & (comparison['time_step'] <= end_step)].copy()
    if zoom.empty:
        raise ValueError(f'No rows found between time_step={start_step} and time_step={end_step}.')

    window = min(10, len(learning_trace))
    learning_trace = learning_trace.copy()
    learning_trace['rolling_episode_reward'] = learning_trace['episode_reward'].rolling(window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    axes[0].plot(
        learning_trace['episode'],
        learning_trace['rolling_episode_reward'],
        lw=2.0,
        color='#2a9d8f',
        label=f'Rolling reward (window={window})',
    )
    axes[0].set_title('Learning progress')
    axes[0].set_ylabel('Rolling episode reward')
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].plot(zoom['time_step'], zoom['fixed_reward'], '--', lw=1.6, color=FIXED_COLOR, label='Fixed')
    axes[1].plot(zoom['time_step'], zoom['q_learning_reward'], lw=1.8, color=LEARNED_COLOR, label='Q-learning')
    axes[1].set_title(f'Step reward from {start_step} to {end_step}')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Reward')
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=9)

    fig.suptitle('Learning progress and zoomed step reward')
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a time-based comparison for fixed vs Q-learning trajectories.')
    parser.add_argument('--fixed', type=Path, default=DEFAULT_FIXED)
    parser.add_argument('--learned', type=Path, default=DEFAULT_LEARNED)
    parser.add_argument('--learning-trace', type=Path, default=DEFAULT_LEARNING_TRACE)
    parser.add_argument('--output-csv', type=Path, default=DEFAULT_MERGED)
    parser.add_argument('--output-figure', type=Path, default=DEFAULT_FIGURE)
    parser.add_argument('--zoom-output-figure', type=Path, default=DEFAULT_ZOOM_FIGURE)
    parser.add_argument('--combined-output-figure', type=Path, default=DEFAULT_COMBINED_FIGURE)
    parser.add_argument('--zoom-start-step', type=int, default=0)
    parser.add_argument('--zoom-end-step', type=int, default=96)
    args = parser.parse_args()

    comparison = build_comparison_frame(args.fixed, args.learned)
    learning_trace = load_learning_trace(args.learning_trace)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_figure.parent.mkdir(parents=True, exist_ok=True)
    args.zoom_output_figure.parent.mkdir(parents=True, exist_ok=True)
    args.combined_output_figure.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False)
    save_figure(comparison, args.output_figure)
    save_zoom_figure(comparison, args.zoom_output_figure, args.zoom_start_step, args.zoom_end_step)
    save_learning_and_reward_figure(
        comparison,
        learning_trace,
        args.combined_output_figure,
        args.zoom_start_step,
        args.zoom_end_step,
    )

    print(f'Saved merged CSV to: {args.output_csv}')
    print(f'Saved figure to: {args.output_figure}')
    print(f'Saved zoom figure to: {args.zoom_output_figure}')
    print(f'Saved combined figure to: {args.combined_output_figure}')
    print(f'Time steps compared: {len(comparison)}')
    print(f'Total reward advantage (final): {comparison.iloc[-1]["cumulative_reward_delta_q_minus_fixed"]:.3f}')
    print(f'Total grid import savings (final): {comparison.iloc[-1]["cumulative_grid_import_delta_fixed_minus_q"]:.3f} kWh')


if __name__ == '__main__':
    main()