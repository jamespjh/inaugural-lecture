"""CLI entry point: read a benchmark CSV and produce a log-log PNG figure.

The CSV is produced by ``generate-figures --benchmark``.  Its first column is
the row parameter (n-bodies) and each subsequent column is an engine label.

Usage::

    plot-benchmark <input.csv> <output.png>
"""

import argparse
import csv
import math


# Fixed colour palette keyed by engine name.
ENGINE_COLORS = {
    'numpy':     '#4878d0',
    'jax-cpu':   '#ee854a',
    'jax-gpu':   '#6acc65',
    'jax-metal': '#d65f5f',
    'mlx-cpu':   '#956cb4',
    'mlx-gpu':   '#8c613c',
    'cupy':      '#dc7ec0',
    'torch-cpu': '#797979',
    'torch-gpu': '#d5bb67',
    'torch-mps': '#82c6e2',
}


def plot_benchmark_csv(csv_path, output_path):
    """Read *csv_path* and save a log-log figure to *output_path*."""
    import matplotlib.pyplot as plt

    with open(csv_path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        rows = list(reader)

    row_label = headers[0]
    engine_labels = headers[1:]

    x_vals = [float(row[0]) for row in rows]
    engine_times = {}
    for col_idx, engine in enumerate(engine_labels, start=1):
        times = []
        for row in rows:
            val = row[col_idx]
            try:
                t = float(val)
            except (ValueError, IndexError):
                t = float('nan')
            times.append(t)
        engine_times[engine] = times

    fig, ax = plt.subplots(figsize=(8, 5))

    for engine, times in engine_times.items():
        # Filter out NaN pairs so loglog doesn't choke
        pairs = [(x, t) for x, t in zip(x_vals, times)
                 if not math.isnan(t) and t > 0]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        color = ENGINE_COLORS.get(engine)
        ax.loglog(xs, ys, 'o-', label=engine, color=color,
                  linewidth=2, markersize=5)

    ax.set_xlabel(row_label.replace('-', ' ').title(), fontsize=13)
    ax.set_ylabel('Time per simulation step (s)', fontsize=13)
    ax.set_title('Gravity simulation performance by engine', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    if x_vals:
        ax.set_xlim(min(x_vals) * 0.8, max(x_vals) * 1.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Plot a benchmark CSV produced by generate-figures --benchmark'))
    parser.add_argument('csv_file', help='Input CSV file')
    parser.add_argument(
        'output',
        help='Output PNG (or other matplotlib-supported format)')
    args = parser.parse_args()
    plot_benchmark_csv(args.csv_file, args.output)


if __name__ == '__main__':
    main()
