import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("Teachgrav")

plt.style.use('dark_background')

_ASPECT_FIGSIZE = {'page': (12.8, 7.2), 'column': (6.4, 7.2)}
# Default side length (inches) for each subplot cell in grid_plot.
_GRID_CELL_SIZE = 2.5


def figsize_from_aspect(aspect):
    return _ASPECT_FIGSIZE.get(aspect, _ASPECT_FIGSIZE['column'])


def _equal_aspect_limits(mins, maxs, buffer=1.0, figsize=None):
    """Compute axis limits that preserve physical aspect ratio.

    The y axis is sized to contain all data (plus buffer).  The x axis is
    then scaled proportionally to the figure width-to-height ratio so that
    one data unit spans the same number of pixels in both directions.

    Args:
        mins: array-like of length 2, minimum [x, y] values in the data.
        maxs: array-like of length 2, maximum [x, y] values in the data.
        buffer: extra padding added outside the data range on each side.
        figsize: (width_inches, height_inches) of the figure.  When None a
            square figure is assumed (aspect ratio 1).

    Returns:
        Tuple (xlim, ylim) where each is a (lo, hi) tuple.
    """
    x_mid = (mins[0] + maxs[0]) / 2.0
    y_mid = (mins[1] + maxs[1]) / 2.0
    x_range = maxs[0] - mins[0]
    y_range = maxs[1] - mins[1]

    if figsize is not None:
        fig_w, fig_h = figsize
        aspect = fig_w / fig_h if fig_h > 0 else 1.0
    else:
        aspect = 1.0

    # Choose y_half large enough to contain y data and, via the aspect ratio,
    # x data as well.  x_half is then derived so that one data unit spans the
    # same number of pixels on both axes.
    y_half = max(y_range / 2.0 + buffer,
                 (x_range / 2.0 + buffer) / aspect)
    x_half = y_half * aspect
    xlim = (x_mid - x_half, x_mid + x_half)
    ylim = (y_mid - y_half, y_mid + y_half)
    return xlim, ylim


def _apply_axis_style(ax):
    """Apply the standard dark-background axis styling shared by all plots."""
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.tick_params(labelsize=8, colors='dimgrey')


def _save_or_show_animation(ani, output, fps, log_msg=None):
    """Save *ani* to *output* with FFMpeg, or show interactively if None."""
    from matplotlib.animation import FFMpegWriter
    if output:
        writer = FFMpegWriter(fps=fps)
        ani.save(filename=output, writer=writer)
        if log_msg:
            logger.info(log_msg)
    else:
        plt.show()


def visualize(trajectory, output, mode='video', options='dot', duration=30,
              fps=20, figsize=(6.4, 7.2)):
    trajectory.data = np.array(trajectory.data)
    # Convert to numpy for visualization
    if trajectory.D != 2:
        raise ValueError(
            "Visualization only supports 2D trajectories, " +
            f"but got D={trajectory.D}")
    if mode == 'video':
        animate(trajectory, output, options, duration=duration, fps=fps,
                figsize=figsize)
    else:
        plot(trajectory, output, options, figsize=figsize)


def marker_sizes_from_masses(masses, fig_width_points):
    masses = np.array(masses, dtype=float)
    if np.any(masses <= 0):
        raise ValueError("Masses must be strictly positive for log scaling")

    width_to_min_marker_divisor = 200.0
    width_to_max_marker_divisor = 50.0
    min_marker_size = fig_width_points / width_to_min_marker_divisor
    max_marker_size = fig_width_points / width_to_max_marker_divisor

    log_masses = np.log(masses)
    log_min = np.min(log_masses)
    log_max = np.max(log_masses)

    if np.isclose(log_min, log_max):
        return np.full(masses.shape, min_marker_size)

    normalized = (log_masses - log_min) / (log_max - log_min)
    return min_marker_size + normalized * (max_marker_size - min_marker_size)


def axes(trajectory, options, figsize):
    # Animate the trajectory
    fig, ax = plt.subplots(figsize=figsize)

    mins = np.min(trajectory.positions(), axis=(0, 1))
    maxs = np.max(trajectory.positions(), axis=(0, 1))

    buffer = 1.0

    xlim, ylim = _equal_aspect_limits(mins, maxs, buffer, figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    _apply_axis_style(ax)

    # One line or dot per body, with the option to show trails or just current
    # positions

    lines = []
    num_bodies = trajectory.positions().shape[1]

    if options == 'trail':
        for _ in range(num_bodies):
            line, = ax.plot([], [], color='lemonchiffon')
            lines.append(line)
    elif options == 'dot':
        points_per_inch = 72.0
        fig_width_points = fig.get_figwidth() * points_per_inch
        marker_sizes = marker_sizes_from_masses(
            trajectory.masses,
            fig_width_points)
        for i in range(num_bodies):
            line, = ax.plot([], [], 'o', color='lemonchiffon',
                            markersize=marker_sizes[i])
            lines.append(line)
    else:
        raise ValueError(f"Unknown animation option: {options}")

    return [fig, ax, lines]


def animate(trajectory, output, options, figsize, duration=30, fps=20):
    from matplotlib.animation import FuncAnimation
    fig, _, lines = axes(trajectory, options, figsize=figsize)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    # Get trajectory data and compute time values
    positions = trajectory.positions()
    steps = len(trajectory)

    # Generate time values for each output frame
    number_of_frames = max(1, int(duration * fps))
    frame_times = np.linspace(0, steps - 1, number_of_frames)

    def get_interpolated_positions(t):
        """Interpolate trajectory positions at time t."""
        # Find the two steps surrounding this time
        step_idx = int(np.floor(t))
        step_idx = np.clip(step_idx, 0, steps - 2)
        alpha = t - step_idx

        # Linear interpolation between step_idx and step_idx+1
        pos_current = positions[step_idx]
        pos_next = positions[step_idx + 1]
        return pos_current * (1 - alpha) + pos_next * alpha

    if options == 'trail':
        def update_frame(t):
            # For trail, show all positions up to the interpolated time
            step_idx = int(np.floor(t))
            step_idx = np.clip(step_idx, 0, steps - 1)
            interp_positions = get_interpolated_positions(t)

            for i, line in enumerate(lines):
                if step_idx == 0:
                    trail_positions = positions[:1, i, :]
                else:
                    trail_positions = positions[:step_idx + 1, i, :]
                    # Add interpolated final point
                    if step_idx < steps - 1:
                        trail_positions = np.vstack(
                            [trail_positions, [interp_positions[i]]])
                line.set_data(*trail_positions.T)
            return lines
    elif options == 'dot':
        def update_frame(t):
            interp_pos = get_interpolated_positions(t)
            for i, line in enumerate(lines):
                line.set_data(*interp_pos[i:i + 1, :].T)
            return lines
    else:
        raise ValueError(f"Unknown animation option: {options}")

    interval = int(1000 / fps)  # milliseconds per frame
    logger.info(
        f"Animating trajectory with {steps} steps, " +
        f"visualizing {number_of_frames} frames with linear position "
        f"interpolation")

    ani = FuncAnimation(fig,
                        update_frame,
                        init_func=init,
                        frames=frame_times,
                        interval=interval,
                        blit=False)

    _save_or_show_animation(ani, output, fps)


def plot(trajectory, output, options, figsize):
    fig, ax, lines = axes(trajectory, options=options, figsize=figsize)
    position = len(trajectory) - 1
    for i, line in enumerate(lines):
        line.set_data(
            *trajectory.positions()[:position, i, :].T)
    if output:
        plt.savefig(output)
    else:
        plt.show()


def grid_plot(trajectories, grid_size, output, options='trail'):
    """Create an N×N grid of trail plots from multiple trajectories.

    Each cell in the grid shows the full-trail visualisation of one trajectory
    (typically a different sample from a stochastic scenario generator).

    Args:
        trajectories: sequence of Trajectory objects; the first
            ``grid_size ** 2`` entries are used.
        grid_size: N; produces an N×N subplot grid.
        output: file path for the saved image (PNG/SVG/PDF) or None to show
            the figure interactively.
        options: visualisation style – ``'trail'`` (default) draws a line for
            each body's full path; ``'dot'`` draws only the final position.
    """
    n = grid_size
    cell_size = _GRID_CELL_SIZE
    fig, ax_grid = plt.subplots(n, n, figsize=(n * cell_size, n * cell_size))

    # matplotlib.pyplot.subplots returns different shapes depending on n:
    #   n=1 → a single Axes object (0-D)
    #   n>1, single row/col → a 1-D array
    #   n>1 → a 2-D array
    # Normalise to a 2-D array so indexing is always [row, col].
    if n == 1:
        ax_grid = np.array([[ax_grid]])
    elif n > 1 and ax_grid.ndim == 1:
        ax_grid = ax_grid[np.newaxis, :]

    for idx, traj in enumerate(trajectories[:n * n]):
        row = idx // n
        col = idx % n
        ax = ax_grid[row, col]

        positions = np.array(traj.positions())
        num_bodies = positions.shape[1]

        mins = np.min(positions, axis=(0, 1))
        maxs = np.max(positions, axis=(0, 1))
        xlim, ylim = _equal_aspect_limits(
            mins, maxs, buffer=0.5, figsize=(cell_size, cell_size))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _apply_axis_style(ax)
        ax.set_xticks([])
        ax.set_yticks([])

        if options == 'trail':
            for i in range(num_bodies):
                ax.plot(positions[:, i, 0], positions[:, i, 1],
                        color='lemonchiffon', linewidth=0.5)
        elif options == 'dot':
            points_per_inch = 72.0
            fig_width_points = cell_size * points_per_inch
            marker_sizes = marker_sizes_from_masses(
                traj.masses, fig_width_points)
            for i in range(num_bodies):
                ax.plot(positions[-1, i, 0], positions[-1, i, 1],
                        'o', color='lemonchiffon',
                        markersize=marker_sizes[i])
        else:
            raise ValueError(f"Unknown grid_plot option: {options}")

    plt.tight_layout(pad=0.2)
    if output:
        plt.savefig(output, dpi=100, bbox_inches='tight')
        logger.info(f"Grid plot written to {output}")
    else:
        plt.show()
    plt.close(fig)
