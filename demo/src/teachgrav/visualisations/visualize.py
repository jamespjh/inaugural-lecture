import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("Teachgrav")

plt.style.use('dark_background')

_ASPECT_FIGSIZE = {'page': (12.8, 7.2), 'column': (6.4, 7.2)}


def figsize_from_aspect(aspect):
    return _ASPECT_FIGSIZE.get(aspect, _ASPECT_FIGSIZE['column'])


def _equal_limits(mins, maxs, buffer=1.0):
    """Compute equal-aspect axis limits for a plot of any dimensionality.

    All axes share the same half-range so that one data unit spans the same
    distance in every direction.  Works for 2-D (returns xlim, ylim) and
    3-D (returns xlim, ylim, zlim) trajectories alike.

    Args:
        mins: array-like of length D, minimum values in each dimension.
        maxs: array-like of length D, maximum values in each dimension.
        buffer: extra padding added outside the data range on each side.

    Returns:
        Tuple of (lo, hi) pairs, one per dimension.
    """
    centres = (np.asarray(mins) + np.asarray(maxs)) / 2.0
    half_range = np.max(np.asarray(maxs) - np.asarray(mins)) / 2.0 + buffer
    return tuple((c - half_range, c + half_range) for c in centres)


def _apply_axis_style(ax):
    """Apply the standard dark-background axis styling shared by all plots."""
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')
    ax.tick_params(labelsize=8, colors='dimgrey')


def _apply_axis_style_3d(ax):
    """Apply minimal dark-background styling to a 3-D axes object."""
    _apply_axis_style(ax)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('dimgrey')
    ax.yaxis.pane.set_edgecolor('dimgrey')
    ax.zaxis.pane.set_edgecolor('dimgrey')
    ax.grid(False)


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


def _set_line_positions(line, positions, is_3d):
    """Set position data on a 2-D or 3-D matplotlib line artist.

    Args:
        line: a Line2D (2-D) or Line3D (3-D) artist.
        positions: numpy array of shape (n_points, D) where D is 2 or 3.
        is_3d: True when *line* is a 3-D artist.
    """
    if is_3d:
        line.set_data_3d(positions[:, 0], positions[:, 1], positions[:, 2])
    else:
        line.set_data(*positions.T)


def visualize(trajectory, output, mode='video', options='dot', duration=30,
              fps=20, figsize=(6.4, 7.2)):
    trajectory.data = np.array(trajectory.data)
    if mode == 'video':
        animate(trajectory, output, options, duration=duration, fps=fps,
                figsize=figsize)
    else:
        plot(trajectory, output, options, figsize=figsize)


def marker_sizes_from_masses(masses, fig_width_points):
    masses = np.array(masses, dtype=float)
    if np.any(masses <= 0):
        raise ValueError("Masses must be strictly positive for log scaling")

    width_to_min_marker_divisor = 500.0
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
    """Create figure and line artists for a 2-D or 3-D trajectory."""
    if trajectory.D not in (2, 3):
        raise ValueError(
            f"Visualization supports 2D and 3D trajectories only, "
            f"but got D={trajectory.D}")
    is_3d = trajectory.D == 3

    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=figsize)

    mins = np.min(trajectory.positions(), axis=(0, 1))
    maxs = np.max(trajectory.positions(), axis=(0, 1))
    buffer = 1.0

    if is_3d:
        xlim, ylim, zlim = _equal_limits(mins, maxs, buffer)
        ax.set_xlim3d(*xlim)
        ax.set_ylim3d(*ylim)
        ax.set_zlim3d(*zlim)
        _apply_axis_style_3d(ax)
    else:
        xlim, ylim = _equal_limits(mins, maxs, buffer)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _apply_axis_style(ax)

    lines = []
    num_bodies = trajectory.positions().shape[1]
    plot_args = ([], [], []) if is_3d else ([], [])

    if options == 'trail':
        for _ in range(num_bodies):
            line, = ax.plot(*plot_args, color='lemonchiffon')
            lines.append(line)
    elif options == 'dot':
        points_per_inch = 72.0
        fig_width_points = fig.get_figwidth() * points_per_inch
        marker_sizes = marker_sizes_from_masses(
            trajectory.masses,
            fig_width_points)
        for i in range(num_bodies):
            line, = ax.plot(*plot_args, 'o', color='lemonchiffon',
                            markersize=marker_sizes[i])
            lines.append(line)
    else:
        raise ValueError(f"Unknown animation option: {options}")

    return [fig, ax, lines]


def animate(trajectory, output, options, figsize, duration=30, fps=20):
    from matplotlib.animation import FuncAnimation
    is_3d = trajectory.D == 3
    fig, _, lines = axes(trajectory, options, figsize=figsize)

    def init():
        empty = np.empty((0, trajectory.D))
        for line in lines:
            _set_line_positions(line, empty, is_3d)
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
                _set_line_positions(line, trail_positions, is_3d)
            return lines
    elif options == 'dot':
        def update_frame(t):
            interp_pos = get_interpolated_positions(t)
            for i, line in enumerate(lines):
                _set_line_positions(line, interp_pos[i:i + 1, :], is_3d)
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
    is_3d = trajectory.D == 3
    fig, ax, lines = axes(trajectory, options=options, figsize=figsize)
    position = len(trajectory) - 1
    for i, line in enumerate(lines):
        _set_line_positions(
            line, trajectory.positions()[:position, i, :], is_3d)
    if output:
        plt.savefig(output)
    else:
        plt.show()
