import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger("Teachgrav")

plt.style.use('dark_background')


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
              fps=20):
    trajectory.data = np.array(trajectory.data)
    # Convert to numpy for visualization
    if trajectory.D != 2:
        raise ValueError(
            "Visualization only supports 2D trajectories, " +
            f"but got D={trajectory.D}")
    if mode == 'video':
        animate(trajectory, output, options, duration, fps=fps)
    else:
        plot(trajectory, output, options)


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


def axes(trajectory, options):
    # Animate the trajectory
    fig, ax = plt.subplots()

    mins = np.min(trajectory.positions(), axis=(0, 1))
    maxs = np.max(trajectory.positions(), axis=(0, 1))

    buffer = 1.0

    ax.set_xlim(mins[0] - buffer, maxs[0] + buffer)
    ax.set_ylim(mins[1] - buffer, maxs[1] + buffer)

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


def animate(trajectory, output, options, duration=30, fps=20):
    from matplotlib.animation import FuncAnimation
    fig, _, lines = axes(trajectory, options)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    if options == 'trail':
        def animate(position):
            for i, line in enumerate(lines):
                line.set_data(
                    *trajectory.positions()[:position, i, :].T)
            return lines
    elif options == 'dot':
        def animate(position):
            for i, line in enumerate(lines):
                line.set_data(*trajectory.positions()[position - 1:position,
                                                      i, :].T)
            return lines
    else:
        raise ValueError(f"Unknown animation option: {options}")

    steps = len(trajectory)
    interval = int(1000 / fps)  # milliseconds per frame
    number_of_frames = max(1, int(duration * fps))

    steps_for_viz = np.linspace(1, steps, number_of_frames, dtype=int)
    logger.info(
        f"Animating trajectory with {steps} steps, " +
        f"visualizing {number_of_frames} frames at steps {steps_for_viz}")

    ani = FuncAnimation(fig,
                        animate,
                        init_func=init,
                        frames=steps_for_viz,
                        interval=interval,
                        blit=False)

    _save_or_show_animation(ani, output, fps)


def plot(trajectory, output, options):
    fig, ax, lines = axes(trajectory, options=options)
    position = len(trajectory) - 1
    for i, line in enumerate(lines):
        line.set_data(
            *trajectory.positions()[:position, i, :].T)
    if output:
        plt.savefig(output)
    else:
        plt.show()


def convergence_video(trajectories, output, fps=20, options='trail',
                      ref_trajectory=None):
    """Create a video showing trajectory convergence across training steps.

    Each frame shows the full trail of the simulated trajectory produced by
    one training checkpoint.  An optional reference trajectory (e.g. the true
    law) is overlaid in a contrasting colour so the viewer can see the fitted
    law converging toward the ground truth.

    Args:
        trajectories: sequence of Trajectory objects, one per checkpoint.
        output: output file path for the MP4 (or None to show interactively).
        fps: frames per second (default: 20).
        options: visualisation style; currently only ``'trail'`` is supported.
        ref_trajectory: optional Trajectory to overlay on every frame in a
                        different colour (e.g. the true-law trajectory).
    """
    from matplotlib.animation import FuncAnimation

    # ------------------------------------------------------------------ #
    # Determine global axis bounds so all frames share the same scale.    #
    # ------------------------------------------------------------------ #
    if ref_trajectory is not None:
        # Keep the frame fixed to the true-law extent so fitted trajectories
        # are judged against a consistent reference box.
        all_flat = np.array(ref_trajectory.data)[:, 0, :, :].reshape(-1, 2)
    else:
        # Fall back to the best fitted trajectory (the final checkpoint).
        all_flat = np.array(trajectories[-1].data)[:, 0, :, :].reshape(-1, 2)

    finite = all_flat[np.isfinite(all_flat).all(axis=1)]
    if len(finite) == 0:
        finite = np.array([[-10.0, -10.0], [10.0, 10.0]])
    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    buffer = 1.0

    fig, ax = plt.subplots()
    ax.set_xlim(mins[0] - buffer, maxs[0] + buffer)
    ax.set_ylim(mins[1] - buffer, maxs[1] + buffer)
    _apply_axis_style(ax)

    num_bodies = trajectories[0].positions().shape[1]

    # Lines for the fitted-law trajectory.
    fitted_lines = []
    for _ in range(num_bodies):
        line, = ax.plot([], [], color='lemonchiffon', alpha=0.9)
        fitted_lines.append(line)

    # Lines for the reference trajectory (optional).
    ref_lines = []
    if ref_trajectory is not None:
        for _ in range(num_bodies):
            line, = ax.plot([], [], color='dodgerblue',
                            alpha=0.7, linestyle='--')
            ref_lines.append(line)

    all_lines = fitted_lines + ref_lines

    # Pre-compute reference positions so they are not repeated every frame.
    if ref_trajectory is not None:
        ref_positions = np.array(ref_trajectory.data)[:, 0, :, :]
    else:
        ref_positions = None

    def init():
        for line in all_lines:
            line.set_data([], [])
        return all_lines

    def update(frame_idx):
        traj = trajectories[frame_idx]
        positions = np.array(traj.data)[:, 0, :, :]
        for i, line in enumerate(fitted_lines):
            line.set_data(positions[:, i, 0], positions[:, i, 1])
        if ref_positions is not None:
            for i, line in enumerate(ref_lines):
                line.set_data(ref_positions[:, i, 0], ref_positions[:, i, 1])
        return all_lines

    ani = FuncAnimation(
        fig, update, init_func=init,
        frames=len(trajectories),
        interval=int(1000 / fps),
        blit=False)

    _save_or_show_animation(
        ani, output, fps,
        log_msg=f"Convergence video written to {output}")

    plt.close(fig)
