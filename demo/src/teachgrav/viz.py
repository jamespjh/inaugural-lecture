import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger("Teachgrav")

plt.style.use('dark_background')


def visualize(trajectory, output, mode='video', options='dot', duration=30):
    trajectory.data = np.array(trajectory.data)
    # Convert to numpy for visualization
    if trajectory.D != 2:
        raise ValueError(
            "Visualization only supports 2D trajectories, " +
            f"but got D={trajectory.D}")
    if mode == 'video':
        animate(trajectory, output, options, duration)
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

    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_color('dimgrey')

    ax.tick_params(labelsize=8, colors='dimgrey')

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


def animate(trajectory, output, options, duration=30):
    from matplotlib.animation import FuncAnimation, FFMpegWriter
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
    fps = 20
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

    if output:
        writer = FFMpegWriter(fps=fps)
        ani.save(filename=output, writer=writer)
    else:
        plt.show()


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
