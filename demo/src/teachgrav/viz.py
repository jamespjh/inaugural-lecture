import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger("Teachgrav")


def visualize(trajectory, output, mode='video', options='dot'):
    trajectory.data = np.array(trajectory.data)  # Convert to numpy for visualization
    if trajectory.D != 2:
        raise ValueError(
            "Visualization only supports 2D trajectories, " +
            f"but got D={trajectory.D}")
    if mode == 'video':
        animate(trajectory, output, options)
    else:
        plot(trajectory, output, options)


def axes(trajectory, options):
    # Animate the trajectory
    fig, ax = plt.subplots()

    mins = np.min(trajectory.positions(), axis=(0, 1))
    maxs = np.max(trajectory.positions(), axis=(0, 1))

    buffer = 1.0

    ax.set_xlim(mins[0] - buffer, maxs[0] + buffer)
    ax.set_ylim(mins[1] - buffer, maxs[1] + buffer)

    # One line or dot per body, with the option to show trails or just current
    # positions

    lines = []
    num_bodies = trajectory.positions().shape[1]

    if options == 'trail':
        for _ in range(num_bodies):
            line, = ax.plot([], [], 'b-')
            lines.append(line)
    elif options == 'dot':
        for _ in range(num_bodies):
            line, = ax.plot([], [], 'bo')
            lines.append(line)
    else:
        raise ValueError(f"Unknown animation option: {options}")

    return [fig, ax, lines]


def animate(trajectory, output, options):
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
    time = 30 * 1000  # miliseconds
    interval = 200  # miliseconds per frame
    number_of_frames = time // interval

    steps_for_viz = np.linspace(0, steps - 1, number_of_frames, dtype=int)
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
        writer = FFMpegWriter()
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
