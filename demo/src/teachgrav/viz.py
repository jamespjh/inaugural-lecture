import matplotlib.pyplot as plt
import numpy as np


def visualize(trajectory, output, mode='video', options='dot'):
    if mode == 'video':
        animate(trajectory, output, options)
    else:
        plot(trajectory, output, options)


def axes(trajectory, options):
    # Animate the trajectory
    fig, ax = plt.subplots()

    # Get the limits of the plot based on the trajectory data
    x_min, x_max = np.min(trajectory.positions[:, :, 0]), np.max(
        trajectory.positions[:, :, 0])
    y_min, y_max = np.min(trajectory.positions[:, :, 1]), np.max(
        trajectory.positions[:, :, 1])

    buffer = 1.0

    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)

    # One line or dot per body, with the option to show trails or just current
    # positions

    lines = []
    num_bodies = trajectory.positions.shape[1]

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
                    trajectory.positions[:position, i, 0],
                    trajectory.positions[:position, i, 1])
            return lines
    elif options == 'dot':
        def animate(position):
            for i, line in enumerate(lines):
                line.set_data([trajectory.positions[position, i, 0]],
                              [trajectory.positions[position, i, 1]])
            return lines
    else:
        raise ValueError(f"Unknown animation option: {options}")

    ani = FuncAnimation(fig,
                        animate,
                        init_func=init,
                        frames=len(trajectory),
                        interval=100,
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
            trajectory.positions[:position, i, 0],
            trajectory.positions[:position, i, 1])
    if output:
        plt.savefig(output)
    else:
        plt.show()
