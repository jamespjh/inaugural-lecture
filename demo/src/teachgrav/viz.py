import matplotlib.pyplot as plt
import numpy as np
    
def visualize(trajectory, output, mode = 'video', options = 'dot'):
    if mode == 'video':
        animate(trajectory, output, options)
    else:
        plot(trajectory, output)

def axes(trajectory, options):
    # Animate the trajectory
    fig, ax = plt.subplots()

    # Get the limits of the plot based on the trajectory data
    x_min, x_max = np.min(trajectory.positions[:,:, 0]), np.max(trajectory.positions[:,:, 0])
    y_min, y_max = np.min(trajectory.positions[:,:, 1]), np.max(trajectory.positions[:,:, 1])
    
    buffer = 1.0

    ax.set_xlim(x_min - buffer, x_max + buffer)
    ax.set_ylim(y_min - buffer, y_max + buffer)


    if options == 'trail':
        line, = ax.plot([], [], 'b-')   
    elif options == 'dot':
        line, = ax.plot([], [], 'bo')
    else:
        raise ValueError(f"Unknown animation option: {options}")
    return [fig, ax, line]

def animate(trajectory, output, options):
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    fig, _, line = axes(trajectory, options)

    def init():
        line.set_data([], [])
        return line,

    if options == 'trail':
        def animate(position):
            line.set_data(trajectory.positions[:position, 0], trajectory.positions[:position, 1])
            return []
    elif options == 'dot':
        def animate(position):
            line.set_data([trajectory.positions[position, 0]], [trajectory.positions[position, 1]])
            return []
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
    

def plot(trajectory, output):
    fig, ax, _ = axes(trajectory, options=None)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-')
    if output:
        plt.savefig(output)
    else:
        plt.show()