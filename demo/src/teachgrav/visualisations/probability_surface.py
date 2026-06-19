import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("Teachgrav")

plt.style.use('dark_background')


def plot_probability_surface(likelihoods, outfile,
                              G_values=None,
                              power_values=None,
                              figsize=(6.4, 6.4)):
    """Save a heatmap of a 2-D probability distribution over (G, power) space.

    Args:
        likelihoods: 2-D numpy array of shape ``(N_G, N_power)`` containing
                     the (unnormalised) posterior probability for each grid
                     point.
        outfile: path to write the PNG image.
        G_values: 1-D array of G grid values.  When ``None`` a uniform grid
                  over ``[-5, 5]`` matching the array's first dimension is
                  used.
        power_values: 1-D array of power (n) grid values.  When ``None`` a
                      uniform grid over ``[-5, 5]`` matching the array's
                      second dimension is used.
        figsize: ``(width_inches, height_inches)`` of the saved figure
                 (default: ``(6.4, 6.4)``).
    """
    N_G, N_power = likelihoods.shape
    if G_values is None:
        G_values = np.linspace(-5.0, 5.0, N_G)
    if power_values is None:
        power_values = np.linspace(-5.0, 5.0, N_power)

    fig, ax = plt.subplots(figsize=figsize)
    img = ax.pcolormesh(
        power_values, G_values, likelihoods,
        cmap='inferno', shading='auto')
    fig.colorbar(img, ax=ax, label='Posterior probability')
    ax.set_xlabel('Power (n)')
    ax.set_ylabel('G')
    ax.set_title('Posterior probability distribution over (G, n)')

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Saved probability surface plot to {outfile}")
