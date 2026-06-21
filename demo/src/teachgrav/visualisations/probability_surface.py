import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from .visualize import _save_or_show_animation

logger = logging.getLogger("Teachgrav")

plt.style.use("dark_background")


def _resolve_surface_axes(likelihoods, G_values=None, power_values=None):
    """Resolve default parameter axes for a likelihood grid."""
    N_G, N_power = likelihoods.shape
    if G_values is None:
        G_values = np.linspace(-5.0, 5.0, N_G)
    if power_values is None:
        power_values = np.linspace(-5.0, 5.0, N_power)
    return G_values, power_values


def _to_log_likelihoods(likelihoods):
    """Convert probability masses to log-space with finite values."""
    return np.log(np.clip(likelihoods, np.finfo(float).tiny, None))


def _posterior_stats(likelihoods, G_values, power_values):
    """Return posterior means/stds for G and power from a 2-D grid."""
    weights = np.asarray(likelihoods, dtype=float)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        weights = np.ones_like(weights, dtype=float)
        total = float(np.sum(weights))
    weights = weights / total

    G_grid = np.asarray(G_values, dtype=float)[:, np.newaxis]
    power_grid = np.asarray(power_values, dtype=float)[np.newaxis, :]

    mean_G = float(np.sum(weights * G_grid))
    mean_power = float(np.sum(weights * power_grid))
    var_G = float(np.sum(weights * (G_grid - mean_G) ** 2))
    var_power = float(np.sum(weights * (power_grid - mean_power) ** 2))
    std_G = float(np.sqrt(max(var_G, 0.0)))
    std_power = float(np.sqrt(max(var_power, 0.0)))

    return mean_G, std_G, mean_power, std_power


def _draw_stats_overlay(ax, mean_G, std_G, mean_power, std_power,
                        frame_label=None):
    """Draw mean marker and top overlay text for posterior statistics."""
    ellipse = Ellipse(
        (mean_power, mean_G),
        width=max(2.0 * std_power, 1e-6),
        height=max(2.0 * std_G, 1e-6),
        angle=0.0,
        fill=False,
        edgecolor="cyan",
        linewidth=2.0,
        zorder=6,
    )
    ax.add_patch(ellipse)

    parts = []
    if frame_label:
        parts.append(frame_label)
    parts.append(f"G {mean_G:.2f} ± {std_G:.2f}")
    parts.append(f"n {mean_power:.2f} ± {std_power:.2f}")
    text = "   ".join(parts)
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.35, "pad": 3},
    )


def _draw_log_probability_surface(ax, log_likelihoods, G_values, power_values,
                                  vmin=None, vmax=None):
    """Draw one log-probability heatmap frame and return the image object."""
    img = ax.pcolormesh(
        power_values,
        G_values,
        log_likelihoods,
        cmap="inferno",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Power (n)")
    ax.set_ylabel("G")
    ax.set_title("Log posterior probability over (G, n)")
    return img


def plot_probability_surface(
    likelihoods,
    outfile,
    G_values=None,
    power_values=None,
    figsize=(6.4, 6.4),
):
    """Save a heatmap of a 2-D log-probability distribution over (G, n).

    Args:
        likelihoods: 2-D numpy array of shape ``(N_G, N_power)`` containing
                 posterior probability masses for each grid point.
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
    G_values, power_values = _resolve_surface_axes(
        likelihoods,
        G_values=G_values,
        power_values=power_values,
    )
    log_likelihoods = _to_log_likelihoods(likelihoods)

    fig, ax = plt.subplots(figsize=figsize)
    img = _draw_log_probability_surface(
        ax,
        log_likelihoods,
        G_values,
        power_values,
    )
    mean_G, std_G, mean_power, std_power = _posterior_stats(
        likelihoods,
        G_values,
        power_values,
    )
    _draw_stats_overlay(ax, mean_G, std_G, mean_power, std_power)
    fig.colorbar(img, ax=ax, label="Log posterior probability")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    logger.info(f"Saved log-probability surface plot to {outfile}")


def probability_surface_video(
    likelihood_grids,
    output,
    G_values=None,
    power_values=None,
    fps=20,
    duration=None,
    figsize=(6.4, 6.4),
):
    """Create an MP4 showing posterior log-probability surface evolution."""
    from matplotlib.animation import FuncAnimation

    if len(likelihood_grids) == 0:
        raise ValueError("likelihood_grids must contain at least one frame")

    first = likelihood_grids[0]
    G_values, power_values = _resolve_surface_axes(
        first,
        G_values=G_values,
        power_values=power_values,
    )
    likelihood_grids = [np.asarray(grid) for grid in likelihood_grids]
    frame_positions = np.arange(len(likelihood_grids), dtype=float)
    render_likelihood_grids = likelihood_grids
    if duration is not None and fps > 0:
        target_frames = max(1, int(round(duration * fps)))
        frame_positions = np.linspace(
            0,
            len(likelihood_grids) - 1,
            num=target_frames,
        )
        if target_frames > len(likelihood_grids):
            # Upsample by linear interpolation between stored posterior grids.
            interpolated = []
            for pos in frame_positions:
                lo = int(np.floor(pos))
                hi = int(np.ceil(pos))
                if lo == hi:
                    interpolated.append(likelihood_grids[lo])
                else:
                    alpha = pos - lo
                    interpolated.append(
                        (1.0 - alpha) * likelihood_grids[lo]
                        + alpha * likelihood_grids[hi]
                    )
            render_likelihood_grids = interpolated
        else:
            # Downsample by nearest stored posterior grid.
            frame_indices = frame_positions.astype(int)
            render_likelihood_grids = [
                likelihood_grids[idx] for idx in frame_indices
            ]

    log_grids = [_to_log_likelihoods(grid) for grid in render_likelihood_grids]
    frame_stats = [
        _posterior_stats(grid, G_values, power_values)
        for grid in render_likelihood_grids
    ]
    vmin = min(float(np.min(grid)) for grid in log_grids)
    vmax = max(float(np.max(grid)) for grid in log_grids)

    fig, ax = plt.subplots(figsize=figsize)
    img = _draw_log_probability_surface(
        ax,
        log_grids[0],
        G_values,
        power_values,
        vmin=vmin,
        vmax=vmax,
    )
    _draw_stats_overlay(
        ax,
        frame_stats[0][0],
        frame_stats[0][1],
        frame_stats[0][2],
        frame_stats[0][3],
        frame_label=f"Step index {int(round(frame_positions[0]))}",
    )
    fig.colorbar(img, ax=ax, label="Log posterior probability")

    def update(frame_idx):
        ax.clear()
        frame_img = _draw_log_probability_surface(
            ax,
            log_grids[frame_idx],
            G_values,
            power_values,
            vmin=vmin,
            vmax=vmax,
        )
        _draw_stats_overlay(
            ax,
            frame_stats[frame_idx][0],
            frame_stats[frame_idx][1],
            frame_stats[frame_idx][2],
            frame_stats[frame_idx][3],
            frame_label=f"Step index {int(round(frame_positions[frame_idx]))}",
        )
        return [frame_img]

    ani = FuncAnimation(
        fig,
        update,
        frames=len(log_grids),
        interval=int(1000 / fps) if fps > 0 else 50,
        blit=False,
    )
    _save_or_show_animation(
        ani,
        output,
        fps,
        log_msg=f"Saved log-probability surface video to {output}",
    )
    plt.close(fig)
