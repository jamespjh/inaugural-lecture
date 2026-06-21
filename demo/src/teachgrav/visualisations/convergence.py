import traceback
import numpy as np
from ..integrator import integrate_trajectory
from ..laws.pl import PLModel
from ..scenarios import ScenarioFactory
import logging
from .visualize import _apply_axis_style
from .visualize import _save_or_show_animation
from .visualize import _equal_aspect_limits

import matplotlib.pyplot as plt

logger = logging.getLogger("Teachgrav")


def convergence_video(trajectories, output, fps=20, options='trail',
                      ref_trajectory=None, figsize=(6.4, 7.2)):
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
        figsize: (width_inches, height_inches) of the figure (default: half a
                 16:9 projector column, ``(6.4, 7.2)``).
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

    fig, ax = plt.subplots(figsize=figsize)
    xlim, ylim = _equal_aspect_limits(mins, maxs, buffer, figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
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


def generate_stable_keyframes(checkpoints, integrate_power_trajectory):
    """Return stable checkpoint keyframes with their integrated trajectory."""
    stable_keyframes = []
    for ckpt in checkpoints:
        logger.info(
            "Integrating trajectory for checkpoint "
            f"G={ckpt['G']:.4f}, power={ckpt['power']:.4f}…"
        )
        try:
            traj = integrate_power_trajectory(ckpt['G'], ckpt['power'])
        except Exception as exc:  # pragma: no cover
            logger.debug(traceback.format_exc())
            logger.debug(
                f"Skipping unstable checkpoint G={ckpt['G']:.4f}, "
                f"power={ckpt['power']:.4f} (integration failed: {exc!r})."
            )
            continue
        # Skip trajectories that blew up (common during early optimization).
        if not np.isfinite(traj.data).all():
            logger.debug(
                f"Skipping unstable checkpoint G={ckpt['G']:.4f}, "
                f"power={ckpt['power']:.4f} (non-finite trajectory).")
            continue
        stable_keyframes.append({
            'G': float(ckpt['G']),
            'power': float(ckpt['power']),
            'trajectory': traj,
        })
    return stable_keyframes


def generate_upsampled_trajectories(stable_keyframes, target_frames,
                                    integrate_power_trajectory):
    """Build upsampled trajectories via checkpoint interpolation."""
    keyframe_count = len(stable_keyframes)
    keyframe_positions = np.arange(keyframe_count, dtype=float)
    schedule_positions = np.linspace(0, keyframe_count - 1, target_frames)
    keyframe_g = np.array([k['G'] for k in stable_keyframes], dtype=float)
    keyframe_power = np.array(
        [k['power'] for k in stable_keyframes],
        dtype=float,
    )
    scheduled_g = np.interp(schedule_positions, keyframe_positions,
                            keyframe_g)
    scheduled_power = np.interp(schedule_positions, keyframe_positions,
                                keyframe_power)
    scheduled_g[0] = keyframe_g[0]
    scheduled_g[-1] = keyframe_g[-1]
    scheduled_power[0] = keyframe_power[0]
    scheduled_power[-1] = keyframe_power[-1]

    logger.info(
        f"Upsampling convergence frames from {keyframe_count} "
        f"to {target_frames} by parameter interpolation.")

    trajectories = []
    for g_value, power_value in zip(scheduled_g, scheduled_power):
        try:
            traj = integrate_power_trajectory(g_value, power_value)
        except Exception as exc:  # pragma: no cover
            logger.debug(
                "Skipping scheduled frame "
                f"G={g_value:.4f}, power={power_value:.4f} "
                f"(integration failed: {exc!r})."
            )
            continue
        if not np.isfinite(traj.data).all():
            logger.debug(
                "Skipping scheduled frame "
                f"G={g_value:.4f}, power={power_value:.4f} "
                "(non-finite trajectory)."
            )
            continue
        trajectories.append(traj)

    return trajectories


def generate_convergence_video(checkpoints, scenario, output,
                               checkpoint_interval=1,
                               show_true_law=False,
                               seed=None,
                               method='euler',
                               dt=0.01,
                               until=10.0,
                               duration=30,
                               fps=20,
                               scenario_kwargs=None,
                               figsize=(6.4, 7.2)):
    """Generate a convergence video from power-law training checkpoints.

    For each checkpoint, simulate the scatter scenario using given integration
    with the checkpoint parameters and collect the resulting trajectory.
    These trajectories are combined into a single MP4 that shows the fitted
    law converging toward the true law.

    Checkpoints whose trajectories contain non-finite values (e.g. due to
    numerically unstable early-training parameters) are silently skipped.

    Args:
        checkpoints: list of {'G': float, 'power': float} dicts from training.
        scenario: scenario name to visualise.
        output: output path for the generated video.
        checkpoint_interval: use every Nth checkpoint.
        show_true_law: overlay the true-law trajectory if True.
        seed: random seed for the visualisation scenario.
        method: integrator method name.
        dt: integration timestep.
        until: integration end time.
        duration: target output duration in seconds.
        fps: target output frames per second.
        scenario_kwargs: keyword arguments for create_scenario.
    """
    scenario_kwargs = scenario_kwargs or {}

    selected = checkpoints[::checkpoint_interval]
    if not selected:
        logger.warning("No checkpoints to generate convergence video from.")
        return

    target_frames = max(1, int(duration * fps))

    logger.info(
        f"Generating convergence video from {len(selected)} checkpoints "
        f"(interval={checkpoint_interval}), target frames={target_frames}…")

    # Use a fixed seed for the visualisation scenario so all frames show the
    # same initial conditions and only the law changes.
    viz_seed = seed if seed is not None else 42
    viz_factory = ScenarioFactory('numpy', seed=viz_seed)
    system = viz_factory.create_scenario(scenario, **scenario_kwargs)

    def integrate_power_trajectory(g_value, power_value):
        """Integrate a power-law trajectory for a single parameter pair."""
        pl_model = PLModel(factory=viz_factory, G=g_value, power=power_value)
        traj = integrate_trajectory(
            system,
            method=method,
            factory=viz_factory,
            law=pl_model,
            model=pl_model,
            dt=dt,
            until=until)
        traj.data = np.array(traj.data)
        return traj

    stable_keyframes = generate_stable_keyframes(
        selected, integrate_power_trajectory)

    if not stable_keyframes:
        logger.warning(
            "All checkpoint trajectories were numerically unstable; "
            "convergence video not generated.")
        return

    keyframe_count = len(stable_keyframes)
    logger.info(
        f"Using {keyframe_count} stable keyframes for frame scheduling.")
    stable_trajectories = [k['trajectory'] for k in stable_keyframes]

    trajectories = []
    if keyframe_count >= target_frames:
        # Downsample stable keyframes to the requested output frame count.
        sample_positions = np.linspace(0, keyframe_count - 1, target_frames)
        sampled_indices = np.round(sample_positions).astype(int)
        trajectories = [stable_trajectories[idx] for idx in sampled_indices]
    else:
        trajectories = generate_upsampled_trajectories(
            stable_keyframes,
            target_frames,
            integrate_power_trajectory,
        )

    if not trajectories:
        logger.warning(
            "All scheduled convergence trajectories were numerically "
            "unstable; "
            "convergence video not generated.")
        return

    if len(trajectories) < target_frames:
        logger.warning(
            f"Generated {len(trajectories)} convergence frames, below "
            f"target {target_frames} due to unstable integrations.")

    logger.info(
        f"Rendering convergence video with {len(trajectories)} frames.")

    ref_trajectory = None
    if show_true_law:
        try:
            ref_traj = integrate_trajectory(
                system,
                method=method,
                factory=viz_factory,
                law='gravity',
                dt=dt,
                until=until)
            ref_traj.data = np.array(ref_traj.data)
            ref_trajectory = ref_traj
        except Exception as exc:  # pragma: no cover
            logger.warning(
                f"Failed to generate true-law overlay trajectory: {exc!r}. "
                "Continuing without overlay."
            )

    convergence_video(
        trajectories,
        output=output,
        fps=fps,
        ref_trajectory=ref_trajectory,
        figsize=figsize)
    print(f"Convergence video saved to: {output}")
