from .laws import law
from .system import System, Trajectory


def integrate_step(system: System, method: str, dt: float) -> System:
    """Take the system state forward using the specified method."""
    if method == 'euler':
        derivatives = law(system)
        new_positions = system.positions + derivatives[0, :, :] * dt
        new_velocities = system.velocities + derivatives[1, :, :] * dt
        return system.update(new_positions, new_velocities)
    elif method == 'rk4':
        # Placeholder for RK4 implementation
        return system
    else:
        raise ValueError(f"Unknown integration method: {method}")


def integrate_trajectory(system: System, method: str,
                         dt: float, steps: int) -> Trajectory:
    """Integrate the system state forward in time for a number of steps."""
    trajectory = Trajectory(system, steps)

    for step in range(0, steps):
        system = integrate_step(system, method, dt)
        trajectory.positions[step+1] = system.positions
        trajectory.velocities[step+1] = system.velocities

    return trajectory
