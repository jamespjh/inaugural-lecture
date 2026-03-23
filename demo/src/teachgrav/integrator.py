from .laws import law
from .system import System, Trajectory
import scipy as sp


def integrate_step(system: System, method: str, dt: float) -> System:
    """Take the system state forward using the specified method."""
    if method == 'euler':
        return system + law(system)*dt
    elif method == 'rk4':
        # Placeholder for RK4 implementation
        # Get RK integrator from scipy
        N = len(system.positions())

        def wrapper(_, flat_state):
            # Reshape y back into positions and velocities
            data = flat_state.reshape((2, N, 2))
            temp_system = System(data, system.masses, system.immobile)
            return law(temp_system).flatten()

        # Initial state: positions and velocities flattened
        y0 = system.data.flatten()
        # Integrate for one time step
        integrator = sp.integrate.RK45(wrapper, t0=0, y0=y0, t_bound=dt)
        integrator.step()
        # Get the new state
        new_y = integrator.y
        return system.update(new_y.reshape((2, N, 2)))
    else:
        raise ValueError(f"Unknown integration method: {method}")


def integrate_trajectory(system: System, method: str,
                         dt: float, steps: int) -> Trajectory:
    """Integrate the system state forward in time for a number of steps."""
    trajectory = Trajectory(system, steps)

    for step in range(0, steps):
        system = integrate_step(system, method, dt)
        trajectory.data[step + 1] = system.data

    return trajectory
