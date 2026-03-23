from .laws import law
from .system import System, Trajectory
import scipy as sp
import numpy as np


def rk_integrator(system: System, dt: float):
    """Integrate the system forward by one time step using RK4."""
    # Initial state: positions and velocities flattened
    y0 = system.data.flatten()
    # Integrate for one time step
    integrator = sp.integrate.RK45(lambda _, y: system.flat_helper(law)(y),
                                   t0=0, y0=y0, t_bound=dt)
    return integrator


def integrate_step(system: System, method: str, dt: float) -> System:
    """Take the system state forward using the specified method."""
    if method == 'euler':
        return system + law(system)*dt
    elif method == 'rk4':
        # Placeholder for RK4 implementation
        # Get RK integrator from scipy
        integrator = rk_integrator(system, dt)
        integrator.step()
        # Get the new state
        new_y = integrator.y
        return system.update_flat(new_y)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def integrate_trajectory(system: System, method: str,
                         dt: float, until: float) -> Trajectory:
    """Integrate the system state forward in time for a number of steps."""
    steps = int(until / dt)
    trajectory = Trajectory(system, steps)

    if method == 'euler':
        for step in range(0, steps):
            system = integrate_step(system, method, dt)
            trajectory.data[step + 1] = system.data

    elif method == 'rk4':
        y0 = system.data.flatten()

        res = sp.integrate.solve_ivp(lambda _, y: system.flat_helper(law)(y),
                                     (0, dt*steps),
                                     y0, method='RK45', rtol=1e-6, a_tol=1e-6,
                                     t_eval=np.arange(0, dt*steps+dt, dt))
        trajectory.data = res.y.T.reshape((steps+1, 2,
                                           len(system), system.D))
    else:
        raise ValueError(f"Unknown integration method: {method}")

    return trajectory
