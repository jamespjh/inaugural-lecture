from .laws import law, flat_law
from .system import System, Trajectory
from .gp import gp_law
import scipy as sp
import numpy as np
import logging
logger = logging.getLogger("Teachgrav")


def rk_integrator(system: System, dt: float):
    """Integrate the system forward by one time step using RK4."""
    # Initial state: positions and velocities flattened
    y0 = system.data.flatten()

    def fun(t, y):
        logger.info(f"Integrating step {t:.2f}")
        return flat_law(y, system.masses, system.immobile)

    # Integrate for one time step
    integrator = sp.integrate.RK45(fun,
                                   t0=0, y0=y0, t_bound=dt)
    return integrator


def integrate_step(system: System, method: str, dt: float,
                   pars = None) -> System:
    """Take the system state forward using the specified method."""
    if method == 'gp':
        return system + gp_law(system, pars) * dt
    if method == 'euler':
        return system + law(system) * dt
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
            logger.info(f"Integrating step {step * dt:.2f}/{until:.2f}")
            system = integrate_step(system, method, dt)
            trajectory.data[step + 1] = system.data

    elif method in ['RK45', 'RK23', 'LSODA', 'DOP853', 'Radau', 'BDF']:
        y0 = system.data.flatten()

        def fun(t, y):
            logger.info(f"Integrating step {t:.2f}/{until:.2f}")
            return flat_law(y, system.masses, system.immobile)

        res = sp.integrate.solve_ivp(fun,
                                     (0, dt * steps),
                                     y0, method=method, rtol=1e-6,
                                     t_eval=np.arange(0, dt * steps + dt, dt))
        trajectory.data = res.y.T.reshape((steps + 1, 2,
                                           len(system), system.D))
    else:
        raise ValueError(f"Unknown integration method: {method}")

    return trajectory
