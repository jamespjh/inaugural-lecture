from .laws import law, flat_law
from .gp import gp_law
from .system import System, Trajectory, Change
import jax.numpy as np
from diffrax import diffeqsolve, ODETerm, PIDController
import diffrax
import logging
logger = logging.getLogger("Teachgrav")

def fun(t, y, args):
    return flat_law(y, args[0], args[1])

term = ODETerm(fun)

def step_integrator(system: System, method: str, dt: float):
    """Integrate the system forward by one time step using RK4."""
    # Initial state: positions and velocities flattened
    y0 = system.data.flatten()
    # Get the solver corresponding to the method string
    solver = getattr(diffrax, method)
    solve = diffeqsolve(term, solver(), t0=0, t1=dt, dt0=dt, y0=y0, args = [system.masses, system.immobile])
    return solve.ys

def integrate_step(system: System, method: str, dt: float,
                   pars=None) -> System:
    """Take the system state forward using the specified method."""
    if method == 'gp':
        return system + Change(gp_law(system, pars) * dt)
    if method == 'euler':
        return system + Change(law(system) * dt)
    elif method in ['Tsit5', 'Dopri5']:
        # Placeholder for RK4 implementation
        # Get RK integrator from scipy

        new_y = step_integrator(system, method, dt)
        return system.update_flat(new_y)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def integrate_trajectory(system: System, method: str,
                         dt: float, until: float) -> Trajectory:
    """Integrate the system state forward in time for a number of steps."""
    steps = int(until / dt)
    trajectory = Trajectory(system)

    if method == 'euler':
        for step in range(0, steps):
            logger.info(f"Integrating step {step * dt:.2f}/{until:.2f}")
            system = integrate_step(system, method, dt)
            trajectory.append(system.data)

    elif method in ['Tsit5', 'Dopri5']:
        y0 = system.data.flatten()
        solver = getattr(diffrax, method)
        solve = diffeqsolve(term, solver(), t0=0, t1=dt+steps*dt, dt0=dt, y0=y0, args = [system.masses, system.immobile], 
                            saveat = diffrax.SaveAt(ts=np.arange(0, dt * steps + dt, dt)),
                            stepsize_controller = PIDController(rtol=1e-6, atol =1e-9))
        y = solve.ys
        trajectory.data = y.reshape((steps + 1, 2,
                                       len(system), system.D))
    else:
        raise ValueError(f"Unknown integration method: {method}")

    return trajectory
