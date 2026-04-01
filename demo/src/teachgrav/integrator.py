from .laws import law, flat_law
from .gp import gp_law
from .system import System, Trajectory, Change
import jax.numpy as np
from jax import jit
import jax
from diffrax import diffeqsolve, ODETerm, PIDController
import diffrax
import logging
import equinox as eqx
from functools import partial
logger = logging.getLogger("Teachgrav")
jax.config.update("jax_platforms", 'cpu')

diffrax_methods = ['Tsit5', 'Dopri5', 'Kvaerno5']
scipy_methods = ['RK45', 'LSODA']

def solve_diffrax(method, t1, dt, y0, saveat, masses, immobile):
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        solve = diffrax_solve(method, t1, dt, y0, 
            saveat, masses, immobile)
    return solve.ys

def solve_numpy(method, t1, dt, y0, saveat, masses, immobile):
    from scipy.integrate import solve_ivp
    def fun(t, y):
        return flat_law(y, masses, immobile)
    solve = solve_ivp(fun, (0, t1), y0, method=method, t_eval=saveat, rtol=1e-6)
    return solve.y.T

@partial(jit, static_argnames=['method'])
@eqx.debug.assert_max_traces(max_traces=3)
def diffrax_solve(method, t1, dt, y0, saveat, masses, immobile):
    def fun(t, y, args):
        return flat_law(y, args[0], args[1])

    term = ODETerm(fun)
    solver = getattr(diffrax, method)
    with jax.transfer_guard('log'):
        solve = diffeqsolve(term, solver(), t0=0, t1=t1, dt0=dt, y0=y0, args=(masses, immobile), 
                                saveat = diffrax.SaveAt(ts=saveat),
                                stepsize_controller = PIDController(rtol=1e-6, atol =1e-6))
    return solve

def integrate_trajectory(system: System, method: str,
                         dt: float, until: float, pars=None) -> Trajectory:
    """Integrate the system state forward in time for a number of steps."""
    steps = int(until / dt)
    trajectory = Trajectory(system)

    if method == 'gp':
        for step in range(0, steps):
            logger.info(f"Integrating step {step * dt:.2f}/{until:.2f}")
            system = system + Change(gp_law(system, pars) * dt)
            trajectory.append(system.data)

    if method == 'euler':
        for step in range(0, steps):
            logger.info(f"Integrating step {step * dt:.2f}/{until:.2f}")
            system = system + Change(law(system) * dt)
            trajectory.append(system.data)

    else:
        y0 = system.data.flatten()
        system.to_cpu()  # Ensure data is on CPU for ODE solvers


        if method in diffrax_methods:
            res = solve_diffrax(method, until, dt, y0, np.arange(0, dt * steps + dt, dt), system.masses, system.immobile)

        elif method in scipy_methods:
            res = solve_numpy(method, until, dt, y0, np.arange(0, dt * steps + dt, dt), system.masses, system.immobile)

        else:
            raise ValueError(f"Unknown integration method: {method}")
        
        trajectory.data = res.reshape((steps + 1, 2, len(system), system.D))

    return trajectory
