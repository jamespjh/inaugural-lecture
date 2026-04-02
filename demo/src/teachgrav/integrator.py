from .laws import law, flat_law
from .system import System, Trajectory, Change

import logging


logger = logging.getLogger("Teachgrav")

diffrax_methods = ['Tsit5', 'Dopri5', 'Kvaerno5']
scipy_methods = ['RK45', 'LSODA']


def solve_numpy(method, t1, dt, y0, saveat, masses, immobile):
    from scipy.integrate import solve_ivp

    def fun(t, y):
        return flat_law(y, masses, immobile)
    solve = solve_ivp(fun, (0, t1), y0, method=method,
                      t_eval=saveat, rtol=1e-6)
    return solve.y.T


def integrate_trajectory(system: System, method: str,
                         dt: float, until: float, model=None) -> Trajectory:
    """Integrate the system state forward in time for a number of steps."""
    steps = int(until / dt)
    trajectory = Trajectory(system)

    if method == 'gp':
        assert model is not None, "GP method requires a trained model"
        for step in range(0, steps):
            logger.info(f"Integrating step {step * dt:.2f}/{until:.2f}")
            system = system + Change(model.gp_law(system) * dt)
            trajectory.append(system.data)

    if method == 'euler':
        for step in range(0, steps):
            logger.info(f"Integrating step {step * dt:.2f}/{until:.2f}")
            system = system + Change(law(system) * dt)
            trajectory.append(system.data)

    else:
        y0 = system.data.flatten()
        np = system.data.__array_namespace__()

        if method in diffrax_methods:
            from .jax_integrator import solve_diffrax
            res = solve_diffrax(method, until, dt, y0, np.arange(
                0, dt * steps + dt, dt), system.masses, system.immobile)

        elif method in scipy_methods:
            res = solve_numpy(method, until, dt, y0, np.arange(
                0, dt * steps + dt, dt), system.masses, system.immobile)

        else:
            raise ValueError(f"Unknown integration method: {method}")

        trajectory.data = res.reshape((steps + 1, 2, len(system), system.D))

    return trajectory
