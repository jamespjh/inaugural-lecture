from jax import jit
import jax
from diffrax import diffeqsolve, ODETerm, PIDController
import diffrax
import equinox as eqx
from functools import partial
from .laws import flat_law


@partial(jit, static_argnames=['method'])
@eqx.debug.assert_max_traces(max_traces=3)
def diffrax_solve(method, t1, dt, y0, saveat, masses, immobile):
    def fun(t, y, args):
        return flat_law(y, args[0], args[1])

    term = ODETerm(fun)
    solver = getattr(diffrax, method)
    with jax.transfer_guard('log'):
        solve = diffeqsolve(
            term, solver(), t0=0, t1=t1, dt0=dt, y0=y0, args=(
                masses, immobile), saveat=diffrax.SaveAt(
                ts=saveat), stepsize_controller=PIDController(
                rtol=1e-6, atol=1e-6))
    return solve


def solve_diffrax(method, t1, dt, y0, saveat, masses, immobile):
    solve = diffrax_solve(method, t1, dt, y0, saveat, masses, immobile)
    return solve.ys
