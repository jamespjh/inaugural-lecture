import argparse
import logging
from .scenarios import create_scenario
from .integrator import integrate_trajectory
from .viz import visualize
logger = logging.getLogger(__name__)


def entry():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    logger.info(
        f'Running scenario: {args.scenario} with method: {args.method}')
    trajectory = solve(args.scenario, args.method)
    visualize(trajectory, output=None)


def parse_args(force_args=None):
    logger.info('Teachgrav called')
    parser = argparse.ArgumentParser(description='Teachgrav simulation')
    parser.add_argument('--scenario', default='moon',
                        choices=['moon', 'scatter'])
    parser.add_argument('--method', default='euler', choices=['euler', 'rk4'])
    return parser.parse_args(force_args.split() if force_args else None)


def solve(scenario: str, method: str):
    system = create_scenario(scenario)
    trajectory = integrate_trajectory(system, method, dt=0.01, steps=1000)
    logger.info('Simulation complete')
    return trajectory
