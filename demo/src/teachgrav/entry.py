import sys
import argparse
import logging
from .scenarios import create_scenario
from .integrator import integrate_trajectory
from .viz import visualize
logger = logging.getLogger("Teachgrav")


def entry():
    args = parse_args()
    logger.setLevel(args.loglevel)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(args.loglevel)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logging.basicConfig(level=args.loglevel)
    logger.info(f'Loglevel set to {args.loglevel}')

    logger.info(
        f'Running scenario: {args.scenario} with method: {args.method}')
    trajectory = solve(args.scenario, args.method)

    if args.visualise:
        logger.info(f'Visualizing results with options: {args.visualise}')
        logger.info(f'Output file: {args.outfile}')
        if args.video:
            logger.info('video mode enabled')
        visualize(
            trajectory,
            output=args.outfile,
            mode='video' if args.video else 'plot',
            options=args.visualise)
    else:
        logger.info(
            "Outputting trajectory data to " +
            f"{args.outfile if args.outfile else 'stdout'}")
        stream = open(args.outfile, 'w') if args.outfile else sys.stdout
        trajectory.write(stream, args.format)


def parse_args(force_args=None):
    logger.info('Teachgrav called')
    parser = argparse.ArgumentParser(description='Teachgrav simulation')
    parser.add_argument('--scenario', default='moon',
                        choices=['moon', 'scatter', 'sun'])
    parser.add_argument('--method', default='euler', choices=['euler', 'rk4'])
    parser.add_argument(
        '--outfile',
        default=None,
        help='Output file for visualization (e.g. .mp4 or .gif)')
    parser.add_argument('--visualise', default='trail',
                        choices=['trail', 'dot'], help='Visualization style')
    parser.add_argument('--loglevel', default='WARNING',
                        help='Logging level (e.g. DEBUG, INFO, WARNING)')
    parser.add_argument('--log-file', default=None,
                        help='File to save log output')
    parser.add_argument(
        '--video',
        action='store_true',
        help='Whether to create a video output (implies --visualise)')
    parser.add_argument(
        '--format',
        default=None,
        choices=[
            'csv',
            'mp4',
            'png'],
        help='Output format for trajectory data (e.g. csv, json, png).' +
             'Inferred from outfile extension if not specified.')
    args = parser.parse_args(force_args.split() if force_args else None)
    if args.outfile and not args.format:
        logger.info(
            f"Selecting output format based on file extension: {args.outfile}")
        if args.outfile.endswith('.mp4'):
            args.video = True
            args.format = 'mp4'
        elif args.outfile.endswith('.csv'):
            args.video = False
            args.format = 'csv'
            args.visualise = None
        elif args.outfile.endswith('.png'):
            args.video = False
            args.format = 'png'
        else:
            logger.warning(
                f"Unknown file extension for output: {args.outfile}" +
                ". Defaulting to stdout text.")
            args.visualise = None
            args.format = 'csv'
            args.outfile = None  # Output to stdout
    if not args.outfile:
        logger.info("No output file specified. Defaulting to stdout text.")
        args.visualise = None
        args.format = 'csv'
    return args


def solve(scenario: str, method: str):
    system = create_scenario(scenario)
    trajectory = integrate_trajectory(system, method, dt=0.01, until=7.0)
    logger.info('Simulation complete')
    return trajectory
