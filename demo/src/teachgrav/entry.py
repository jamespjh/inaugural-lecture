import sys
import argparse
import logging
from .scenarios import ScenarioFactory, STOCHASTIC_SCENARIOS
from .engine_support import get_available_engines
from .integrator import integrate_trajectory, diffrax_methods, scipy_methods
from .laws.laws import create_law
from .visualisations.visualize import visualize, figsize_from_aspect, grid_plot
from .visualisations.convergence import generate_convergence_video
from .benchmark import benchmark_engine
logger = logging.getLogger("Teachgrav")

FITTED_LAWS = ['gaussian', 'power']


def entry():
    args = parse_args()
    execute_scenario(args)


def _train_model(args, factory):
    """Train a fitted law and write either model output or video output."""
    checkpoints = []

    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    scenario_kwargs = {}
    if args.scenario == 'scatter' and args.n_bodies is not None:
        scenario_kwargs['n_bodies'] = args.n_bodies

    logger.info(f"Training {args.law} model on {args.n_systems} "
                f"'{args.scenario}' systems...")

    if args.law == 'power':
        from .laws.pl import PLModel
        model = PLModel(factory=factory)
    elif args.law == 'gaussian':
        from .laws.gp import GPModel
        model = GPModel(factory=factory)
    else:
        raise ValueError(f"Unsupported law for training: {args.law}")
    checkpoints = model.train(args.n_systems, **scenario_kwargs)
    print("Trained model.")
    if not args.video:
        model.save(args.outfile)
        logger.info(f"Saved model to {args.outfile}")
        print(f"Model saved to: {args.outfile}")

    else:
        _generate_convergence_video(
            args, checkpoints, scenario_kwargs, output=args.outfile)


def _generate_convergence_video(args, checkpoints, scenario_kwargs, output):
    """Translate parsed CLI args into explicit convergence-video inputs."""
    generate_convergence_video(
        checkpoints=checkpoints,
        scenario=args.scenario,
        output=output,
        checkpoint_interval=args.checkpoint_interval,
        show_true_law=args.show_true_law,
        seed=args.seed,
        method=args.method,
        dt=args.dt,
        until=args.until,
        duration=args.duration,
        fps=args.fps,
        scenario_kwargs=scenario_kwargs,
        figsize=args.figsize,
    )


def execute_scenario(args):
    logger.setLevel(args.log_level)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(args.log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logging.basicConfig(level=args.log_level)
    logger.info(f'Loglevel set to {args.log_level}')
    factory = ScenarioFactory(
        args.engine, seed=args.seed,
        via_numpy=args.engine_consistent_seed)

    if args.train:
        _train_model(args, factory)
        return

    create_scenario = factory.create_scenario
    scenario_kwargs = {}
    if args.n_bodies is not None:
        scenario_kwargs['n_bodies'] = args.n_bodies

    if args.benchmark or args.benchmark_solve:
        benchmark_scenario(args)
        return

    if args.grid is not None:
        _execute_grid_scenario(args, factory, scenario_kwargs)
        return

    system = create_scenario(args.scenario, **scenario_kwargs)
    logger.info(
        f'Running scenario: {args.scenario} with method: {args.method} '
        f'and law: {args.law}')
    trajectory = solve(system, args.method, args.law, factory=factory,
                       dt=args.dt, until=args.until,
                       model_data=args.model_data)

    if args.visualise:
        logger.info(f'Visualizing results with options: {args.visualise}')
        logger.info(f'Output file: {args.outfile}')
        if args.video:
            logger.info('video mode enabled')
        visualize(
            trajectory,
            output=args.outfile,
            mode='video' if args.video else 'plot',
            options=args.visualise,
            duration=args.duration,
            fps=args.fps,
            figsize=args.figsize)
    else:
        logger.info(
            "Outputting trajectory data to " +
            f"{args.outfile if args.outfile else 'stdout'}")
        stream = open(args.outfile, 'w') if args.outfile else sys.stdout
        trajectory.write(stream, args.format)


def _execute_grid_scenario(args, factory, scenario_kwargs):
    """Generate N*N scenario samples and write a grid trail plot."""
    grid_size = args.grid
    n_samples = grid_size * grid_size
    logger.info(
        f'Generating {n_samples} samples for {grid_size}x{grid_size} grid '
        f'of scenario: {args.scenario}')

    trajectories = []
    for _ in range(n_samples):
        system = factory.create_scenario(args.scenario, **scenario_kwargs)
        traj = solve(system, args.method, args.law, factory=factory,
                     dt=args.dt, until=args.until,
                     model_data=args.model_data)
        trajectories.append(traj)

    options = args.visualise or 'trail'
    grid_plot(trajectories, grid_size, output=args.outfile, options=options)
    if args.outfile:
        logger.info(f'Grid plot saved to {args.outfile}')


def _validate_args(args):
    """Validate parsed args and raise ValueError for incompatible options."""
    if args.benchmark and args.benchmark_solve:
        raise ValueError(
            "--benchmark and --benchmark-solve cannot be used together.")
    if args.method in diffrax_methods and args.engine == 'numpy':
        raise ValueError(
            f"Method {args.method} is not compatible"
            f"with engine {args.engine}")

    if args.method in diffrax_methods and not args.engine:
        args.engine = 'jax-cpu'
        logger.info(
            f"Method {args.method} requires a JAX engine."
            f"Defaulting to {args.engine}.")
    else:
        args.engine = args.engine or 'numpy'
        logger.info(f"Using engine: {args.engine}")

    if args.outfile:
        logger.info(
            f"Selecting output format based on file extension: {
                args.outfile}")
        _resolve_output_format(args)

    if args.train:
        _validate_train_args(args)

    if not args.outfile:
        logger.info("No output file specified. Defaulting to stdout text.")
        if args.grid is None:
            args.visualise = None
        args.format = 'csv'
        args.video = False

    if args.duration is not None and not args.video:
        raise ValueError(
            "Option --duration can only be used with video output")

    if not args.train and args.law in FITTED_LAWS and args.model_data is None:
        raise ValueError(
            f"Law '{args.law}' requires a pre-trained model file. "
            f"Use --model-data to specify the path, or run "
            f"'teachgrav --train --law {args.law} --outfile <path>' "
            f"to train a model first.")
    # Resolve --aspect to a figsize tuple.
    args.figsize = figsize_from_aspect(args.aspect)

    # Default to 30 seconds for video duration.
    args.duration = args.duration or 30
    # Default integration timestep and end time.
    args.dt = args.dt if args.dt is not None else 0.01
    args.until = args.until if args.until is not None else 10.0

    # Enforce n_bodies is only used with scatter scenario
    if args.n_bodies is not None and args.scenario != 'scatter':
        raise ValueError(
            f"Option --n-bodies can only be used with the scatter scenario, "
            f"not '{args.scenario}'.")
    if args.n_bodies is not None and args.n_bodies < 1:
        raise ValueError("Option --n-bodies must be at least 1.")

    if args.grid is not None and args.grid < 1:
        raise ValueError("Option --grid must be at least 1.")


def benchmark_scenario(args):
    """Run a benchmark and return the mean timing in seconds.

    If ``args.benchmark`` is set, times a single law evaluation.
    If ``args.benchmark_solve`` is set, times the full ODE solver.
    """
    factory = ScenarioFactory(
        args.engine, seed=args.seed,
        via_numpy=args.engine_consistent_seed)
    scenario_kwargs = {}
    if args.n_bodies is not None:
        scenario_kwargs['n_bodies'] = args.n_bodies
    system = factory.create_scenario(args.scenario, **scenario_kwargs)

    if args.benchmark_solve:
        def run_once():
            return solve(system, args.method, args.law, factory=factory,
                         dt=args.dt, until=args.until,
                         model_data=getattr(args, 'model_data', None))
    else:
        model = create_law(
            args.law,
            factory=factory,
            model_data=getattr(args, 'model_data', None),
        )

        def run_once():
            return model.law(system)

    return benchmark_engine(run_once, args.engine)


def _validate_train_args(args):
    """Validate arguments specific to --train mode."""
    if args.law not in FITTED_LAWS:
        raise ValueError(
            f"--train requires a fitted law (gaussian or power). "
            f"'{args.law}' does not require training.")
    if args.scenario not in STOCHASTIC_SCENARIOS:
        valid = ', '.join(STOCHASTIC_SCENARIOS)
        raise ValueError(
            f"--train requires a stochastic scenario. "
            f"'{args.scenario}' is not suitable for training. "
            f"Valid training scenarios: {valid}.")
    if args.outfile is None:
        raise ValueError(
            "--train requires --outfile to specify either the saved model "
            "path (yaml/joblib) or an .mp4 convergence video path.")
    if args.model_data is not None:
        raise ValueError(
            "Use --outfile for --train output; --model-data is only for "
            "loading fitted laws during simulation.")
    if args.video and args.law != 'power':
        raise ValueError(
            "Convergence video output is only supported for --law power.")
    if args.checkpoint_interval < 1:
        raise ValueError("--checkpoint-interval must be at least 1.")


def _resolve_output_format(args):
    """Detect and set output format from file extension."""
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
    elif args.outfile.endswith('.yaml') or args.outfile.endswith('.yml'):
        args.video = False
        args.format = 'yaml'
        args.visualise = None
    elif args.outfile.endswith('.joblib'):
        args.video = False
        args.format = 'joblib'
        args.visualise = None
    else:
        logger.warning(
            f"Unknown file extension for output: {args.outfile}" +
            ". Defaulting to stdout text.")
        args.visualise = None
        args.video = False
        args.format = 'csv'
        args.outfile = None  # Output to stdout


def parse_args(force_args=None):
    logger.info('Teachgrav called')
    parser = argparse.ArgumentParser(description='Teachgrav simulation')
    parser.add_argument('--scenario', default='moon',
                        choices=['moon', 'scatter', 'sun', 'single', 'boids'])
    parser.add_argument('--method', default='euler', choices=['euler'] +
                        diffrax_methods + scipy_methods)
    parser.add_argument('--law', default='gravity',
                        choices=['gravity', 'boids', 'constant', 'gaussian',
                                 'power'],
                        help='Physics law/acceleration model to use')
    parser.add_argument('--model-data', dest='model_data', default=None,
                        help=(
                            'Path to a trained model file for simulation '
                            '(required for gaussian and power laws).'))
    parser.add_argument('--train', action='store_true',
                        help=(
                            'Train a fitted law model and write output to '
                            '--outfile instead of running a simulation'))
    parser.add_argument('--n-systems', dest='n_systems', type=int, default=256,
                        help='Number of training systems (used with --train)')
    parser.add_argument('--n-bodies', dest='n_bodies', type=int, default=None,
                        help='Number of bodies per system (used with --train '
                             'and scatter scenario)')
    # Add CUPY, Torch and MLX later.
    parser.add_argument(
        '--engine',
        choices=get_available_engines(),
        help='Computation engine to use')
    # Add CuPy and Torch later.
    parser.add_argument(
        '--outfile',
        default=None,
        help='Output path. Format and behaviour inferred from extension.' +
        ' Supported extensions: .csv, .png, .mp4, .yaml/.yml (for models), '
        '.joblib (for models). ' +
        'If no extension is given, defaults to stdout text output in '
        'CSV format.')
    parser.add_argument('--visualise', default='trail',
                        choices=['trail', 'dot'], help='Visualization style')
    parser.add_argument('--log-level', '--loglevel', dest='log_level',
                        default='WARNING',
                        help='Logging level (e.g. DEBUG, INFO, WARNING)')
    parser.add_argument('--log-file', default=None,
                        help='File to save log output')
    parser.add_argument('--benchmark', action='store_true',
                        help='Whether to run in benchmark mode (law only)')
    parser.add_argument('--benchmark-solve', dest='benchmark_solve',
                        action='store_true',
                        help=(
                            'Benchmark the full ODE solver; '
                            'uses --until to control the duration.'))
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducible scenarios (e.g. scatter).')
    parser.add_argument(
        '--engine-consistent-seed',
        dest='engine_consistent_seed',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Generate scenarios via numpy first for cross-engine RNG '
             'consistency when --seed is used (default: enabled).')
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Video duration in seconds (default: 30).')
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Frames per second for all video outputs (default: 20).')
    parser.add_argument(
        '--dt',
        type=float,
        default=None,
        help='Integration timestep (default: 0.01).')
    parser.add_argument(
        '--until',
        type=float,
        default=None,
        help='Simulation end time (default: 10).')
    parser.add_argument(
        '--checkpoint-interval',
        dest='checkpoint_interval',
        type=int,
        default=1,
        help='Use every Nth training checkpoint when building the convergence '
             'video (default: 1, i.e. every step).')
    parser.add_argument(
        '--aspect',
        default='column',
        choices=['page', 'column'],
        help='Figure size preset: "column" (6.4×7.2 in, default) or '
             '"page" (12.8×7.2 in).')
    parser.add_argument(
        '--show-true-law',
        dest='show_true_law',
        action='store_true',
        help='Overlay the true-law trajectory in each convergence-video frame '
             'so the viewer can see the fitted law converging toward it.')
    parser.add_argument(
        '--grid',
        type=int,
        default=None,
        help='Generate an N×N static grid image where each cell shows the '
             'trail of one independent sample from the scenario generator. '
             'Use with --outfile <path>.png (or omit for interactive '
             'display).')
    args = parser.parse_args(force_args.split() if force_args else None)
    _validate_args(args)

    return args


def solve(system, method: str, law: str = 'gravity', factory=None,
          dt: float = 0.01, until: float = 10, model_data: str | None = None):
    trajectory = integrate_trajectory(
        system, method, law=law, factory=factory, dt=dt, until=until,
        model_data=model_data)
    logger.info('Simulation complete')
    return trajectory
