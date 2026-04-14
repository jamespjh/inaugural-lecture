import sys
import argparse
import logging
from .scenarios import ScenarioFactory, STOCHASTIC_SCENARIOS
from .engine_support import jax_engines, mlx_engines
from .integrator import integrate_trajectory, diffrax_methods, scipy_methods
from .laws.laws import create_law
from .viz import visualize
from .benchmark import benchmark_engine
logger = logging.getLogger("Teachgrav")

FITTED_LAWS = ['gaussian', 'power']


def entry():
    args = parse_args()
    execute_scenario(args)


def _train_model(args, factory):
    """Train a fitted law and save the model to args.model_data."""
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
        model.train(args.n_systems, **scenario_kwargs)
        model.save(args.model_data)
        logger.info(f"Saved power law model to {args.model_data}")
        print(f"Trained power law model: G={model.G:.6f}, "
              f"power={model.power:.6f}")
        print(f"Model saved to: {args.model_data}")
    elif args.law == 'gaussian':
        from .laws.gp import GPModel
        model = GPModel(factory=factory)
        model.train(args.n_systems, **scenario_kwargs)
        model.save(args.model_data)
        logger.info(f"Saved GP model to {args.model_data}")
        print("Trained Gaussian Process model.")
        print(f"Model saved to: {args.model_data}")


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
        engine_consistent_seed=args.engine_consistent_seed)

    if getattr(args, 'train', False):
        _train_model(args, factory)
        return

    create_scenario = factory.create_scenario
    scenario_kwargs = {}
    if args.n_bodies is not None:
        scenario_kwargs['n_bodies'] = args.n_bodies
    system = create_scenario(args.scenario, **scenario_kwargs)
    if args.benchmark:
        logger.info('Running in benchmark mode')
        print(f'Benchmarking scenario: {args.scenario} with law: {args.law}')

        model = create_law(args.law, factory=factory,
                           model_data=getattr(args, 'model_data', None))

        def run_once():
            return model.law(system)

        time = benchmark_engine(run_once, args.engine)
        print(f'Benchmark time: {time:.5f} seconds')
        return

    logger.info(
        f'Running scenario: {args.scenario} with method: {args.method} '
        f'and law: {args.law}')
    trajectory = solve(system, args.method, args.law, factory=factory,
                       model_data=getattr(args, 'model_data', None))

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
            duration=args.duration)
    else:
        logger.info(
            "Outputting trajectory data to " +
            f"{args.outfile if args.outfile else 'stdout'}")
        stream = open(args.outfile, 'w') if args.outfile else sys.stdout
        trajectory.write(stream, args.format)

def _validate_args(args):
    """Validate parsed args and raise ValueError for incompatible options."""
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

    if args.outfile and not args.format and not args.train:
        logger.info(
            f"Selecting output format based on file extension: {args.outfile}")
        _resolve_output_format(args)

    if args.train:
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
        if args.model_data is None:
            raise ValueError(
                "--train requires --model-data to specify the output path "
                "for the saved model.")
        if args.video or args.outfile is not None:
            raise ValueError(
                "--train cannot be used with visualization options "
                "(--video, --outfile).")

    if args.duration is not None and not args.video:
        raise ValueError(
            "Option --duration can only be used with video output")

    if not args.train and args.law in FITTED_LAWS and args.model_data is None:
        raise ValueError(
            f"Law '{args.law}' requires a pre-trained model file. "
            f"Use --model-data to specify the path, or run "
            f"'teachgrav --train --law {args.law} --model-data <path>' "
            f"to train a model first.")

    if not args.outfile:
        logger.info("No output file specified. Defaulting to stdout text.")
        args.visualise = None
        args.format = 'csv'
    # Default to 30 seconds for video duration.
    args.duration = args.duration or 30

    # Enforce n_bodies is only used with scatter scenario
    if args.n_bodies is not None and args.scenario != 'scatter':
        raise ValueError(
            f"Option --n-bodies can only be used with the scatter scenario, "
            f"not '{args.scenario}'.")
    if args.n_bodies is not None and args.n_bodies < 1:
        raise ValueError("Option --n-bodies must be at least 1.")

    # Enforce law-solver compatibility
    if args.law in FITTED_LAWS and args.method != 'euler':
        logger.warning(
            f"Fitted law '{args.law}' is not compatible with solver "
            f"'{args.method}'. Switching to euler method.")
        args.method = 'euler'


def benchmark_scenario(args):
    """Run a single benchmark and return the mean timing in seconds."""
    factory = ScenarioFactory(
        args.engine, seed=args.seed,
        engine_consistent_seed=getattr(args, 'engine_consistent_seed', True))
    scenario_kwargs = {}
    if args.n_bodies is not None:
        scenario_kwargs['n_bodies'] = args.n_bodies
    system = factory.create_scenario(args.scenario, **scenario_kwargs)

    def run_once():
        return solve(system, args.method, law=args.law,
                     factory=factory, dt=0.01, until=0.05,
                     model_data=getattr(args, 'model_data', None))

    return benchmark_engine(run_once, args.engine)


def _validate_args(args):
    """Validate parsed args and raise ValueError for incompatible options."""
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

    if args.outfile and not args.format and not args.train:
        logger.info(
            f"Selecting output format based on file extension: {args.outfile}")
        _resolve_output_format(args)

    if args.train:
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
        if args.model_data is None:
            raise ValueError(
                "--train requires --model-data to specify the output path "
                "for the saved model.")
        if args.video or args.outfile is not None:
            raise ValueError(
                "--train cannot be used with visualization options "
                "(--video, --outfile).")

    if args.duration is not None and not args.video:
        raise ValueError(
            "Option --duration can only be used with video output")

    if not args.train and args.law in FITTED_LAWS and args.model_data is None:
        raise ValueError(
            f"Law '{args.law}' requires a pre-trained model file. "
            f"Use --model-data to specify the path, or run "
            f"'teachgrav --train --law {args.law} --model-data <path>' "
            f"to train a model first.")

    if not args.outfile:
        logger.info("No output file specified. Defaulting to stdout text.")
        args.visualise = None
        args.format = 'csv'
    # Default to 30 seconds for video duration.
    args.duration = args.duration or 30

    # Enforce n_bodies is only used with scatter scenario
    if args.n_bodies is not None and args.scenario != 'scatter':
        raise ValueError(
            f"Option --n-bodies can only be used with the scatter scenario, "
            f"not '{args.scenario}'.")
    if args.n_bodies is not None and args.n_bodies < 1:
        raise ValueError("Option --n-bodies must be at least 1.")

    # Enforce law-solver compatibility
    if args.law in FITTED_LAWS and args.method != 'euler':
        logger.warning(
            f"Fitted law '{args.law}' is not compatible with solver "
            f"'{args.method}'. Switching to euler method.")
        args.method = 'euler'


def benchmark_scenario(args):
    """Run a single benchmark and return the mean timing in seconds."""
    factory = ScenarioFactory(
        args.engine, seed=args.seed,
        engine_consistent_seed=getattr(args, 'engine_consistent_seed', True))
    scenario_kwargs = {}
    if args.n_bodies is not None:
        scenario_kwargs['n_bodies'] = args.n_bodies
    system = factory.create_scenario(args.scenario, **scenario_kwargs)

    def run_once():
        return solve(system, args.method, law=args.law,
                     factory=factory, dt=0.01, until=0.05,
                     model_data=getattr(args, 'model_data', None))

    return benchmark_engine(run_once, args.engine)


def _validate_args(args):
    """Validate parsed args and raise ValueError for incompatible options."""
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

    if args.outfile and not args.format and not args.train:
        logger.info(
            f"Selecting output format based on file extension: {args.outfile}")
        _resolve_output_format(args)

    if args.train:
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
        if args.model_data is None:
            raise ValueError(
                "--train requires --model-data to specify the output path "
                "for the saved model.")
        if args.video or args.outfile is not None:
            raise ValueError(
                "--train cannot be used with visualization options "
                "(--video, --outfile).")

    if args.duration is not None and not args.video:
        raise ValueError(
            "Option --duration can only be used with video output")

    if not args.train and args.law in FITTED_LAWS and args.model_data is None:
        raise ValueError(
            f"Law '{args.law}' requires a pre-trained model file. "
            f"Use --model-data to specify the path, or run "
            f"'teachgrav --train --law {args.law} --model-data <path>' "
            f"to train a model first.")

    if not args.outfile:
        logger.info("No output file specified. Defaulting to stdout text.")
        args.visualise = None
        args.format = 'csv'
    # Default to 30 seconds for video duration.
    args.duration = args.duration or 30

    # Enforce n_bodies is only used with scatter scenario
    if args.n_bodies is not None and args.scenario != 'scatter':
        raise ValueError(
            f"Option --n-bodies can only be used with the scatter scenario, "
            f"not '{args.scenario}'.")
    if args.n_bodies is not None and args.n_bodies < 1:
        raise ValueError("Option --n-bodies must be at least 1.")

    # Enforce law-solver compatibility
    if args.law in FITTED_LAWS and args.method != 'euler':
        logger.warning(
            f"Fitted law '{args.law}' is not compatible with solver "
            f"'{args.method}'. Switching to euler method.")
        args.method = 'euler'


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
    else:
        logger.warning(
            f"Unknown file extension for output: {args.outfile}" +
            ". Defaulting to stdout text.")
        args.visualise = None
        args.format = 'csv'
        args.outfile = None  # Output to stdout


def parse_args(force_args=None):
    logger.info('Teachgrav called')
    parser = argparse.ArgumentParser(description='Teachgrav simulation')
    parser.add_argument('--scenario', default='moon',
                        choices=['moon', 'scatter', 'sun', 'single'])
    parser.add_argument('--method', default='euler', choices=['euler'] +
                        diffrax_methods + scipy_methods)
    parser.add_argument('--law', default='gravity',
                        choices=['gravity', 'constant', 'gaussian', 'power'],
                        help='Physics law/acceleration model to use')
    parser.add_argument('--model-data', dest='model_data', default=None,
                        help='Path to a trained model file for simulation '
                             '(required for gaussian and power laws), or '
                             'output path when used with --train')
    parser.add_argument('--train', action='store_true',
                        help='Train a fitted law model and save it to '
                             '--model-data instead of running a simulation')
    parser.add_argument('--n-systems', dest='n_systems', type=int, default=256,
                        help='Number of training systems (used with --train)')
    parser.add_argument('--n-bodies', dest='n_bodies', type=int, default=None,
                        help='Number of bodies per system (used with --train '
                             'and scatter scenario)')
    # Add CUPY, Torch and MLX later.
    parser.add_argument(
        '--engine',
        choices=['numpy'] +
        jax_engines +
        mlx_engines,
        help='Computation engine to use')
    # Add CuPy and Torch later.
    parser.add_argument(
        '--outfile',
        default=None,
        help='Output file for visualization (e.g. .mp4 or .gif)')
    parser.add_argument('--visualise', default='trail',
                        choices=['trail', 'dot'], help='Visualization style')
    parser.add_argument('--log-level', '--loglevel', dest='log_level',
                        default='WARNING',
                        help='Logging level (e.g. DEBUG, INFO, WARNING)')
    parser.add_argument('--log-file', default=None,
                        help='File to save log output')
    parser.add_argument('--benchmark', action='store_true',
                        help='Whether to run in benchmark mode')
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
        '--video',
        action='store_true',
        help='Whether to create a video output (implies --visualise)')
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Video duration in seconds (default: 30).')
    parser.add_argument(
        '--format',
        default=None,
        choices=[
            'csv',
            'mp4',
            'png'],
        help='Output format for trajectory data (e.g. csv, mp4, png).' +
             'Inferred from outfile extension if not specified.')
    args = parser.parse_args(force_args.split() if force_args else None)

    # Validate --train mode before outfile processing so args.outfile
    # is still the original user-supplied value.
    _validate_args(args)

    return args


def solve(system, method: str, law: str = 'gravity', factory=None,
          dt: float = 0.01, until: float = 10, model_data: str = None):
    trajectory = integrate_trajectory(
        system, method, law=law, factory=factory, dt=dt, until=until,
        model_data=model_data)
    logger.info('Simulation complete')
    return trajectory
