"""Training entry point for fitted physics laws."""
import argparse
import logging
from .scenarios import ScenarioFactory

logger = logging.getLogger("Teachgrav")


def train_model(force_args=None):
    """Train a fitted law model and save it to a file."""
    parser = argparse.ArgumentParser(
        description='Train a physics law model and save it to a file')
    parser.add_argument(
        '--law', required=True,
        choices=['gaussian', 'power'],
        help='Fitted law to train')
    parser.add_argument(
        '--output', required=True,
        help='Output file path for the trained model '
             '(YAML for power, joblib for gaussian)')
    parser.add_argument(
        '--scenario', default='scatter',
        choices=['moon', 'scatter', 'sun', 'single'],
        help='Scenario to use for generating training data')
    parser.add_argument(
        '--n-systems', dest='n_systems', type=int, default=256,
        help='Number of random systems to use as training data')
    parser.add_argument(
        '--n-bodies', dest='n_bodies', type=int, default=3,
        help='Number of bodies per training system (scatter scenario)')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility')
    parser.add_argument(
        '--engine', choices=['numpy', 'jax-gpu', 'jax-cpu', 'jax-metal',
                             'mlx-cpu', 'mlx-gpu'],
        default='numpy',
        help='Computation engine to use')
    parser.add_argument('--log-level', '--loglevel', dest='log_level',
                        default='WARNING',
                        help='Logging level (e.g. DEBUG, INFO, WARNING)')
    parser.add_argument('--log-file', default=None,
                        help='File to save log output')

    args = parser.parse_args(force_args.split() if force_args else None)

    logging.basicConfig(level=args.log_level)
    logger.setLevel(args.log_level)
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(args.log_level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if args.seed is not None:
        import numpy as np
        np.random.seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")

    factory = ScenarioFactory(args.engine)

    scenario_kwargs = {}
    if args.scenario == 'scatter':
        scenario_kwargs['n_bodies'] = args.n_bodies

    logger.info(f"Training {args.law} model on {args.n_systems} "
                f"'{args.scenario}' systems...")

    if args.law == 'power':
        from .laws.pl import PLModel
        model = PLModel(factory=factory)
        model.train(args.n_systems, **scenario_kwargs)
        model.save(args.output)
        logger.info(f"Saved power law model to {args.output}")
        print(f"Trained power law model: G={model.G:.6f}, "
              f"power={model.power:.6f}")
        print(f"Model saved to: {args.output}")
    elif args.law == 'gaussian':
        from .laws.gp import GPModel
        model = GPModel(factory=factory)
        model.train(args.n_systems, **scenario_kwargs)
        model.save(args.output)
        logger.info(f"Saved GP model to {args.output}")
        print("Trained Gaussian Process model.")
        print(f"Model saved to: {args.output}")
