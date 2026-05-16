"""
Evaluation script for trained RL agent.

Evaluates the agent on test set and compares with baseline methods
(GP, ANN, MOSVR) from the research paper.
"""

import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import sys
sys.path.append('..')

from src.environment.wiebe_env import WiebeEnvV4 as WiebeEnvironment
from src.utils.data_utils import (
    load_experimental_data,
    generate_synthetic_dataset,
    normalize_dataset,
    CombustionDataset
)
from src.utils.thermodynamics import (
    calculate_rms_error,
    calculate_r_squared,
    calculate_normalized_rmse
)
from src.models.wiebe_functions import denormalize_wiebe_params


class AgentEvaluator:
    """Evaluator for trained RL agent."""

    def __init__(
        self,
        model_path: str,
        vec_normalize_path: str = None,
        wiebe_type: str = 'single',
        device: str = 'auto'
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
            vec_normalize_path: Path to VecNormalize stats
            wiebe_type: 'single' or 'double'
            device: Device to use
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.wiebe_type = wiebe_type

        # Load model
        self.logger.info(f"Loading model from {model_path}")
        if 'PPO' in str(model_path) or Path(model_path).parent.name.startswith('PPO'):
            self.model = PPO.load(model_path, device=device)
        else:
            self.model = SAC.load(model_path, device=device)

        # Load normalization stats if available
        self.vec_normalize = None
        if vec_normalize_path and Path(vec_normalize_path).exists():
            self.logger.info(f"Loading VecNormalize from {vec_normalize_path}")
            # Will be loaded with environment

    def evaluate_on_dataset(
        self,
        dataset: CombustionDataset,
        n_cases: int = None,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate agent on dataset.

        Args:
            dataset: Dataset to evaluate on
            n_cases: Number of cases to evaluate (None = all)
            deterministic: Use deterministic policy

        Returns:
            Dictionary with evaluation results
        """
        if n_cases is None:
            n_cases = len(dataset)
        else:
            n_cases = min(n_cases, len(dataset))

        self.logger.info(f"Evaluating on {n_cases} cases")

        # Storage for results
        results = {
            'case_ids': [],
            'rmse': [],
            'nrmse': [],
            'r2': [],
            'predicted_params': [],
            'true_params': [],
            'pressure_traces': []
        }

        # Evaluate each case
        for i in range(n_cases):
            case = dataset.get_case(i)

            # Create environment for this case
            env = WiebeEnvironment(
                wiebe_type=self.wiebe_type,
                reward_type='r2',
                normalize_observations=True
            )

            # Reset with case
            obs, _ = env.reset(options={'case': case})

            # Get prediction from model
            action, _ = self.model.predict(obs, deterministic=deterministic)

            # Execute action
            _, reward, _, _, info = env.step(action)

            # Store results
            results['case_ids'].append(case.get('case_id', f'case_{i}'))
            results['rmse'].append(info['rmse'])
            results['nrmse'].append(
                calculate_normalized_rmse(
                    info['pressure_predicted'],
                    case['pressure']
                )
            )
            results['r2'].append(info['r2'])
            results['predicted_params'].append(info['params'])

            # Store true params if available
            if 'wiebe_params_true' in case:
                results['true_params'].append(case['wiebe_params_true'])

            # Store pressure traces
            results['pressure_traces'].append({
                'measured': case['pressure'],
                'predicted': info['pressure_predicted'],
                'crank_angle': case['crank_angle'],
                'mfb': info['mfb']
            })

            env.close()

            if (i + 1) % 10 == 0:
                self.logger.info(f"Evaluated {i + 1}/{n_cases} cases")

        # Convert to numpy arrays
        results['rmse'] = np.array(results['rmse'])
        results['nrmse'] = np.array(results['nrmse'])
        results['r2'] = np.array(results['r2'])

        # Calculate summary statistics
        results['summary'] = {
            'mean_rmse': np.mean(results['rmse']),
            'std_rmse': np.std(results['rmse']),
            'mean_nrmse': np.mean(results['nrmse']),
            'std_nrmse': np.std(results['nrmse']),
            'mean_r2': np.mean(results['r2']),
            'std_r2': np.std(results['r2']),
            'median_r2': np.median(results['r2']),
            'min_r2': np.min(results['r2']),
            'max_r2': np.max(results['r2'])
        }

        self.logger.info("Evaluation completed")
        self.logger.info(f"Mean R²: {results['summary']['mean_r2']:.4f} "
                        f"± {results['summary']['std_r2']:.4f}")
        self.logger.info(f"Mean NRMSE: {results['summary']['mean_nrmse']:.2f}%")

        return results


def compare_with_baselines(
    rl_results: Dict[str, Any],
    baseline_results: Dict[str, Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Compare RL agent with baseline methods from paper.

    Args:
        rl_results: Results from RL evaluation
        baseline_results: Results from baseline methods (optional)

    Returns:
        Comparison dataframe
    """
    # If no baseline results provided, use typical values from paper
    if baseline_results is None:
        # Example values - replace with actual values from your paper
        baseline_results = {
            'ANN': {'mean_r2': 0.95, 'std_r2': 0.03, 'mean_nrmse': 2.5},
            'GP': {'mean_r2': 0.97, 'std_r2': 0.02, 'mean_nrmse': 1.8},
            'MOSVR': {'mean_r2': 0.96, 'std_r2': 0.025, 'mean_nrmse': 2.1}
        }

    # Create comparison dataframe
    methods = list(baseline_results.keys()) + ['RL (Ours)']
    data = []

    for method in methods:
        if method == 'RL (Ours)':
            row = {
                'Method': method,
                'Mean R²': rl_results['summary']['mean_r2'],
                'Std R²': rl_results['summary']['std_r2'],
                'Median R²': rl_results['summary']['median_r2'],
                'Mean NRMSE (%)': rl_results['summary']['mean_nrmse'],
                'Std NRMSE (%)': rl_results['summary']['std_nrmse']
            }
        else:
            row = {
                'Method': method,
                'Mean R²': baseline_results[method]['mean_r2'],
                'Std R²': baseline_results[method]['std_r2'],
                'Median R²': baseline_results[method].get('median_r2', np.nan),
                'Mean NRMSE (%)': baseline_results[method]['mean_nrmse'],
                'Std NRMSE (%)': baseline_results[method].get('std_nrmse', np.nan)
            }
        data.append(row)

    df = pd.DataFrame(data)
    return df


def save_results(
    results: Dict[str, Any],
    save_dir: str,
    experiment_name: str
) -> None:
    """
    Save evaluation results to file.

    Args:
        results: Evaluation results
        save_dir: Directory to save results
        experiment_name: Name of experiment
    """
    save_path = Path(save_dir) / experiment_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Save summary statistics
    summary_path = save_path / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results['summary'], f, indent=2)

    # Save detailed results
    results_df = pd.DataFrame({
        'case_id': results['case_ids'],
        'rmse': results['rmse'],
        'nrmse': results['nrmse'],
        'r2': results['r2']
    })
    results_df.to_csv(save_path / 'detailed_results.csv', index=False)

    # Save parameters
    params_data = []
    for i, params in enumerate(results['predicted_params']):
        row = {'case_id': results['case_ids'][i]}
        for j, val in enumerate(params):
            row[f'param_{j}'] = val
        params_data.append(row)

    params_df = pd.DataFrame(params_data)
    params_df.to_csv(save_path / 'predicted_parameters.csv', index=False)

    logging.info(f"Results saved to {save_path}")


def plot_results(
    results: Dict[str, Any],
    save_dir: str,
    n_cases_to_plot: int = 5
) -> None:
    """
    Create visualization plots.

    Args:
        results: Evaluation results
        save_dir: Directory to save plots
        n_cases_to_plot: Number of example cases to plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 150

    # 1. R² distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(results['r2'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results['summary']['mean_r2'], color='red',
              linestyle='--', linewidth=2, label=f"Mean: {results['summary']['mean_r2']:.3f}")
    ax.axvline(results['summary']['median_r2'], color='green',
              linestyle='--', linewidth=2, label=f"Median: {results['summary']['median_r2']:.3f}")
    ax.set_xlabel('R² Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of R² Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'r2_distribution.png')
    plt.close()

    # 2. NRMSE distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(results['nrmse'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(results['summary']['mean_nrmse'], color='red',
              linestyle='--', linewidth=2, label=f"Mean: {results['summary']['mean_nrmse']:.2f}%")
    ax.set_xlabel('NRMSE (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Normalized RMSE', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / 'nrmse_distribution.png')
    plt.close()

    # 3. Example pressure traces
    n_to_plot = min(n_cases_to_plot, len(results['pressure_traces']))

    for i in range(n_to_plot):
        trace = results['pressure_traces'][i]
        case_id = results['case_ids'][i]
        r2 = results['r2'][i]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Pressure traces
        ax1.plot(trace['crank_angle'], trace['measured'] / 1e5,
                'k-', linewidth=2, label='Measured')
        ax1.plot(trace['crank_angle'], trace['predicted'] / 1e5,
                'r--', linewidth=2, label='Predicted')
        ax1.set_xlabel('Crank Angle [deg ATDC]', fontsize=12)
        ax1.set_ylabel('Pressure [bar]', fontsize=12)
        ax1.set_title(f'Pressure Trace - {case_id} (R² = {r2:.4f})',
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # MFB profile
        ax2.plot(trace['crank_angle'], trace['mfb'], 'b-', linewidth=2)
        ax2.set_xlabel('Crank Angle [deg ATDC]', fontsize=12)
        ax2.set_ylabel('Mass Fraction Burned [-]', fontsize=12)
        ax2.set_title('Predicted MFB Profile', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / f'example_case_{i+1}_{case_id}.png')
        plt.close()

    # 4. Scatter plot: Measured vs Predicted peak pressure
    measured_peaks = []
    predicted_peaks = []

    for trace in results['pressure_traces']:
        measured_peaks.append(np.max(trace['measured']) / 1e5)
        predicted_peaks.append(np.max(trace['predicted']) / 1e5)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(measured_peaks, predicted_peaks, alpha=0.6, s=50)

    # Perfect prediction line
    min_val = min(min(measured_peaks), min(predicted_peaks))
    max_val = max(max(measured_peaks), max(predicted_peaks))
    ax.plot([min_val, max_val], [min_val, max_val],
           'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Measured Peak Pressure [bar]', fontsize=12)
    ax.set_ylabel('Predicted Peak Pressure [bar]', fontsize=12)
    ax.set_title('Peak Pressure Comparison', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path / 'peak_pressure_comparison.png')
    plt.close()

    logging.info(f"Plots saved to {save_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained RL agent'
    )

    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--vec-normalize-path', type=str, default=None,
                       help='Path to VecNormalize stats')
    parser.add_argument('--wiebe-type', type=str, default='single',
                       choices=['single', 'double'],
                       help='Type of Wiebe function')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to test dataset')
    parser.add_argument('--n-synthetic', type=int, default=50,
                       help='Number of synthetic test cases if no dataset')
    parser.add_argument('--n-cases', type=int, default=None,
                       help='Number of cases to evaluate')
    parser.add_argument('--save-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default='evaluation',
                       help='Experiment name')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting evaluation: {args.experiment_name}")

    # Set random seed
    np.random.seed(args.seed)

    # Load or generate test dataset
    if args.dataset_path and Path(args.dataset_path).exists():
        logger.info(f"Loading test dataset from {args.dataset_path}")
        test_dataset = load_experimental_data(args.dataset_path)
    else:
        logger.info(f"Generating {args.n_synthetic} synthetic test cases")
        test_dataset = generate_synthetic_dataset(
            n_cases=args.n_synthetic,
            wiebe_type=args.wiebe_type,
            random_state=args.seed
        )

    # Create evaluator
    evaluator = AgentEvaluator(
        model_path=args.model_path,
        vec_normalize_path=args.vec_normalize_path,
        wiebe_type=args.wiebe_type,
        device=args.device
    )

    # Evaluate
    results = evaluator.evaluate_on_dataset(
        test_dataset,
        n_cases=args.n_cases,
        deterministic=True
    )

    # Compare with baselines
    comparison_df = compare_with_baselines(results)
    logger.info("\nComparison with Baselines:")
    logger.info("\n" + comparison_df.to_string(index=False))

    # Save results
    save_results(results, args.save_dir, args.experiment_name)
    comparison_df.to_csv(
        Path(args.save_dir) / args.experiment_name / 'baseline_comparison.csv',
        index=False
    )

    # Create plots
    plot_results(
        results,
        Path(args.save_dir) / args.experiment_name / 'plots',
        n_cases_to_plot=5
    )

    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
