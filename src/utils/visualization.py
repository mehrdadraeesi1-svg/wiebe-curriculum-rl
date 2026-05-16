"""
Visualization utilities for combustion modeling and RL results.

Provides functions for plotting:
- Pressure traces
- MFB profiles
- Heat release rates
- Performance metrics
- Training curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd


# Set default style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10


def plot_pressure_comparison(
    theta: np.ndarray,
    P_measured: np.ndarray,
    P_predicted: np.ndarray,
    title: str = 'Pressure Trace Comparison',
    r2: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot measured vs predicted pressure traces.

    Args:
        theta: Crank angle array [deg ATDC]
        P_measured: Measured pressure [Pa]
        P_predicted: Predicted pressure [Pa]
        title: Plot title
        r2: R² value to display
        save_path: Path to save figure
        show: Whether to show plot

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(theta, P_measured / 1e5, 'k-', linewidth=2.5,
           label='Measured', alpha=0.8)
    ax.plot(theta, P_predicted / 1e5, 'r--', linewidth=2,
           label='Predicted', alpha=0.8)

    ax.set_xlabel('Crank Angle [deg ATDC]', fontsize=12)
    ax.set_ylabel('Pressure [bar]', fontsize=12)

    if r2 is not None:
        title += f' (R² = {r2:.4f})'

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_mfb_profile(
    theta: np.ndarray,
    x_b: np.ndarray,
    title: str = 'Mass Fraction Burned Profile',
    x_b_true: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot mass fraction burned profile.

    Args:
        theta: Crank angle array [deg ATDC]
        x_b: MFB profile [-]
        title: Plot title
        x_b_true: True MFB profile (if available)
        save_path: Path to save figure
        show: Whether to show plot

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if x_b_true is not None:
        ax.plot(theta, x_b_true, 'k-', linewidth=2.5,
               label='True', alpha=0.8)
        ax.plot(theta, x_b, 'b--', linewidth=2,
               label='Predicted', alpha=0.8)
    else:
        ax.plot(theta, x_b, 'b-', linewidth=2.5, alpha=0.8)

    ax.set_xlabel('Crank Angle [deg ATDC]', fontsize=12)
    ax.set_ylabel('Mass Fraction Burned [-]', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])

    if x_b_true is not None:
        ax.legend(fontsize=11, loc='best')

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_complete_analysis(
    theta: np.ndarray,
    P_measured: np.ndarray,
    P_predicted: np.ndarray,
    x_b: np.ndarray,
    metrics: Dict[str, float],
    case_id: str = '',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create comprehensive analysis plot with pressure, MFB, and residuals.

    Args:
        theta: Crank angle array [deg ATDC]
        P_measured: Measured pressure [Pa]
        P_predicted: Predicted pressure [Pa]
        x_b: MFB profile [-]
        metrics: Dictionary with RMSE, R², etc.
        case_id: Case identifier
        save_path: Path to save figure
        show: Whether to show plot

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Pressure traces
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(theta, P_measured / 1e5, 'k-', linewidth=2, label='Measured', alpha=0.8)
    ax1.plot(theta, P_predicted / 1e5, 'r--', linewidth=2, label='Predicted', alpha=0.8)
    ax1.set_xlabel('Crank Angle [deg ATDC]', fontsize=11)
    ax1.set_ylabel('Pressure [bar]', fontsize=11)
    ax1.set_title(f'Pressure Trace - {case_id}', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Residuals
    ax2 = fig.add_subplot(gs[1, :])
    residuals = (P_predicted - P_measured) / 1e5
    ax2.plot(theta, residuals, 'g-', linewidth=1.5, alpha=0.7)
    ax2.axhline(0, color='k', linestyle='--', linewidth=1)
    ax2.fill_between(theta, residuals, 0, alpha=0.3)
    ax2.set_xlabel('Crank Angle [deg ATDC]', fontsize=11)
    ax2.set_ylabel('Residual [bar]', fontsize=11)
    ax2.set_title('Prediction Residuals', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. MFB profile
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(theta, x_b, 'b-', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Crank Angle [deg ATDC]', fontsize=11)
    ax3.set_ylabel('MFB [-]', fontsize=11)
    ax3.set_title('Mass Fraction Burned', fontsize=12, fontweight='bold')
    ax3.set_ylim([-0.05, 1.05])
    ax3.grid(True, alpha=0.3)

    # 4. Metrics table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    metrics_text = "Performance Metrics\n" + "="*30 + "\n\n"
    if 'r2' in metrics:
        metrics_text += f"R²: {metrics['r2']:.4f}\n\n"
    if 'rmse' in metrics:
        metrics_text += f"RMSE: {metrics['rmse']:.2e} Pa\n\n"
    if 'nrmse' in metrics:
        metrics_text += f"NRMSE: {metrics['nrmse']:.2f}%\n\n"

    # Peak pressure comparison
    P_peak_meas = np.max(P_measured) / 1e5
    P_peak_pred = np.max(P_predicted) / 1e5
    metrics_text += f"Peak Pressure:\n"
    metrics_text += f"  Measured: {P_peak_meas:.2f} bar\n"
    metrics_text += f"  Predicted: {P_peak_pred:.2f} bar\n"
    metrics_text += f"  Error: {abs(P_peak_pred - P_peak_meas):.2f} bar\n"

    ax4.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_metrics_distribution(
    metrics: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot distribution of evaluation metrics.

    Args:
        metrics: Dictionary with metric arrays (r2, rmse, nrmse)
        save_path: Path to save figure
        show: Whether to show plot

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # R² distribution
    if 'r2' in metrics:
        ax = axes[0]
        ax.hist(metrics['r2'], bins=30, edgecolor='black', alpha=0.7, color='blue')
        ax.axvline(np.mean(metrics['r2']), color='red', linestyle='--',
                  linewidth=2, label=f"Mean: {np.mean(metrics['r2']):.3f}")
        ax.axvline(np.median(metrics['r2']), color='green', linestyle='--',
                  linewidth=2, label=f"Median: {np.median(metrics['r2']):.3f}")
        ax.set_xlabel('R² Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('R² Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # RMSE distribution
    if 'rmse' in metrics:
        ax = axes[1]
        ax.hist(metrics['rmse'] / 1e5, bins=30, edgecolor='black',
               alpha=0.7, color='orange')
        ax.axvline(np.mean(metrics['rmse']) / 1e5, color='red', linestyle='--',
                  linewidth=2, label=f"Mean: {np.mean(metrics['rmse'])/1e5:.2f}")
        ax.set_xlabel('RMSE [bar]', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('RMSE Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # NRMSE distribution
    if 'nrmse' in metrics:
        ax = axes[2]
        ax.hist(metrics['nrmse'], bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.axvline(np.mean(metrics['nrmse']), color='red', linestyle='--',
                  linewidth=2, label=f"Mean: {np.mean(metrics['nrmse']):.2f}%")
        ax.set_xlabel('NRMSE (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('NRMSE Distribution', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_training_curves(
    log_path: str,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training curves from tensorboard logs.

    Args:
        log_path: Path to tensorboard log directory
        save_path: Path to save figure
        show: Whether to show plot

    Returns:
        Figure object
    """
    # This is a simplified version - actual implementation would parse tensorboard logs
    # For now, create a placeholder

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Placeholder for demonstration
    steps = np.arange(0, 10000, 100)
    reward = np.random.randn(len(steps)).cumsum() / 10 + 50

    axes[0, 0].plot(steps, reward)
    axes[0, 0].set_xlabel('Training Steps', fontsize=11)
    axes[0, 0].set_ylabel('Episode Reward', fontsize=11)
    axes[0, 0].set_title('Training Reward', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Add more plots as needed

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_baseline_comparison(
    comparison_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot comparison with baseline methods.

    Args:
        comparison_df: DataFrame with method comparison results
        save_path: Path to save figure
        show: Whether to show plot

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = comparison_df['Method'].values
    mean_r2 = comparison_df['Mean R²'].values
    std_r2 = comparison_df['Std R²'].values
    mean_nrmse = comparison_df['Mean NRMSE (%)'].values

    # R² comparison
    ax = axes[0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(methods, mean_r2, yerr=std_r2, capsize=5,
                  color=colors[:len(methods)], alpha=0.8, edgecolor='black')

    # Highlight RL method
    if 'RL' in methods[-1]:
        bars[-1].set_color('#d62728')
        bars[-1].set_alpha(1.0)

    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Comparison Across Methods', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.85, 1.0])

    # NRMSE comparison
    ax = axes[1]
    bars = ax.bar(methods, mean_nrmse, capsize=5,
                  color=colors[:len(methods)], alpha=0.8, edgecolor='black')

    if 'RL' in methods[-1]:
        bars[-1].set_color('#d62728')
        bars[-1].set_alpha(1.0)

    ax.set_ylabel('NRMSE (%)', fontsize=12)
    ax.set_title('NRMSE Comparison Across Methods', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    # Test visualization functions
    import sys
    sys.path.append('..')
    from src.models.wiebe_functions import single_wiebe
    from src.utils.thermodynamics import (
        EngineGeometry, EngineOperatingConditions,
        GasProperties, simulate_pressure_trace
    )

    # Generate test data
    theta = np.linspace(-180, 180, 721)
    x_b = single_wiebe(theta, -5.0, 40.0, 5.0, 2.0)

    geometry = EngineGeometry(0.086, 0.086, 0.143, 10.0)
    operating = EngineOperatingConditions(2000.0, 1.0, 100000.0, 320.0)
    gas_props = GasProperties()

    P, T, Q = simulate_pressure_trace(theta, x_b, geometry, operating, gas_props)

    # Add noise for "predicted"
    P_pred = P + np.random.normal(0, 0.02 * np.max(P), len(P))

    # Test plots
    from src.utils.thermodynamics import calculate_r_squared, calculate_normalized_rmse

    r2 = calculate_r_squared(P_pred, P)
    nrmse = calculate_normalized_rmse(P_pred, P)

    metrics = {'r2': r2, 'nrmse': nrmse, 'rmse': np.sqrt(np.mean((P_pred - P)**2))}

    # Complete analysis plot
    plot_complete_analysis(
        theta, P, P_pred, x_b, metrics,
        case_id='test_case',
        save_path='test_analysis.png',
        show=False
    )

    print("Visualization test completed!")
    print(f"Test plot saved to test_analysis.png")
