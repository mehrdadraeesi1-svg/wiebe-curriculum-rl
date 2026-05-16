"""
Wiebe function implementations for combustion modeling.

The Wiebe function describes the mass fraction burned (MFB) during combustion
in internal combustion engines.
"""

import numpy as np
from typing import Tuple, Dict
from numba import jit


@jit(nopython=True)
def single_wiebe(theta: np.ndarray, theta_0: float, delta_theta: float,
                 a: float, m: float) -> np.ndarray:
    """
    Calculate mass fraction burned using single Wiebe function.

    The Wiebe function is defined as:
    x_b(θ) = 1 - exp(-a * ((θ - θ₀) / Δθ)^(m+1))

    Args:
        theta: Crank angle array [degrees]
        theta_0: Start of combustion [degrees]
        delta_theta: Combustion duration [degrees]
        a: Efficiency parameter (typically 5-7 for complete combustion)
        m: Form factor (shape parameter, typically 0.5-4)

    Returns:
        Mass fraction burned array
    """
    # Initialize MFB array
    x_b = np.zeros_like(theta)

    # Calculate MFB only where theta > theta_0
    mask = theta >= theta_0
    theta_normalized = (theta[mask] - theta_0) / delta_theta

    # Wiebe function
    x_b[mask] = 1.0 - np.exp(-a * theta_normalized**(m + 1.0))

    # Clip to ensure physical bounds [0, 1]
    x_b = np.clip(x_b, 0.0, 1.0)

    return x_b


@jit(nopython=True)
def double_wiebe(theta: np.ndarray, theta_0: float, delta_theta: float,
                 a: float, m1: float, m2: float, lambda_w: float,
                 k: float) -> np.ndarray:
    """
    Calculate mass fraction burned using double Wiebe function.

    The double Wiebe function combines two Wiebe functions to model
    premixed and diffusion combustion phases:

    x_b(θ) = λ * x_b1(θ) + (1-λ) * x_b2(θ)

    where:
    - x_b1: fast burn phase (premixed)
    - x_b2: slow burn phase (diffusion)
    - Δθ₂ = k * Δθ₁

    Args:
        theta: Crank angle array [degrees]
        theta_0: Start of combustion [degrees]
        delta_theta: Duration of first Wiebe function [degrees]
        a: Efficiency parameter for both functions
        m1: Form factor for fast burn (premixed)
        m2: Form factor for slow burn (diffusion)
        lambda_w: Weight factor for fast burn [0, 1]
        k: Duration ratio (Δθ₂/Δθ₁)

    Returns:
        Mass fraction burned array
    """
    # First Wiebe (fast burn - premixed)
    x_b1 = single_wiebe(theta, theta_0, delta_theta, a, m1)

    # Second Wiebe (slow burn - diffusion)
    delta_theta_2 = k * delta_theta
    x_b2 = single_wiebe(theta, theta_0, delta_theta_2, a, m2)

    # Combined MFB
    x_b = lambda_w * x_b1 + (1.0 - lambda_w) * x_b2

    # Clip to ensure physical bounds
    x_b = np.clip(x_b, 0.0, 1.0)

    return x_b


@jit(nopython=True)
def calculate_burn_rate(theta: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    """
    Calculate the burn rate (dx_b/dθ) from mass fraction burned.

    Args:
        theta: Crank angle array [degrees]
        x_b: Mass fraction burned array

    Returns:
        Burn rate array [1/degree]
    """
    # Calculate derivative using central differences
    dx_b_dtheta = np.zeros_like(x_b)

    # Forward difference for first point
    if len(theta) > 1:
        dx_b_dtheta[0] = (x_b[1] - x_b[0]) / (theta[1] - theta[0])

    # Central differences for middle points
    for i in range(1, len(theta) - 1):
        dx_b_dtheta[i] = (x_b[i + 1] - x_b[i - 1]) / (theta[i + 1] - theta[i - 1])

    # Backward difference for last point
    if len(theta) > 1:
        dx_b_dtheta[-1] = (x_b[-1] - x_b[-2]) / (theta[-1] - theta[-2])

    return dx_b_dtheta


def get_wiebe_parameter_bounds(wiebe_type: str = 'single') -> Dict[str, Tuple[float, float]]:
    """
    Get physically meaningful bounds for Wiebe parameters.

    Args:
        wiebe_type: 'single' or 'double'

    Returns:
        Dictionary mapping parameter names to (min, max) tuples
    """
    if wiebe_type == 'single':
        return {
            'theta_0': (-20.0, 20.0),      # Start of combustion [deg ATDC]
            'delta_theta': (10.0, 100.0),  # Combustion duration [deg]
            'a': (0.5, 25.0),              # Efficiency parameter
            'm': (0.5, 4.0),               # Form factor
            'eta': (0.7, 1.0)              # Burned fuel fraction
        }
    elif wiebe_type == 'double':
        return {
            'theta_0': (-20.0, 20.0),      # Start of combustion [deg ATDC]
            'delta_theta': (10.0, 100.0),  # Combustion duration [deg]
            'a': (0.5, 25.0),              # Efficiency parameter
            'm1': (0.5, 4.0),              # Form factor (fast burn)
            'm2': (0.5, 4.0),              # Form factor (slow burn)
            'lambda_w': (0.05, 0.95),      # Weight factor
            'k': (1.0, 5.0),               # Duration ratio
            'eta': (0.7, 1.0)              # Burned fuel fraction
        }
    else:
        raise ValueError(f"Unknown wiebe_type: {wiebe_type}")


def normalize_wiebe_params(params: np.ndarray, wiebe_type: str = 'single') -> np.ndarray:
    """
    Normalize Wiebe parameters from physical range to [0, 1].

    Args:
        params: Array of physical parameter values
        wiebe_type: 'single' or 'double'

    Returns:
        Normalized parameters in [0, 1]
    """
    bounds = get_wiebe_parameter_bounds(wiebe_type)
    param_names = list(bounds.keys())

    normalized = np.zeros_like(params)
    for i, name in enumerate(param_names):
        min_val, max_val = bounds[name]
        normalized[i] = (params[i] - min_val) / (max_val - min_val)

    return normalized


def denormalize_wiebe_params(normalized_params: np.ndarray,
                             wiebe_type: str = 'single') -> np.ndarray:
    """
    Denormalize Wiebe parameters from [0, 1] to physical range.

    Args:
        normalized_params: Array of normalized parameter values [0, 1]
        wiebe_type: 'single' or 'double'

    Returns:
        Physical parameter values
    """
    bounds = get_wiebe_parameter_bounds(wiebe_type)
    param_names = list(bounds.keys())

    params = np.zeros_like(normalized_params)
    for i, name in enumerate(param_names):
        min_val, max_val = bounds[name]
        params[i] = normalized_params[i] * (max_val - min_val) + min_val

    return params


if __name__ == "__main__":
    # Test the Wiebe functions
    import matplotlib.pyplot as plt

    # Test parameters
    theta = np.linspace(-180, 180, 721)
    theta_0 = -5.0
    delta_theta = 40.0
    a = 5.0
    m = 2.0

    # Single Wiebe
    x_b_single = single_wiebe(theta, theta_0, delta_theta, a, m)
    dx_b_single = calculate_burn_rate(theta, x_b_single)

    # Double Wiebe
    m1 = 2.0
    m2 = 0.5
    lambda_w = 0.7
    k = 2.0
    x_b_double = double_wiebe(theta, theta_0, delta_theta, a, m1, m2, lambda_w, k)
    dx_b_double = calculate_burn_rate(theta, x_b_double)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(theta, x_b_single, label='Single Wiebe', linewidth=2)
    ax1.plot(theta, x_b_double, label='Double Wiebe', linewidth=2, linestyle='--')
    ax1.set_xlabel('Crank Angle [deg ATDC]')
    ax1.set_ylabel('Mass Fraction Burned [-]')
    ax1.set_title('Wiebe Function - MFB')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(theta, dx_b_single, label='Single Wiebe', linewidth=2)
    ax2.plot(theta, dx_b_double, label='Double Wiebe', linewidth=2, linestyle='--')
    ax2.set_xlabel('Crank Angle [deg ATDC]')
    ax2.set_ylabel('Burn Rate [1/deg]')
    ax2.set_title('Wiebe Function - Burn Rate')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig('wiebe_test.png', dpi=150)
    print("Test plot saved to wiebe_test.png")
