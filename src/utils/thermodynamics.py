"""
Thermodynamic calculation utilities for engine combustion modeling.

Implements:
- Energy conservation equations
- Heat transfer models (Woschni) with learnable scaling factors
- Gas property calculations
- Pressure trace calculation from Wiebe parameters
"""

import numpy as np
from typing import Tuple, Dict, Optional
from numba import jit
from dataclasses import dataclass


@dataclass
class EngineGeometry:
    """Engine geometric parameters."""
    bore: float  # Cylinder bore [m]
    stroke: float  # Piston stroke [m]
    connecting_rod: float  # Connecting rod length [m]
    compression_ratio: float  # Compression ratio [-]

    def __post_init__(self):
        """Calculate derived parameters."""
        self.cylinder_area = np.pi * (self.bore / 2.0) ** 2
        self.displacement = self.cylinder_area * self.stroke
        self.clearance_volume = self.displacement / (self.compression_ratio - 1.0)


@dataclass
class EngineOperatingConditions:
    """Engine operating conditions."""
    speed: float  # Engine speed [RPM]
    lambda_: float  # Relative air-fuel ratio [-]
    intake_pressure: float  # Intake manifold pressure [Pa]
    intake_temperature: float  # Intake manifold temperature [K]
    fuel_LHV: float = 44.0e6  # Fuel lower heating value [J/kg]
    stoich_AFR: float = 14.7  # Stoichiometric air-fuel ratio [-]

    def __post_init__(self):
        """Calculate derived parameters."""
        self.angular_velocity = self.speed * 2.0 * np.pi / 60.0  # [rad/s]


@dataclass
class GasProperties:
    """Gas thermodynamic properties."""
    R: float = 287.0  # Gas constant for air [J/(kg·K)]
    gamma_u: float = 1.35  # Specific heat ratio (unburned)
    gamma_b: float = 1.28  # Specific heat ratio (burned)

    def cv(self, gamma: float) -> float:
        """Specific heat at constant volume [J/(kg·K)]"""
        return self.R / (gamma - 1.0)

    def cp(self, gamma: float) -> float:
        """Specific heat at constant pressure [J/(kg·K)]"""
        return gamma * self.R / (gamma - 1.0)


@jit(nopython=True)
def piston_position(theta: np.ndarray, stroke: float, connecting_rod: float) -> np.ndarray:
    """
    Calculate piston position relative to TDC.

    Args:
        theta: Crank angle [rad]
        stroke: Piston stroke [m]
        connecting_rod: Connecting rod length [m]

    Returns:
        Piston position [m] (positive = below TDC)
    """
    crank_radius = stroke / 2.0
    lambda_ratio = crank_radius / connecting_rod

    # Piston position
    x = crank_radius * ((1.0 - np.cos(theta)) +
                        lambda_ratio / 4.0 * (1.0 - np.cos(2.0 * theta)))

    return x


@jit(nopython=True)
def cylinder_volume(theta: np.ndarray, geometry_params: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Calculate instantaneous cylinder volume.

    Args:
        theta: Crank angle [rad]
        geometry_params: (bore, stroke, connecting_rod, clearance_volume)

    Returns:
        Cylinder volume [m³]
    """
    bore, stroke, connecting_rod, clearance_volume = geometry_params
    cylinder_area = np.pi * (bore / 2.0) ** 2

    # Piston position
    x = piston_position(theta, stroke, connecting_rod)

    # Volume
    V = clearance_volume + cylinder_area * x

    return V


@jit(nopython=True)
def woschni_heat_transfer(pressure: float, temperature: float, volume: float,
                          cylinder_area: float, bore: float, mean_piston_speed: float,
                          pressure_motored: float, volume_ref: float,
                          temperature_ref: float, x_b: float,
                          ht_scale_global: float = 1.0,
                          ht_scale_combustion: float = 1.0) -> float:
    """
    Calculate heat transfer coefficient using Woschni correlation with learnable scaling.

    Woschni correlation:
    h = C * B^(-0.2) * P^0.8 * T^(-0.53) * w^0.8

    where w is the characteristic gas velocity:
    w = C1 * S_p + C2 * (V_d * T_ref / (P_ref * V_ref)) * (P - P_motored)

    Args:
        pressure: Instantaneous cylinder pressure [Pa]
        temperature: Instantaneous gas temperature [K]
        volume: Instantaneous cylinder volume [m³]
        cylinder_area: Cylinder surface area [m²]
        bore: Cylinder bore [m]
        mean_piston_speed: Mean piston speed [m/s]
        pressure_motored: Motored pressure at same crank angle [Pa]
        volume_ref: Reference volume (at IVC) [m³]
        temperature_ref: Reference temperature (at IVC) [K]
        x_b: Mass fraction burned [-]
        ht_scale_global: Scaling factor for global heat transfer coefficient (C)
        ht_scale_combustion: Scaling factor for combustion velocity term (C2)

    Returns:
        Heat transfer rate [W]
    """
    # Woschni constants
    # Apply global scaling to C
    C = 130.0 * ht_scale_global  # [W/(m²·K·Pa^0.8·K^-0.53)]
    
    C1 = 2.28  # During compression/expansion
    C2_compression = 0.0  # During compression
    
    # Apply combustion scaling to C2
    C2_combustion = 0.00324 * ht_scale_combustion  # During combustion
    
    P_bar = pressure / 1e5 

    # Select C2 based on combustion state
    C2 = C2_compression if x_b < 0.001 else C2_combustion

    # Characteristic gas velocity
    displacement = volume_ref  # Approximation
    w = (C1 * mean_piston_speed +
         C2 * (displacement * temperature_ref / (pressure * volume_ref)) *
         (pressure - pressure_motored))

    # Heat transfer coefficient
    h = C * bore**(-0.2) * P_bar**0.8 * temperature**(-0.53) * w**0.8

    # Heat transfer rate
    Q_dot = h * cylinder_area * (temperature - 400.0)  # Assuming wall temp = 400 K

    return Q_dot


@jit(nopython=True)
def calculate_pressure_from_heat_release(
    theta: np.ndarray,
    Q_chem: np.ndarray,
    V: np.ndarray,
    geometry_params: Tuple[float, float, float],
    operating_params: Tuple[float, float, float],
    gas_props: Tuple[float, float, float],
    P_IVC: float,
    T_IVC: float,
    include_heat_transfer: bool = True,
    ht_scale_global: float = 1.0,
    ht_scale_combustion: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate pressure trace from chemical heat release using energy equation.

    Energy equation:
    dU = δQ_chem - δW - δQ_ht

    Args:
        theta: Crank angle array [rad]
        Q_chem: Cumulative chemical heat release [J]
        V: Cylinder volume array [m³]
        geometry_params: (bore, cylinder_area, mean_piston_speed)
        operating_params: (mass, pressure_motored_ref, volume_ref)
        gas_props: (R, gamma_u, gamma_b)
        P_IVC: Pressure at intake valve closing [Pa]
        T_IVC: Temperature at intake valve closing [K]
        include_heat_transfer: Whether to include heat transfer
        ht_scale_global: Global heat transfer scaling factor
        ht_scale_combustion: Combustion heat transfer scaling factor

    Returns:
        (pressure, temperature) arrays
    """
    bore, cylinder_area, mean_piston_speed = geometry_params
    mass, pressure_motored_ref, volume_ref = operating_params
    R, gamma_u, gamma_b = gas_props

    n_points = len(theta)
    P = np.zeros(n_points)
    T = np.zeros(n_points)

    # Initial conditions (at IVC)
    P[0] = P_IVC
    T[0] = T_IVC
    U = mass * R / (gamma_u - 1.0) * T[0]  # Initial internal energy

    # Integration loop
    for i in range(1, n_points):
        dtheta = theta[i] - theta[i-1]

        # Chemical heat release rate
        dQ_chem = Q_chem[i] - Q_chem[i-1]

        # Work done
        P_avg = P[i-1]  # Use previous pressure
        dV = V[i] - V[i-1]
        dW = P_avg * dV

        # Heat transfer (simplified - using previous values)
        dQ_ht = 0.0
        if include_heat_transfer and i > 1:
            # Approximate mass fraction burned
            x_b = Q_chem[i] / (Q_chem[-1] + 1e-10)

            # Motored pressure (polytropic compression, simplified)
            P_motored = pressure_motored_ref * (volume_ref / V[i])**gamma_u

            # Heat transfer coefficient with scaling factors
            Q_dot_ht = woschni_heat_transfer(
                P[i-1], T[i-1], V[i-1], cylinder_area, bore,
                mean_piston_speed, P_motored, volume_ref, T_IVC, x_b,
                ht_scale_global, ht_scale_combustion
            )

            # Convert to energy per crank angle
            dQ_ht = Q_dot_ht * dtheta / (2.0 * np.pi * mean_piston_speed / bore)  # Simplified time step
            dQ_ht = max(0.0, dQ_ht)  # Heat loss is positive

        # Update internal energy
        U = U + dQ_chem - dW - dQ_ht

        # Calculate temperature from internal energy
        # Assume average gamma (can be improved with burned fraction)
        gamma_avg = gamma_u  # Simplified
        T[i] = U * (gamma_avg - 1.0) / (mass * R)
        T[i] = max(T[i], 300.0)  # Minimum temperature

        # Calculate pressure from ideal gas law
        P[i] = mass * R * T[i] / V[i]

    return P, T


def simulate_pressure_trace(
    theta_deg: np.ndarray,
    x_b: np.ndarray,
    geometry: EngineGeometry,
    operating: EngineOperatingConditions,
    gas_props: GasProperties,
    eta: float = 1.0,
    include_heat_transfer: bool = True,
    ht_scale_global: float = 1.0,
    ht_scale_combustion: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate complete pressure trace from mass fraction burned profile.

    Args:
        theta_deg: Crank angle array [degrees ATDC]
        x_b: Mass fraction burned profile [-]
        geometry: Engine geometry parameters
        operating: Engine operating conditions
        gas_props: Gas thermodynamic properties
        eta: Combustion efficiency (burned fuel fraction)
        include_heat_transfer: Whether to include heat transfer
        ht_scale_global: Global heat transfer scaling factor
        ht_scale_combustion: Combustion heat transfer scaling factor

    Returns:
        (pressure [Pa], temperature [K], heat_release [J]) arrays
    """
    # Convert to radians
    theta_rad = np.deg2rad(theta_deg)

    # Calculate cylinder volume
    geom_params = (geometry.bore, geometry.stroke, geometry.connecting_rod,
                   geometry.clearance_volume)
    V = cylinder_volume(theta_rad, geom_params)

    # Estimate in-cylinder mass (simplified)
    volumetric_efficiency = 0.85  # Typical value
    mass = (volumetric_efficiency * operating.intake_pressure * geometry.displacement /
            (gas_props.R * operating.intake_temperature))

    # Calculate fuel mass
    AFR = operating.stoich_AFR * operating.lambda_
    fuel_mass = mass / (1.0 + AFR)

    # Total available heat
    Q_total = eta * fuel_mass * operating.fuel_LHV

    # Chemical heat release
    Q_chem = x_b * Q_total

    # Initial conditions (at start of array, assumed to be compression)
    P_IVC = operating.intake_pressure * 1.1  # Slightly higher than intake
    T_IVC = operating.intake_temperature * 1.05

    # Reference volume for heat transfer (approximate IVC volume)
    V_ref = geometry.displacement + geometry.clearance_volume

    # Mean piston speed
    mean_piston_speed = 2.0 * geometry.stroke * operating.speed / 60.0

    # Pressure motored reference (for heat transfer)
    P_motored_ref = P_IVC

    # Pack parameters
    geom_params_ht = (geometry.bore, geometry.cylinder_area, mean_piston_speed)
    op_params = (mass, P_motored_ref, V_ref)
    gas_params = (gas_props.R, gas_props.gamma_u, gas_props.gamma_b)

    # Calculate pressure and temperature with scaled heat transfer
    P, T = calculate_pressure_from_heat_release(
        theta_rad, Q_chem, V,
        geom_params_ht, op_params, gas_params,
        P_IVC, T_IVC, include_heat_transfer,
        ht_scale_global, ht_scale_combustion
    )

    return P, T, Q_chem


def calculate_rms_error(P_predicted: np.ndarray, P_measured: np.ndarray) -> float:
    """Calculate root mean square error between predicted and measured pressure."""
    return np.sqrt(np.mean((P_predicted - P_measured) ** 2))


def calculate_normalized_rmse(P_predicted: np.ndarray, P_measured: np.ndarray) -> float:
    """Calculate normalized RMSE (NRMSE) as percentage of peak pressure."""
    rmse = calculate_rms_error(P_predicted, P_measured)
    P_max = np.max(P_measured)
    return 100.0 * rmse / P_max


def calculate_r_squared(P_predicted: np.ndarray, P_measured: np.ndarray) -> float:
    """Calculate coefficient of determination (R²)."""
    ss_res = np.sum((P_measured - P_predicted) ** 2)
    ss_tot = np.sum((P_measured - np.mean(P_measured)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-10))


def compute_CA50(theta_deg: np.ndarray, mass_fraction: np.ndarray) -> float:
    """Estimate CA50 (crank angle for 50% mass fraction burned)."""
    if theta_deg.size == 0 or mass_fraction.size == 0:
        return float('nan')

    xb = np.clip(mass_fraction, 0.0, 1.0)
    target = 0.5
    indices = np.where(xb >= target)[0]

    if indices.size == 0:
        return float('nan')

    idx = int(indices[0])
    if idx == 0:
        return float(theta_deg[0])

    xb_prev = xb[idx - 1]
    xb_next = xb[idx]
    theta_prev = theta_deg[idx - 1]
    theta_next = theta_deg[idx]

    denom = xb_next - xb_prev
    if abs(denom) < 1e-8:
        return float(theta_next)

    alpha = (target - xb_prev) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    return float(theta_prev + alpha * (theta_next - theta_prev))


if __name__ == "__main__":
    # Test thermodynamic calculations
    from src.models.wiebe_functions import single_wiebe
    import matplotlib.pyplot as plt

    # Setup
    geometry = EngineGeometry(
        bore=0.086,
        stroke=0.086,
        connecting_rod=0.143,
        compression_ratio=10.0
    )

    operating = EngineOperatingConditions(
        speed=2000.0,
        lambda_=1.0,
        intake_pressure=100000.0,
        intake_temperature=320.0
    )

    gas_props = GasProperties()

    theta_deg = np.linspace(-180, 180, 721)
    x_b = single_wiebe(theta_deg, theta_0=-5.0, delta_theta=40.0, a=5.0, m=2.0)

    # Test with default HT scaling
    P_def, T_def, _ = simulate_pressure_trace(theta_deg, x_b, geometry, operating, gas_props)
    
    # Test with increased HT scaling (expect lower pressure/temp)
    P_high, T_high, _ = simulate_pressure_trace(
        theta_deg, x_b, geometry, operating, gas_props, 
        ht_scale_global=2.0, ht_scale_combustion=2.0
    )

    print(f"Peak Pressure (Default): {np.max(P_def)/1e5:.2f} bar")
    print(f"Peak Pressure (High HT): {np.max(P_high)/1e5:.2f} bar")
    
    assert np.max(P_high) < np.max(P_def), "Higher heat transfer should reduce peak pressure!"
    print("Verification passed: Heat transfer scaling is working.")