"""
Data loading and preprocessing utilities for combustion modeling.

Handles:
- Loading experimental data from various formats
- Data normalization and standardization
- Train/validation/test splits
- Synthetic data generation for testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
from pathlib import Path


@dataclass
class CombustionDataset:
    """Container for combustion dataset."""
    cases: List[Dict[str, Any]]
    feature_names: List[str]
    scaler: Optional[StandardScaler] = None
    metadata: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        return len(self.cases)

    def get_case(self, idx: int) -> Dict[str, Any]:
        """Get a single case by index."""
        return self.cases[idx]

    def get_features(self) -> np.ndarray:
        """Extract feature matrix from all cases."""
        features = []
        for case in self.cases:
            case_features = [case[name] for name in self.feature_names]
            features.append(case_features)
        return np.array(features)

    def split(self, test_size: float = 0.2, val_size: float = 0.1,
              random_state: int = 42) -> Tuple['CombustionDataset', ...]:
        """
        Split dataset into train, validation, and test sets.

        Args:
            test_size: Fraction for test set
            val_size: Fraction of remaining data for validation
            random_state: Random seed

        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        # First split: train+val vs test
        train_val_cases, test_cases = train_test_split(
            self.cases,
            test_size=test_size,
            random_state=random_state
        )

        # Second split: train vs val
        train_cases, val_cases = train_test_split(
            train_val_cases,
            test_size=val_size / (1 - test_size),
            random_state=random_state
        )

        # Create datasets
        train_dataset = CombustionDataset(
            cases=train_cases,
            feature_names=self.feature_names,
            scaler=None,
            metadata=self.metadata
        )

        val_dataset = CombustionDataset(
            cases=val_cases,
            feature_names=self.feature_names,
            scaler=None,
            metadata=self.metadata
        )

        test_dataset = CombustionDataset(
            cases=test_cases,
            feature_names=self.feature_names,
            scaler=None,
            metadata=self.metadata
        )

        return train_dataset, val_dataset, test_dataset


def load_experimental_data(
    data_path: str,
    file_format: str = 'csv'
) -> CombustionDataset:
    """
    Load experimental combustion data from file.

    Expected data format:
    - Each row is one experimental case
    - Columns include: case_id, speed, lambda, intake_pressure,
      intake_temperature, compression_ratio, pressure_trace, crank_angle

    Args:
        data_path: Path to data file
        file_format: 'csv', 'pickle', or 'json'

    Returns:
        CombustionDataset object
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if file_format == 'csv':
        df = pd.read_csv(data_path)
        cases = _dataframe_to_cases(df)
    elif file_format == 'pickle':
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        cases = data['cases'] if isinstance(data, dict) else data
    elif file_format == 'json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        cases = data['cases'] if isinstance(data, dict) else data
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    # Define feature names (operating conditions)
    feature_names = [
        'speed', 'lambda', 'intake_pressure',
        'intake_temperature', 'compression_ratio'
    ]

    metadata = {
        'n_cases': len(cases),
        'source': str(path),
        'format': file_format
    }

    return CombustionDataset(
        cases=cases,
        feature_names=feature_names,
        metadata=metadata
    )


def _dataframe_to_cases(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert pandas DataFrame to list of case dictionaries."""
    cases = []

    for idx, row in df.iterrows():
        case = {
            'case_id': row.get('case_id', f'case_{idx}'),
            'speed': float(row['speed']),
            'lambda': float(row['lambda']),
            'intake_pressure': float(row['intake_pressure']),
            'intake_temperature': float(row['intake_temperature']),
            'compression_ratio': float(row.get('compression_ratio', 10.0))
        }

        # Handle pressure trace (might be stored as string or array)
        if 'pressure' in row:
            pressure = row['pressure']
            if isinstance(pressure, str):
                # Parse string representation
                case['pressure'] = np.fromstring(
                    pressure.strip('[]'),
                    sep=','
                )
            else:
                case['pressure'] = np.array(pressure)

        # Handle crank angle
        if 'crank_angle' in row:
            crank_angle = row['crank_angle']
            if isinstance(crank_angle, str):
                case['crank_angle'] = np.fromstring(
                    crank_angle.strip('[]'),
                    sep=','
                )
            else:
                case['crank_angle'] = np.array(crank_angle)
        else:
            # Default crank angle array
            n_points = len(case.get('pressure', [721]))
            case['crank_angle'] = np.linspace(-180, 180, n_points)

        cases.append(case)

    return cases


def normalize_dataset(
    dataset: CombustionDataset,
    fit: bool = True
) -> CombustionDataset:
    """
    Normalize dataset features using standardization.

    Args:
        dataset: Input dataset
        fit: Whether to fit scaler (True for train, False for val/test)

    Returns:
        Normalized dataset with fitted scaler
    """
    # Extract features
    features = dataset.get_features()

    if fit:
        # Fit and transform
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        dataset.scaler = scaler
    else:
        # Transform only (requires pre-fitted scaler)
        if dataset.scaler is None:
            raise ValueError("Cannot normalize without fitted scaler")
        features_normalized = dataset.scaler.transform(features)

    # Update cases with normalized values
    for i, case in enumerate(dataset.cases):
        for j, name in enumerate(dataset.feature_names):
            case[f'{name}_normalized'] = features_normalized[i, j]

    return dataset


def generate_synthetic_dataset(
    n_cases: int = 100,
    wiebe_type: str = 'single',
    noise_level: float = 0.02,
    random_state: int = 42
) -> CombustionDataset:
    """
    Generate synthetic combustion dataset for testing.

    Args:
        n_cases: Number of cases to generate
        wiebe_type: 'single' or 'double'
        noise_level: Noise level as fraction of peak pressure
        random_state: Random seed

    Returns:
        Synthetic dataset
    """
    from src.models.wiebe_functions import (
        single_wiebe, double_wiebe, get_wiebe_parameter_bounds
    )
    from src.utils.thermodynamics import (
        EngineGeometry, EngineOperatingConditions,
        GasProperties, simulate_pressure_trace
    )

    np.random.seed(random_state)

    # Parameter bounds
    bounds = get_wiebe_parameter_bounds(wiebe_type)

    # Feature names
    feature_names = [
        'speed', 'lambda', 'intake_pressure',
        'intake_temperature', 'compression_ratio'
    ]

    # Feature ranges
    feature_ranges = {
        'speed': (1000.0, 4000.0),
        'lambda': (0.8, 1.2),
        'intake_pressure': (80000.0, 120000.0),
        'intake_temperature': (300.0, 350.0),
        'compression_ratio': (9.0, 12.0)
    }

    # Default geometry (will be updated per case)
    geometry = EngineGeometry(
        bore=0.086,
        stroke=0.086,
        connecting_rod=0.143,
        compression_ratio=10.0
    )

    gas_props = GasProperties()

    cases = []

    for i in range(n_cases):
        # Random operating conditions
        speed = np.random.uniform(*feature_ranges['speed'])
        lambda_ = np.random.uniform(*feature_ranges['lambda'])
        intake_p = np.random.uniform(*feature_ranges['intake_pressure'])
        intake_t = np.random.uniform(*feature_ranges['intake_temperature'])
        cr = np.random.uniform(*feature_ranges['compression_ratio'])

        operating = EngineOperatingConditions(
            speed=speed,
            lambda_=lambda_,
            intake_pressure=intake_p,
            intake_temperature=intake_t
        )

        # Update geometry
        geometry.compression_ratio = cr
        geometry.__post_init__()

        # Random Wiebe parameters
        theta_0 = np.random.uniform(*bounds['theta_0'])
        delta_theta = np.random.uniform(*bounds['delta_theta'])
        a = np.random.uniform(*bounds['a'])
        eta = np.random.uniform(*bounds['eta'])

        # Crank angle array
        theta = np.linspace(-180, 180, 721)

        # Generate MFB profile
        if wiebe_type == 'single':
            m = np.random.uniform(*bounds['m'])
            x_b = single_wiebe(theta, theta_0, delta_theta, a, m)
            true_params = {
                'theta_0': theta_0,
                'delta_theta': delta_theta,
                'a': a,
                'm': m,
                'eta': eta
            }

        else:  # double
            m1 = np.random.uniform(*bounds['m1'])
            m2 = np.random.uniform(*bounds['m2'])
            lambda_w = np.random.uniform(*bounds['lambda_w'])
            k = np.random.uniform(*bounds['k'])
            x_b = double_wiebe(theta, theta_0, delta_theta, a, m1, m2, lambda_w, k)
            true_params = {
                'theta_0': theta_0,
                'delta_theta': delta_theta,
                'a': a,
                'm1': m1,
                'm2': m2,
                'lambda_w': lambda_w,
                'k': k,
                'eta': eta
            }

        # Simulate pressure
        P, T, Q = simulate_pressure_trace(
            theta, x_b, geometry, operating, gas_props, eta=eta
        )

        # Add noise
        noise = np.random.normal(0, noise_level * np.max(P), len(P))
        P_noisy = P + noise

        # Create case
        case = {
            'case_id': f'synth_{i:04d}',
            'speed': speed,
            'lambda': lambda_,
            'intake_pressure': intake_p,
            'intake_temperature': intake_t,
            'compression_ratio': cr,
            'crank_angle': theta,
            'pressure': P_noisy,
            'pressure_clean': P,  # Store clean version
            'temperature': T,
            'heat_release': Q,
            'mfb_true': x_b,
            'wiebe_params_true': true_params
        }

        cases.append(case)

    metadata = {
        'n_cases': n_cases,
        'wiebe_type': wiebe_type,
        'noise_level': noise_level,
        'synthetic': True
    }

    return CombustionDataset(
        cases=cases,
        feature_names=feature_names,
        metadata=metadata
    )


def save_dataset(dataset: CombustionDataset, save_path: str, format: str = 'pickle'):
    """
    Save dataset to file.

    Args:
        dataset: Dataset to save
        save_path: Output file path
        format: 'pickle' or 'json'
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'cases': dataset.cases,
        'feature_names': dataset.feature_names,
        'metadata': dataset.metadata
    }

    if format == 'pickle':
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        cases_json = []
        for case in dataset.cases:
            case_json = case.copy()
            for key, value in case_json.items():
                if isinstance(value, np.ndarray):
                    case_json[key] = value.tolist()
            cases_json.append(case_json)

        data['cases'] = cases_json

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Dataset saved to {path}")


if __name__ == "__main__":
    # Test synthetic data generation
    print("Generating synthetic dataset...")
    dataset = generate_synthetic_dataset(n_cases=50, wiebe_type='single')

    print(f"Dataset size: {len(dataset)} cases")
    print(f"Feature names: {dataset.feature_names}")
    print(f"Metadata: {dataset.metadata}")

    # Test case access
    case = dataset.get_case(0)
    print(f"\nExample case {case['case_id']}:")
    print(f"  Speed: {case['speed']:.1f} RPM")
    print(f"  Lambda: {case['lambda']:.2f}")
    print(f"  Peak pressure: {np.max(case['pressure'])/1e5:.2f} bar")
    print(f"  True Wiebe params: {case['wiebe_params_true']}")

    # Test split
    print("\nSplitting dataset...")
    train_data, val_data, test_data = dataset.split(test_size=0.2, val_size=0.1)
    print(f"Train: {len(train_data)} cases")
    print(f"Val: {len(val_data)} cases")
    print(f"Test: {len(test_data)} cases")

    # Test normalization
    print("\nNormalizing datasets...")
    train_data = normalize_dataset(train_data, fit=True)
    val_data.scaler = train_data.scaler
    val_data = normalize_dataset(val_data, fit=False)

    print("Train features (first 3 cases):")
    print(train_data.get_features()[:3])

    # Save datasets
    save_path = 'data/synthetic_train.pkl'
    save_dataset(train_data, save_path)

    print("\nData utilities test completed successfully!")
