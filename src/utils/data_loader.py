"""
Data loader for Wiebe RL training.
Loads JSON case files and converts them to the format expected by the environment.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WiebeDataLoader:
    """Load and prepare Wiebe training data from JSON files."""
    
    def __init__(self, json_dir: str = 'json_cases_unseen'):
        """
        Initialize data loader.
        
        Args:
            json_dir: Directory containing JSON case files
        """
        self.json_dir = Path(json_dir)
        self.cases = []
        
        if not self.json_dir.exists():
            raise ValueError(f"Directory {json_dir} not found!")
        
        logger.info(f"Data loader initialized for directory: {json_dir}")
    
    def load_all_cases(self, max_cases: int = None) -> List[Dict[str, Any]]:
        """
        Load all JSON cases from directory.
        
        Args:
            max_cases: Maximum number of cases to load (None = all)
        
        Returns:
            List of case dictionaries
        """
        json_files = sorted(list(self.json_dir.glob('*.json')))
        
        if max_cases is not None:
            json_files = json_files[:max_cases]
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        self.cases = []
        failed_count = 0
        
        for json_file in json_files:
            try:
                case = self._load_single_case(json_file)
                if case is not None:
                    self.cases.append(case)
            except Exception as e:
                logger.warning(f"Failed to load {json_file.name}: {e}")
                failed_count += 1
        
        logger.info(f"Successfully loaded {len(self.cases)} cases")
        if failed_count > 0:
            logger.warning(f"Failed to load {failed_count} cases")
        
        return self.cases
    
    def _load_single_case(self, json_file: Path) -> Dict[str, Any]:
        """
        Load a single JSON case and convert to environment format.
        
        Args:
            json_file: Path to JSON file
        
        Returns:
            Case dictionary in environment format
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract geometry
        geom = data['engine_geometry']
        
        # Extract operating conditions
        ops = data['operating_conditions']
        
        # Extract pressure trace
        pressure_data = data['pressure_trace']
        theta = np.array(pressure_data['theta_deg'])
        pressure_Pa = np.array(pressure_data['pressure_Pa'])
        
        # Validate data
        if len(theta) != len(pressure_Pa):
            logger.warning(f"Theta and pressure length mismatch in {json_file.name}")
            return None
        
        if len(theta) == 0:
            logger.warning(f"Empty pressure trace in {json_file.name}")
            return None
        
        # Create case dictionary
        case = {
            # Identification
            'case_id': data.get('logName', json_file.stem).strip(),
            'source_file': json_file.name,
            
            # Operating conditions (basic)
            'speed': float(ops['speed_rpm']),
            'lambda': float(ops['lambda']),
            'intake_pressure': float(ops['intake_pressure_Pa']),
            'intake_temperature': float(ops['intake_temperature_K']),
            
            # Geometry
            'bore': float(geom['bore_m']),
            'stroke': float(geom['stroke_m']),
            'connecting_rod': float(geom['connecting_rod_m']),
            'compression_ratio': float(geom['compression_ratio']),
            
            # Additional operating conditions
            'Tw_wall_K': float(ops.get('Tw_wall_K', 420.0)),
            'gamma': float(ops.get('gamma', 1.35)),
            'fuel_LHV_J_per_kg': float(ops.get('fuel_LHV_J_per_kg', 44000000.0)),
            'spark_timing': float(ops.get('T_ign_deg', 0.0)),
            'injection_timing': float(ops.get('T_inj_deg', 0.0)) if ops.get('T_inj_deg') is not None else None,
            
            # Pressure trace
            'crank_angle': theta,
            'pressure': pressure_Pa
        }
        
        return case
    
    def split_train_test(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split cases into train, validation, and test sets.
        
        Args:
            test_size: Fraction for test set (0.0 to 1.0)
            val_size: Fraction of training data for validation (0.0 to 1.0)
            random_state: Random seed for reproducibility
        
        Returns:
            (train_cases, val_cases, test_cases)
        """
        if len(self.cases) == 0:
            raise ValueError("No cases loaded! Call load_all_cases() first.")
        
        # First split: train+val vs test
        train_val_cases, test_cases = train_test_split(
            self.cases,
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: train vs val
        if val_size > 0:
            train_cases, val_cases = train_test_split(
                train_val_cases,
                test_size=val_size,
                random_state=random_state
            )
        else:
            train_cases = train_val_cases
            val_cases = []
        
        logger.info(f"Data split: Train={len(train_cases)}, "
                   f"Val={len(val_cases)}, Test={len(test_cases)}")
        
        return train_cases, val_cases, test_cases
    
    def get_normalization_stats(
        self,
        cases: List[Dict[str, Any]],
        features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate normalization statistics from training data.
        
        Args:
            cases: List of case dictionaries
            features: List of feature names to include
        
        Returns:
            (mean, std) arrays
        """
        feature_values = []
        
        for case in cases:
            values = []
            for feature in features:
                if feature in case:
                    val = case[feature]
                    if val is not None:
                        values.append(float(val))
                    else:
                        values.append(0.0)
                else:
                    values.append(0.0)
            feature_values.append(values)
        
        feature_array = np.array(feature_values)
        mean = np.mean(feature_array, axis=0)
        std = np.std(feature_array, axis=0)
        
        # Prevent division by zero
        std = np.where(std < 1e-8, 1.0, std)
        
        logger.info(f"Normalization stats computed for {len(features)} features")
        
        return mean, std
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if len(self.cases) == 0:
            return {}
        
        speeds = [c['speed'] for c in self.cases]
        lambdas = [c['lambda'] for c in self.cases]
        intake_p = [c['intake_pressure'] for c in self.cases]
        cr = [c['compression_ratio'] for c in self.cases]
        
        stats = {
            'total_cases': len(self.cases),
            'speed': {
                'min': min(speeds),
                'max': max(speeds),
                'mean': np.mean(speeds),
                'std': np.std(speeds)
            },
            'lambda': {
                'min': min(lambdas),
                'max': max(lambdas),
                'mean': np.mean(lambdas),
                'std': np.std(lambdas)
            },
            'intake_pressure': {
                'min': min(intake_p),
                'max': max(intake_p),
                'mean': np.mean(intake_p),
                'std': np.std(intake_p)
            },
            'compression_ratio': {
                'min': min(cr),
                'max': max(cr),
                'mean': np.mean(cr),
                'std': np.std(cr)
            }
        }
        
        return stats
    
    def print_dataset_info(self):
        """Print information about the loaded dataset."""
        if len(self.cases) == 0:
            print("No cases loaded!")
            return
        
        stats = self.get_dataset_statistics()
        
        print("\n" + "="*70)
        print("DATASET INFORMATION")
        print("="*70)
        print(f"Total cases: {stats['total_cases']}")
        print(f"\nSpeed (RPM):")
        print(f"  Range: {stats['speed']['min']:.0f} - {stats['speed']['max']:.0f}")
        print(f"  Mean ± Std: {stats['speed']['mean']:.0f} ± {stats['speed']['std']:.0f}")
        print(f"\nLambda:")
        print(f"  Range: {stats['lambda']['min']:.3f} - {stats['lambda']['max']:.3f}")
        print(f"  Mean ± Std: {stats['lambda']['mean']:.3f} ± {stats['lambda']['std']:.3f}")
        print(f"\nIntake Pressure (Pa):")
        print(f"  Range: {stats['intake_pressure']['min']:.0f} - {stats['intake_pressure']['max']:.0f}")
        print(f"  Mean ± Std: {stats['intake_pressure']['mean']:.0f} ± {stats['intake_pressure']['std']:.0f}")
        print(f"\nCompression Ratio:")
        print(f"  Range: {stats['compression_ratio']['min']:.1f} - {stats['compression_ratio']['max']:.1f}")
        print(f"  Mean ± Std: {stats['compression_ratio']['mean']:.1f} ± {stats['compression_ratio']['std']:.1f}")
        print("="*70 + "\n")


if __name__ == "__main__":
    # Test the data loader
    loader = WiebeDataLoader('json_cases_unseen')
    
    # Load all cases
    cases = loader.load_all_cases()
    print(f"Loaded {len(cases)} cases")
    
    # Print dataset info
    loader.print_dataset_info()
    
    # Split data
    train, val, test = loader.split_train_test(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(train)} cases")
    print(f"  Validation: {len(val)} cases")
    print(f"  Test: {len(test)} cases")
    
    # Show example case
    if len(cases) > 0:
        print(f"\nExample case:")
        example = cases[0]
        print(f"  ID: {example['case_id']}")
        print(f"  Speed: {example['speed']:.0f} RPM")
        print(f"  Lambda: {example['lambda']:.3f}")
        print(f"  Intake P: {example['intake_pressure']:.0f} Pa")
        print(f"  CR: {example['compression_ratio']:.1f}")
        print(f"  Spark timing: {example['spark_timing']:.1f}°CA")
        print(f"  Pressure points: {len(example['pressure'])}")
