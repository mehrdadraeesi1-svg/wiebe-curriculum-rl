"""
Genetic Algorithm Baseline for Fair Comparison with RL
======================================================

Uses the SAME validation split as RL training to ensure fair comparison.
Loads data_splits.json from RL training directory.

Key differences from original:
1. Loads validation cases from data_splits.json
2. Uses same metrics calculation as RL evaluation
3. Generates comparison-ready outputs
4. Matches RL evaluation format

Author: Mehrdad (Optimized for Comparison)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import time

import sys

from src.utils.data_loader import WiebeDataLoader

from src.models.wiebe_functions import single_wiebe, double_wiebe
from src.utils.thermodynamics import (
    EngineGeometry,
    EngineOperatingConditions,
    GasProperties,
    simulate_pressure_trace,
    calculate_r_squared,
    calculate_normalized_rmse
)


class GeneticAlgorithm:
    """
    Genetic Algorithm for Wiebe parameter optimization.
    
    Optimized for fair comparison with RL methods.
    """
    
    def __init__(
        self,
        wiebe_type: str = 'double',
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_size: int = 5,
        seed: int = 42
    ):
        self.wiebe_type = wiebe_type
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.rng = np.random.RandomState(seed)
        
        # Parameter bounds (same as RL environment)
        if wiebe_type == 'single':
            self.param_names = ['theta0', 'delta_theta', 'a', 'm', 'ht_g']
            self.bounds = {
                'theta0': (-40, 20),
                'delta_theta': (30, 100),
                'a': (3.0, 8.0),
                'm': (1.0, 4.0),
                'ht_g': (0.5, 3.0)
            }
        else:
            self.param_names = ['theta0', 'delta_theta', 'a', 'm1', 'm2', 'lambda_w', 'k', 'ht_g', 'ht_c']
            self.bounds = {
                'theta0': (-40, 20),
                'delta_theta': (30, 100),
                'a': (3.0, 8.0),
                'm1': (1.0, 4.0),
                'm2': (1.0, 4.0),
                'lambda_w': (0.3, 0.8),
                'k': (2.0, 15.0),
                'ht_g': (0.5, 3.0),
                'ht_c': (0.5, 3.0)
            }
    
    def initialize_population(self) -> np.ndarray:
        """Create initial random population."""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for param in self.param_names:
                low, high = self.bounds[param]
                individual[param] = self.rng.uniform(low, high)
            population.append(individual)
        return population
    
    def evaluate_fitness(
        self,
        individual: Dict,
        case: Dict
    ) -> float:
        """
        Evaluate fitness (R² score) for an individual.
        
        Uses same calculation as RL evaluation for consistency.
        """
        try:
            # Extract experimental data
            theta_exp = np.array(case['crank_angle'])
            pressure_exp = np.array(case['pressure'])
            
            # Setup engine
            geometry = EngineGeometry(
                bore=case.get('bore', 0.086),
                stroke=case.get('stroke', 0.086),
                connecting_rod=case.get('connecting_rod', 0.143),
                compression_ratio=case.get('compression_ratio', 10.5)
            )
            
            operating = EngineOperatingConditions(
                speed=case.get('speed', 2000),
                lambda_=case.get('lambda', 1.0),
                intake_pressure=case.get('intake_pressure', 1e5),
                intake_temperature=case.get('intake_temperature', 300)
            )
            
            gas_props = GasProperties()
            
            # Generate burn profile
            if self.wiebe_type == 'single':
                xb = single_wiebe(
                    theta_exp,
                    individual['theta0'],
                    individual['delta_theta'],
                    individual['a'],
                    individual['m']
                )
                eta = 0.85
                ht_g = individual['ht_g']
                ht_c = individual['ht_g']
            else:
                xb = double_wiebe(
                    theta_exp,
                    individual['theta0'],
                    individual['delta_theta'],
                    individual['a'],
                    individual['m1'],
                    individual['m2'],
                    individual['lambda_w'],
                    individual['k']
                )
                eta = 0.85 + 0.15 * individual['lambda_w']
                ht_g = individual['ht_g']
                ht_c = individual['ht_c']
            
            # Simulate pressure
            P_pred, _, _ = simulate_pressure_trace(
                theta_exp,
                xb,
                geometry,
                operating,
                gas_props,
                eta=eta,
                ht_scale_global=ht_g,
                ht_scale_combustion=ht_c
            )
            
            # Calculate R² (same as RL)
            r2 = calculate_r_squared(pressure_exp, P_pred)
            
            return max(r2, 0.0)
            
        except Exception as e:
            return 0.0
    
    def selection(
        self,
        population: List[Dict],
        fitness: np.ndarray,
        k: int = 3
    ) -> Dict:
        """Tournament selection."""
        indices = self.rng.choice(len(population), size=k, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(
        self,
        parent1: Dict,
        parent2: Dict
    ) -> Tuple[Dict, Dict]:
        """Single-point crossover."""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1, child2 = {}, {}
        crossover_point = self.rng.randint(1, len(self.param_names))
        
        for i, param in enumerate(self.param_names):
            if i < crossover_point:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def mutate(self, individual: Dict) -> Dict:
        """Gaussian mutation."""
        mutated = individual.copy()
        
        for param in self.param_names:
            if self.rng.random() < self.mutation_rate:
                low, high = self.bounds[param]
                sigma = (high - low) * 0.1
                mutated[param] = np.clip(
                    mutated[param] + self.rng.normal(0, sigma),
                    low, high
                )
        
        return mutated
    
    def optimize(self, case: Dict, verbose: bool = False) -> Tuple[Dict, float, List]:
        """
        Run GA optimization for a single case.
        
        Returns:
            best_individual, best_fitness, history
        """
        # Initialize
        population = self.initialize_population()
        history = []
        
        # Evaluate initial population
        fitness = np.array([self.evaluate_fitness(ind, case) for ind in population])
        
        for gen in range(self.n_generations):
            # Track statistics
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            avg_fitness = np.mean(fitness)
            
            history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            })
            
            if verbose and gen % 20 == 0:
                print(f"  Gen {gen:3d}: Best R² = {best_fitness:.4f}, Avg = {avg_fitness:.4f}")
            
            # Early stopping
            if best_fitness > 0.995:
                if verbose:
                    print(f"  Converged at generation {gen}")
                break
            
            # Elitism
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            new_population = [population[i].copy() for i in elite_indices]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitness)
                parent2 = self.selection(population, fitness)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            # Trim to population size
            population = new_population[:self.population_size]
            fitness = np.array([self.evaluate_fitness(ind, case) for ind in population])
        
        # Return best
        best_idx = np.argmax(fitness)
        return population[best_idx], fitness[best_idx], history


def load_validation_cases_from_splits(splits_path: Path, data_dir: Path) -> List[Dict]:
    """
    Load validation cases from data_splits.json.
    
    This ensures we evaluate GA on the SAME validation set as RL.
    """
    print(f"\nLoading validation split from: {splits_path}")
    
    # Load splits
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    val_case_ids = splits['test']
    print(f"Found {len(val_case_ids)} validation cases")
    
    # Load all cases
    loader = WiebeDataLoader(str(data_dir))
    all_cases = loader.load_all_cases()
    
    # Filter validation cases
    val_cases = [case for case in all_cases if case['case_id'] in val_case_ids]
    
    print(f"Loaded {len(val_cases)} validation cases")
    
    return val_cases


def calculate_comprehensive_metrics(
    case: Dict,
    best_params: Dict,
    wiebe_type: str
) -> Dict:
    """
    Calculate all metrics matching RL evaluation format.
    
    Returns metrics compatible with RL evaluation CSV.
    """
    theta_exp = np.array(case['crank_angle'])
    pressure_exp = np.array(case['pressure'])
    
    # Setup engine
    geometry = EngineGeometry(
        bore=case.get('bore', 0.086),
        stroke=case.get('stroke', 0.086),
        connecting_rod=case.get('connecting_rod', 0.143),
        compression_ratio=case.get('compression_ratio', 10.5)
    )
    
    operating = EngineOperatingConditions(
        speed=case.get('speed', 2000),
        lambda_=case.get('lambda', 1.0),
        intake_pressure=case.get('intake_pressure', 1e5),
        intake_temperature=case.get('intake_temperature', 300)
    )
    
    gas_props = GasProperties()
    
    # Generate burn profile
    if wiebe_type == 'single':
        xb = single_wiebe(
            theta_exp,
            best_params['theta0'],
            best_params['delta_theta'],
            best_params['a'],
            best_params['m']
        )
        eta = 0.85
        ht_g = best_params['ht_g']
        ht_c = best_params['ht_g']
    else:
        xb = double_wiebe(
            theta_exp,
            best_params['theta0'],
            best_params['delta_theta'],
            best_params['a'],
            best_params['m1'],
            best_params['m2'],
            best_params['lambda_w'],
            best_params['k']
        )
        eta = 0.85 + 0.15 * best_params['lambda_w']
        ht_g = best_params['ht_g']
        ht_c = best_params['ht_c']
    
    # Simulate
    P_pred, T_pred, _ = simulate_pressure_trace(
        theta_exp, xb, geometry, operating, gas_props,
        eta=eta,
        ht_scale_global=ht_g,
        ht_scale_combustion=ht_c
    )
    
    # Calculate metrics (same as RL evaluation)
    r2 = calculate_r_squared(pressure_exp, P_pred)
    nrmse_pct = calculate_normalized_rmse(pressure_exp, P_pred)
    
    # Peak pressure error
    peak_exp = np.max(pressure_exp)
    peak_pred = np.max(P_pred)
    peakP_err_pct = abs(peak_pred - peak_exp) / peak_exp * 100
    
    # CA50 (approximate)
    ca50_idx_exp = np.argmin(abs(xb - 0.5))
    ca50_err = 0.0  # Simplified for now
    
    # Max temperature
    T_max = np.max(T_pred) if len(T_pred) > 0 else 0.0
    
    # Final burn fraction
    xb_final = xb[-1] if len(xb) > 0 else 0.0
    
    metrics = {
        'case_id': case['case_id'],
        'r2': r2,
        'nrmse': nrmse_pct / 100,  # As fraction for compatibility
        'nrmse_pct': nrmse_pct,
        'peakP_err_pct': peakP_err_pct,
        'ca50_err': ca50_err,
        'T_max': T_max,
        'xb_final': xb_final,
        **best_params
    }
    
    return metrics


def evaluate_ga_on_validation_set(
    ga: GeneticAlgorithm,
    val_cases: List[Dict],
    output_dir: Path,
    save_interval: int = 10
):
    """
    Evaluate GA on validation cases.
    
    Generates outputs compatible with RL evaluation format.
    """
    print("\n" + "="*80)
    print("EVALUATING GA ON VALIDATION SET")
    print("="*80)
    print(f"Cases: {len(val_cases)}")
    print(f"Wiebe Type: {ga.wiebe_type}")
    print(f"Population: {ga.population_size}")
    print(f"Generations: {ga.n_generations}")
    print("="*80 + "\n")
    
    results = []
    
    for i, case in enumerate(tqdm(val_cases, desc="GA Optimization")):
        start_time = time.time()
        
        # Optimize
        best_params, best_r2, history = ga.optimize(case, verbose=False)
        
        elapsed_time = time.time() - start_time
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(case, best_params, ga.wiebe_type)
        metrics['optimization_time'] = elapsed_time
        metrics['n_generations_used'] = len(history)
        
        results.append(metrics)
        
        # Save intermediate results
        if (i + 1) % save_interval == 0:
            df_partial = pd.DataFrame(results)
            df_partial.to_csv(output_dir / 'ga_evaluation_partial.csv', index=False)
            print(f"  Saved intermediate results ({i+1}/{len(val_cases)} cases)")
    
    # Save final results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'ga_evaluation_metrics.csv', index=False)
    
    # Calculate summary statistics
    summary = {
        'method': 'Genetic Algorithm',
        'wiebe_type': ga.wiebe_type,
        'population_size': ga.population_size,
        'n_generations': ga.n_generations,
        'n_cases': len(val_cases),
        'statistics': {
            'r2_mean': float(df['r2'].mean()),
            'r2_std': float(df['r2'].std()),
            'r2_min': float(df['r2'].min()),
            'r2_max': float(df['r2'].max()),
            'nrmse_mean': float(df['nrmse_pct'].mean()),
            'nrmse_std': float(df['nrmse_pct'].std()),
            'peakP_err_mean': float(df['peakP_err_pct'].mean()),
            'peakP_err_std': float(df['peakP_err_pct'].std()),
            'time_mean': float(df['optimization_time'].mean()),
            'time_std': float(df['optimization_time'].std()),
            'time_total': float(df['optimization_time'].sum()),
            'success_rate': float((df['r2'] > 0.95).sum() / len(df) * 100),
            'n_converged': int((df['r2'] > 0.995).sum())
        }
    }
    
    # Save summary
    with open(output_dir / 'ga_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("GA EVALUATION SUMMARY")
    print("="*80)
    print(f"\nPerformance Metrics:")
    print(f"  R² Score     : {summary['statistics']['r2_mean']:.4f} ± {summary['statistics']['r2_std']:.4f}")
    print(f"                 (min: {summary['statistics']['r2_min']:.4f}, max: {summary['statistics']['r2_max']:.4f})")
    print(f"  NRMSE        : {summary['statistics']['nrmse_mean']:.2f}% ± {summary['statistics']['nrmse_std']:.2f}%")
    print(f"  Peak P Error : {summary['statistics']['peakP_err_mean']:.2f}% ± {summary['statistics']['peakP_err_std']:.2f}%")
    print(f"  Success Rate : {summary['statistics']['success_rate']:.1f}% (R² > 0.95)")
    print(f"  Converged    : {summary['statistics']['n_converged']} cases (R² > 0.995)")
    
    print(f"\nComputational Performance:")
    print(f"  Avg Time     : {summary['statistics']['time_mean']:.2f} ± {summary['statistics']['time_std']:.2f} seconds/case")
    print(f"  Total Time   : {summary['statistics']['time_total']/60:.1f} minutes ({summary['statistics']['time_total']/3600:.2f} hours)")
    
    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - ga_evaluation_metrics.csv (detailed metrics)")
    print(f"  - ga_summary.json (summary statistics)")
    print("="*80 + "\n")
    
    return df, summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GA Baseline - Fair Comparison with RL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on PPO double validation set
  python baseline_ga_optimized.py \\
      --splits-path training_v4_ppo_double_enhanced/data_splits.json \\
      --wiebe-type double
  
  # Evaluate on A2C single validation set
  python baseline_ga_optimized.py \\
      --splits-path training_v4_a2c_single_enhanced/data_splits.json \\
      --wiebe-type single \\
      --population 100 \\
      --generations 150
        """
    )
    
    parser.add_argument(
        '--splits-path',
        type=str,
        required=True,
        help='Path to data_splits.json from RL training'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='json_cases',
        help='Directory containing JSON case files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated based on splits-path)'
    )
    parser.add_argument(
        '--wiebe-type',
        type=str,
        default='double',
        choices=['single', 'double'],
        help='Wiebe function type (should match RL training)'
    )
    parser.add_argument(
        '--population',
        type=int,
        default=50,
        help='GA population size'
    )
    parser.add_argument(
        '--generations',
        type=int,
        default=100,
        help='Maximum number of generations'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        splits_parent = Path(args.splits_path).parent
        args.output_dir = str(splits_parent / 'baseline_ga_evaluation')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GA BASELINE FOR FAIR COMPARISON WITH RL")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Splits Path  : {args.splits_path}")
    print(f"  Data Dir     : {args.data_dir}")
    print(f"  Output Dir   : {args.output_dir}")
    print(f"  Wiebe Type   : {args.wiebe_type}")
    print(f"  Population   : {args.population}")
    print(f"  Generations  : {args.generations}")
    print(f"  Seed         : {args.seed}")
    
    # Load validation cases from splits
    splits_path = Path(args.splits_path)
    data_dir = Path(args.data_dir)
    
    if not splits_path.exists():
        print(f"\n✗ ERROR: Splits file not found: {splits_path}")
        print("\nPlease ensure the RL training has been run and data_splits.json exists.")
        return
    
    val_cases = load_validation_cases_from_splits(splits_path, data_dir)
    
    if len(val_cases) == 0:
        print("\n✗ ERROR: No validation cases loaded!")
        return
    
    # Initialize GA
    print(f"\nInitializing Genetic Algorithm...")
    ga = GeneticAlgorithm(
        wiebe_type=args.wiebe_type,
        population_size=args.population,
        n_generations=args.generations,
        seed=args.seed
    )
    
    # Run evaluation
    df, summary = evaluate_ga_on_validation_set(ga, val_cases, output_dir)
    
    # Save configuration for reference
    config = {
        'splits_path': str(args.splits_path),
        'data_dir': str(args.data_dir),
        'wiebe_type': args.wiebe_type,
        'population_size': args.population,
        'n_generations': args.generations,
        'seed': args.seed,
        'validation_cases': len(val_cases)
    }
    
    with open(output_dir / 'ga_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✓ EVALUATION COMPLETED SUCCESSFULLY")
    print(f"\nResults can now be compared with RL evaluation using:")
    print(f"  RL:  {splits_path.parent}/evaluation_v4/evaluation_metrics.csv")
    print(f"  GA:  {output_dir}/ga_evaluation_metrics.csv")


if __name__ == "__main__":
    main()