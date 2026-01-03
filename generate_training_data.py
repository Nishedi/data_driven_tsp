"""
Generate training data by testing different SA parameters on various TSP instances.
"""
import numpy as np
from tsp_solver import TSPInstance, SimulatedAnnealingSolver, generate_random_tsp
from typing import List, Tuple
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def evaluate_parameters(args: Tuple) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Evaluate a parameter configuration on a TSP instance.
    
    Args:
        args: Tuple of (tsp_instance, initial_temp, cooling_rate, min_temp, iterations_per_temp)
        
    Returns:
        Tuple of (features, parameters, solution_quality)
    """
    tsp_instance, initial_temp, cooling_rate, min_temp, iterations_per_temp = args
    
    solver = SimulatedAnnealingSolver(
        initial_temperature=initial_temp,
        cooling_rate=cooling_rate,
        min_temperature=min_temp,
        iterations_per_temp=int(iterations_per_temp)
    )
    
    # Run solver multiple times and take average
    results = []
    for _ in range(3):  # Run 3 times for stability
        _, length = solver.solve(tsp_instance)
        results.append(length)
    
    avg_length = np.mean(results)
    
    features = tsp_instance.get_features()
    parameters = np.array([initial_temp, cooling_rate, min_temp, iterations_per_temp])
    
    return features, parameters, avg_length


def generate_training_data(
    n_instances: int = 50,
    city_range: Tuple[int, int] = (10, 50),
    n_param_configs: int = 20,
    n_workers: int = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate training data by testing various parameter configurations.
    
    Args:
        n_instances: Number of TSP instances to generate
        city_range: Range of number of cities (min, max)
        n_param_configs: Number of parameter configurations to test per instance
        n_workers: Number of parallel workers (default: CPU count)
        
    Returns:
        Tuple of (features, optimal_parameters)
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Generating training data with {n_instances} instances...")
    print(f"Using {n_workers} workers")
    
    all_features = []
    all_optimal_params = []
    
    # Generate TSP instances
    for i in range(n_instances):
        n_cities = np.random.randint(city_range[0], city_range[1] + 1)
        tsp_instance = generate_random_tsp(n_cities, seed=i)
        
        # Sample parameter configurations to test
        param_configs = []
        for _ in range(n_param_configs):
            initial_temp = np.random.uniform(20, 200)
            cooling_rate = np.random.uniform(0.95, 0.999)  # Higher cooling rates for slower, better convergence
            min_temp = np.random.uniform(0.001, 0.1)
            iterations_per_temp = np.random.uniform(50, 300)
            param_configs.append((initial_temp, cooling_rate, min_temp, iterations_per_temp))
        
        # Evaluate all configurations
        tasks = [(tsp_instance, *params) for params in param_configs]
        
        results = []
        if n_workers > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(evaluate_parameters, tasks))
        else:
            results = [evaluate_parameters(task) for task in tasks]
        
        # Find best parameters for this instance
        best_idx = np.argmin([r[2] for r in results])
        best_features, best_params, best_quality = results[best_idx]
        
        all_features.append(best_features)
        all_optimal_params.append(best_params)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_instances} instances")
    
    print("Training data generation complete!")
    return all_features, all_optimal_params


def quick_generate_training_data(n_instances: int = 100) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Quick generation with reasonable heuristics (faster than exhaustive search).
    
    Args:
        n_instances: Number of TSP instances
        
    Returns:
        Tuple of (features, parameters)
    """
    print(f"Generating quick training data with {n_instances} instances...")
    
    all_features = []
    all_params = []
    
    for i in range(n_instances):
        # Generate random TSP
        n_cities = np.random.randint(10, 51)
        tsp_instance = generate_random_tsp(n_cities, seed=i)
        
        features = tsp_instance.get_features()
        
        # Heuristic: larger/more complex problems need more iterations and higher initial temp
        # This is a simplified heuristic for quick data generation
        initial_temp = 50 + features[0] * 2  # Based on number of cities
        cooling_rate = 0.95 + (1 - features[0] / 100) * 0.049  # Slower cooling for larger problems (range: 0.95-0.999)
        min_temp = 0.01
        iterations_per_temp = 100 + int(features[0] * 2)  # More iterations for larger problems
        
        params = np.array([initial_temp, cooling_rate, min_temp, iterations_per_temp])
        
        all_features.append(features)
        all_params.append(params)
    
    print("Quick training data generation complete!")
    return all_features, all_params


if __name__ == "__main__":
    # Generate training data
    features, params = quick_generate_training_data(n_instances=200)
    
    # Save data
    np.save('training_features.npy', np.array(features))
    np.save('training_params.npy', np.array(params))
    
    print(f"Saved {len(features)} training samples")
    print(f"Feature shape: {features[0].shape}")
    print(f"Parameter shape: {params[0].shape}")
