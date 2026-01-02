"""
24-hour continuous training module for TSP parameter prediction model.

This module trains the neural network on randomly generated TSP instances
for a specified duration (default: 24 hours), using instances of varying sizes
from 10 to 550 cities. This ensures the model learns on diverse data rather
than benchmark-specific patterns.
"""
import numpy as np
import time
import os
import argparse
from datetime import datetime, timedelta
from typing import Tuple, List
from tsp_solver import TSPInstance, SimulatedAnnealingSolver, generate_random_tsp
from neural_network import ParameterPredictor
import random


def evaluate_parameters_for_instance(
    tsp_instance: TSPInstance,
    param_configs: List[Tuple[float, float, float, float]],
    n_runs: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate multiple parameter configurations on a TSP instance and return the best.
    
    Args:
        tsp_instance: TSP instance to evaluate on
        param_configs: List of (initial_temp, cooling_rate, min_temp, iterations_per_temp) tuples
        n_runs: Number of runs per configuration for stability
        
    Returns:
        Tuple of (best_features, best_parameters)
    """
    features = tsp_instance.get_features()
    best_params = None
    best_avg_length = float('inf')
    
    for params in param_configs:
        initial_temp, cooling_rate, min_temp, iterations_per_temp = params
        
        solver = SimulatedAnnealingSolver(
            initial_temperature=initial_temp,
            cooling_rate=cooling_rate,
            min_temperature=min_temp,
            iterations_per_temp=int(iterations_per_temp)
        )
        
        results = []
        for _ in range(n_runs):
            _, length = solver.solve(tsp_instance)
            results.append(length)
        
        avg_length = np.mean(results)
        
        if avg_length < best_avg_length:
            best_avg_length = avg_length
            best_params = np.array(params)
    
    return features, best_params


def generate_random_param_configs(n_configs: int = 15) -> List[Tuple[float, float, float, float]]:
    """
    Generate random parameter configurations for evaluation.
    
    Args:
        n_configs: Number of configurations to generate
        
    Returns:
        List of parameter tuples
    """
    configs = []
    for _ in range(n_configs):
        initial_temp = np.random.uniform(20, 200)
        cooling_rate = np.random.uniform(0.90, 0.999)
        min_temp = np.random.uniform(0.001, 0.1)
        iterations_per_temp = np.random.uniform(50, 300)
        configs.append((initial_temp, cooling_rate, min_temp, iterations_per_temp))
    return configs


def train_model_24h(
    duration_hours: float = 24.0,
    model_path: str = 'parameter_model_24h.pth',
    checkpoint_interval_minutes: int = 60,
    min_cities: int = 10,
    max_cities: int = 550,
    n_param_configs: int = 15,
    batch_training_size: int = 50,
    epochs_per_batch: int = 20,
    learning_rate: float = 0.001
):
    """
    Train the model continuously for a specified duration on random TSP instances.
    
    This function generates random TSP instances of varying sizes (min_cities to max_cities),
    evaluates different parameter configurations on each instance, and continuously trains
    the neural network on the accumulated data. The model is saved periodically.
    
    Args:
        duration_hours: Training duration in hours (default: 24.0)
        model_path: Path to save the model
        checkpoint_interval_minutes: Save model every N minutes
        min_cities: Minimum number of cities in generated instances
        max_cities: Maximum number of cities in generated instances
        n_param_configs: Number of parameter configurations to evaluate per instance
        batch_training_size: Number of instances to collect before retraining
        epochs_per_batch: Number of epochs to train on each batch
        learning_rate: Learning rate for training (default: 0.001)
    """
    print("=" * 80)
    print(f"STARTING 24-HOUR TRAINING SESSION")
    print("=" * 80)
    print(f"Duration: {duration_hours} hours")
    print(f"Model path: {model_path}")
    print(f"City range: {min_cities} - {max_cities}")
    print(f"Checkpoint interval: {checkpoint_interval_minutes} minutes")
    print(f"Batch size: {batch_training_size} instances")
    print(f"Epochs per batch: {epochs_per_batch}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    end_time = start_time + (duration_hours * 3600)
    last_checkpoint_time = start_time
    checkpoint_interval_seconds = checkpoint_interval_minutes * 60
    
    # Initialize predictor
    predictor = ParameterPredictor(model_path=model_path)
    
    # Training data accumulator
    all_features = []
    all_params = []
    
    # Load existing data if available
    if os.path.exists('training_features_24h.npy') and os.path.exists('training_params_24h.npy'):
        print("Loading existing training data...")
        all_features = list(np.load('training_features_24h.npy'))
        all_params = list(np.load('training_params_24h.npy'))
        print(f"Loaded {len(all_features)} existing samples")
        print()
    
    # Load existing model if available
    if os.path.exists(model_path):
        try:
            predictor.load()
            print(f"Loaded existing model from {model_path}")
            print()
        except Exception as e:
            print(f"Could not load existing model: {e}")
            print("Starting with fresh model")
            print()
    
    instance_count = 0
    batch_features = []
    batch_params = []
    
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Expected completion: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    while time.time() < end_time:
        # Generate random TSP instance with varying size
        n_cities = random.randint(min_cities, max_cities)
        tsp_instance = generate_random_tsp(n_cities)
        
        # Generate and evaluate parameter configurations
        param_configs = generate_random_param_configs(n_configs=n_param_configs)
        features, best_params = evaluate_parameters_for_instance(
            tsp_instance, param_configs, n_runs=3
        )
        
        # Add to batch
        batch_features.append(features)
        batch_params.append(best_params)
        instance_count += 1
        
        # Progress update
        elapsed_hours = (time.time() - start_time) / 3600
        remaining_hours = (end_time - time.time()) / 3600
        progress_pct = (time.time() - start_time) / (duration_hours * 3600) * 100
        
        if instance_count % 5 == 0:
            print(f"[{progress_pct:.1f}%] Processed {instance_count} instances | "
                  f"Elapsed: {elapsed_hours:.2f}h | Remaining: {remaining_hours:.2f}h | "
                  f"Last instance: {n_cities} cities")
        
        # Train on batch when enough data collected
        if len(batch_features) >= batch_training_size:
            print()
            print(f"Training on batch of {len(batch_features)} instances...")
            
            # Add to overall data
            all_features.extend(batch_features)
            all_params.extend(batch_params)
            
            # Train on all accumulated data
            predictor.train(
                all_features,
                all_params,
                epochs=epochs_per_batch,
                batch_size=min(32, len(all_features)),
                learning_rate=learning_rate
            )
            
            print(f"Total training samples: {len(all_features)}")
            
            # Clear batch
            batch_features = []
            batch_params = []
            
            # Save training data
            np.save('training_features_24h.npy', np.array(all_features))
            np.save('training_params_24h.npy', np.array(all_params))
            print(f"Saved training data: {len(all_features)} samples")
            print()
        
        # Periodic checkpoint
        if time.time() - last_checkpoint_time >= checkpoint_interval_seconds:
            print()
            print("=" * 80)
            print(f"CHECKPOINT at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            # Train on any remaining batch data
            if len(batch_features) > 0:
                all_features.extend(batch_features)
                all_params.extend(batch_params)
                
                predictor.train(
                    all_features,
                    all_params,
                    epochs=epochs_per_batch,
                    batch_size=min(32, len(all_features)),
                    learning_rate=learning_rate
                )
                
                batch_features = []
                batch_params = []
            
            # Save model and data
            predictor.save()
            np.save('training_features_24h.npy', np.array(all_features))
            np.save('training_params_24h.npy', np.array(all_params))
            
            print(f"Model saved to {model_path}")
            print(f"Total instances processed: {instance_count}")
            print(f"Total training samples: {len(all_features)}")
            print(f"Progress: {progress_pct:.1f}%")
            print(f"Time elapsed: {elapsed_hours:.2f} hours")
            print(f"Time remaining: {remaining_hours:.2f} hours")
            print("=" * 80)
            print()
            
            last_checkpoint_time = time.time()
    
    # Final training on any remaining data
    if len(batch_features) > 0:
        print()
        print("Final training on remaining batch...")
        all_features.extend(batch_features)
        all_params.extend(batch_params)
        
        predictor.train(
            all_features,
            all_params,
            epochs=epochs_per_batch * 2,  # Extra epochs for final training
            batch_size=min(32, len(all_features)),
            learning_rate=learning_rate
        )
    
    # Final save
    predictor.save()
    np.save('training_features_24h.npy', np.array(all_features))
    np.save('training_params_24h.npy', np.array(all_params))
    
    total_time = (time.time() - start_time) / 3600
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} hours")
    print(f"Total instances processed: {instance_count}")
    print(f"Total training samples: {len(all_features)}")
    print(f"Model saved to: {model_path}")
    print(f"Training data saved to: training_features_24h.npy, training_params_24h.npy")
    print("=" * 80)


def main():
    """Main entry point for 24-hour training."""
    parser = argparse.ArgumentParser(
        description='Train TSP parameter prediction model for 24 hours on random instances'
    )
    
    parser.add_argument('--duration', type=float, default=24.0,
                       help='Training duration in hours (default: 24.0)')
    parser.add_argument('--model-path', type=str, default='parameter_model_24h.pth',
                       help='Path to save the trained model (default: parameter_model_24h.pth)')
    parser.add_argument('--checkpoint-interval', type=int, default=60,
                       help='Save checkpoint every N minutes (default: 60)')
    parser.add_argument('--min-cities', type=int, default=10,
                       help='Minimum number of cities (default: 10)')
    parser.add_argument('--max-cities', type=int, default=550,
                       help='Maximum number of cities (default: 550)')
    parser.add_argument('--param-configs', type=int, default=15,
                       help='Number of parameter configurations to test per instance (default: 15)')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for training (default: 50)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Epochs per batch (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for training (default: 0.001)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.duration <= 0:
        print("Error: Duration must be positive")
        return
    
    if args.min_cities < 3:
        print("Error: Minimum cities must be at least 3")
        return
    
    if args.max_cities < args.min_cities:
        print("Error: Maximum cities must be >= minimum cities")
        return
    
    # Run training
    train_model_24h(
        duration_hours=args.duration,
        model_path=args.model_path,
        checkpoint_interval_minutes=args.checkpoint_interval,
        min_cities=args.min_cities,
        max_cities=args.max_cities,
        n_param_configs=args.param_configs,
        batch_training_size=args.batch_size,
        epochs_per_batch=args.epochs,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
