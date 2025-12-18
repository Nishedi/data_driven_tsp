"""
Main script for training and using the data-driven TSP solver.
"""
import numpy as np
import argparse
import os
from tsp_solver import TSPInstance, SimulatedAnnealingSolver, generate_random_tsp
from neural_network import ParameterPredictor
from generate_training_data import quick_generate_training_data
import matplotlib.pyplot as plt


def train_model(n_instances: int = 200, epochs: int = 100):
    """
    Train the parameter prediction model.
    
    Args:
        n_instances: Number of training instances
        epochs: Number of training epochs
    """
    print("=" * 60)
    print("TRAINING PARAMETER PREDICTION MODEL")
    print("=" * 60)
    
    # Check if training data exists
    if os.path.exists('training_features.npy') and os.path.exists('training_params.npy'):
        print("Loading existing training data...")
        features = np.load('training_features.npy')
        params = np.load('training_params.npy')
        features_list = list(features)
        params_list = list(params)
    else:
        print("Generating training data...")
        features_list, params_list = quick_generate_training_data(n_instances=n_instances)
        # Save for future use
        np.save('training_features.npy', np.array(features_list))
        np.save('training_params.npy', np.array(params_list))
    
    print(f"Training with {len(features_list)} instances")
    
    # Create and train predictor
    predictor = ParameterPredictor(model_path='parameter_model.pth')
    predictor.train(features_list, params_list, epochs=epochs, batch_size=32, learning_rate=0.001)
    
    # Save model
    predictor.save()
    
    print("\nTraining complete!")


def solve_tsp(n_cities: int = 20, use_nn: bool = True):
    """
    Solve a TSP instance with or without neural network parameter selection.
    
    Args:
        n_cities: Number of cities
        use_nn: Whether to use neural network for parameter selection
    """
    print("=" * 60)
    print(f"SOLVING TSP WITH {n_cities} CITIES")
    print("=" * 60)
    
    # Generate TSP instance
    tsp_instance = generate_random_tsp(n_cities)
    features = tsp_instance.get_features()
    
    print(f"\nTSP Instance Features:")
    print(f"  Number of cities: {n_cities}")
    print(f"  Average distance: {features[1]:.2f}")
    print(f"  Distance std dev: {features[2]:.2f}")
    
    if use_nn:
        # Use neural network to predict parameters
        if not os.path.exists('parameter_model.pth'):
            print("\nError: Model not found. Please train the model first using --train")
            return
        
        predictor = ParameterPredictor(model_path='parameter_model.pth')
        predictor.load()
        
        predicted_params = predictor.predict(features)
        initial_temp = predicted_params[0]
        cooling_rate = predicted_params[1]
        min_temp = predicted_params[2]
        iterations_per_temp = int(predicted_params[3])
        
        print("\nNeural Network Predicted Parameters:")
        print(f"  Initial temperature: {initial_temp:.2f}")
        print(f"  Cooling rate: {cooling_rate:.4f}")
        print(f"  Minimum temperature: {min_temp:.4f}")
        print(f"  Iterations per temperature: {iterations_per_temp}")
    else:
        # Use default parameters
        initial_temp = 100.0
        cooling_rate = 0.995
        min_temp = 0.01
        iterations_per_temp = 100
        
        print("\nUsing Default Parameters:")
        print(f"  Initial temperature: {initial_temp:.2f}")
        print(f"  Cooling rate: {cooling_rate:.4f}")
        print(f"  Minimum temperature: {min_temp:.4f}")
        print(f"  Iterations per temperature: {iterations_per_temp}")
    
    # Solve TSP
    solver = SimulatedAnnealingSolver(
        initial_temperature=initial_temp,
        cooling_rate=cooling_rate,
        min_temperature=min_temp,
        iterations_per_temp=iterations_per_temp
    )
    
    print("\nSolving TSP...")
    best_tour, best_length = solver.solve(tsp_instance)
    
    print(f"\nSolution found!")
    print(f"  Tour length: {best_length:.2f}")
    print(f"  Tour: {best_tour[:10]}..." if len(best_tour) > 10 else f"  Tour: {best_tour}")
    
    # Visualize solution
    visualize_solution(tsp_instance, best_tour, use_nn)
    
    return best_length


def visualize_solution(tsp_instance: TSPInstance, tour: list, use_nn: bool):
    """
    Visualize the TSP solution.
    
    Args:
        tsp_instance: TSP instance
        tour: Solution tour
        use_nn: Whether NN was used
    """
    plt.figure(figsize=(10, 8))
    
    # Plot cities
    cities = tsp_instance.cities
    plt.scatter(cities[:, 0], cities[:, 1], c='red', s=100, zorder=2, label='Cities')
    
    # Plot tour
    for i in range(len(tour)):
        start = cities[tour[i]]
        end = cities[tour[(i + 1) % len(tour)]]
        plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', alpha=0.6, zorder=1)
    
    # Label cities
    for i, city in enumerate(cities):
        plt.annotate(str(i), (city[0], city[1]), fontsize=8, ha='center', va='center')
    
    method = "NN-based" if use_nn else "Default"
    plt.title(f'TSP Solution ({method} parameters)\nTour length: {tsp_instance.tour_length(tour):.2f}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'tsp_solution_{"nn" if use_nn else "default"}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nSolution visualization saved to {filename}")


def compare_methods(n_cities: int = 20, n_runs: int = 5):
    """
    Compare neural network-based parameter selection vs default parameters.
    
    Args:
        n_cities: Number of cities
        n_runs: Number of runs for each method
    """
    print("=" * 60)
    print(f"COMPARING METHODS ON {n_cities}-CITY TSP ({n_runs} runs each)")
    print("=" * 60)
    
    if not os.path.exists('parameter_model.pth'):
        print("\nError: Model not found. Please train the model first using --train")
        return
    
    # Generate a TSP instance
    tsp_instance = generate_random_tsp(n_cities, seed=42)
    
    # Test with NN-predicted parameters
    predictor = ParameterPredictor(model_path='parameter_model.pth')
    predictor.load()
    
    features = tsp_instance.get_features()
    predicted_params = predictor.predict(features)
    
    print("\nTesting with NN-predicted parameters...")
    nn_results = []
    solver_nn = SimulatedAnnealingSolver(
        initial_temperature=predicted_params[0],
        cooling_rate=predicted_params[1],
        min_temperature=predicted_params[2],
        iterations_per_temp=int(predicted_params[3])
    )
    
    for i in range(n_runs):
        _, length = solver_nn.solve(tsp_instance)
        nn_results.append(length)
        print(f"  Run {i+1}: {length:.2f}")
    
    # Test with default parameters
    print("\nTesting with default parameters...")
    default_results = []
    solver_default = SimulatedAnnealingSolver(
        initial_temperature=100.0,
        cooling_rate=0.995,
        min_temperature=0.01,
        iterations_per_temp=100
    )
    
    for i in range(n_runs):
        _, length = solver_default.solve(tsp_instance)
        default_results.append(length)
        print(f"  Run {i+1}: {length:.2f}")
    
    # Compare results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"\nNN-predicted parameters:")
    print(f"  Mean: {np.mean(nn_results):.2f}")
    print(f"  Best: {np.min(nn_results):.2f}")
    print(f"  Std:  {np.std(nn_results):.2f}")
    
    print(f"\nDefault parameters:")
    print(f"  Mean: {np.mean(default_results):.2f}")
    print(f"  Best: {np.min(default_results):.2f}")
    print(f"  Std:  {np.std(default_results):.2f}")
    
    improvement = (np.mean(default_results) - np.mean(nn_results)) / np.mean(default_results) * 100
    print(f"\nImprovement: {improvement:+.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Data-Driven TSP Solver with Neural Network Parameter Selection'
    )
    
    parser.add_argument('--train', action='store_true',
                        help='Train the neural network model')
    parser.add_argument('--solve', action='store_true',
                        help='Solve a TSP instance')
    parser.add_argument('--compare', action='store_true',
                        help='Compare NN-based vs default parameters')
    parser.add_argument('--cities', type=int, default=20,
                        help='Number of cities (default: 20)')
    parser.add_argument('--instances', type=int, default=200,
                        help='Number of training instances (default: 200)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--no-nn', action='store_true',
                        help='Use default parameters instead of NN')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(n_instances=args.instances, epochs=args.epochs)
    elif args.compare:
        compare_methods(n_cities=args.cities, n_runs=5)
    elif args.solve:
        solve_tsp(n_cities=args.cities, use_nn=not args.no_nn)
    else:
        # Default: train and then solve
        print("No action specified. Training model and solving example TSP...\n")
        train_model(n_instances=args.instances, epochs=args.epochs)
        print("\n")
        solve_tsp(n_cities=args.cities, use_nn=True)


if __name__ == "__main__":
    main()
