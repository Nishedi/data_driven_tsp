"""
Benchmark comparison script for evaluating TSP solver on TSPLIB instances.

Compares:
1. Neural network-based parameter selection (with learning)
2. Default parameters (without learning)

Generates comprehensive statistics and visualizations.
"""
import numpy as np
import os
import argparse
import time
from typing import List, Dict, Tuple
from tsp_solver import TSPInstance, SimulatedAnnealingSolver
from neural_network import ParameterPredictor
from tsplib_parser import parse_tsplib_file, create_sample_tsplib_instances

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required for visualization.")
    print("Please install it using: pip install matplotlib")
    raise ImportError("matplotlib is required but not installed")


def run_benchmark(
    tsp_instance: TSPInstance,
    use_nn: bool,
    predictor: ParameterPredictor = None,
    n_runs: int = 5
) -> Dict:
    """
    Run benchmark on a single TSP instance.
    
    Args:
        tsp_instance: TSP instance to solve
        use_nn: Whether to use neural network for parameter selection
        predictor: Trained parameter predictor (required if use_nn=True)
        n_runs: Number of runs to average
        
    Returns:
        Dictionary with benchmark results
    """
    features = tsp_instance.get_features()
    
    if use_nn:
        if predictor is None:
            raise ValueError("Predictor required when use_nn=True")
        
        # Get NN-predicted parameters
        predicted_params = predictor.predict(features)
        initial_temp = predicted_params[0]
        cooling_rate = predicted_params[1]
        min_temp = predicted_params[2]
        iterations_per_temp = int(predicted_params[3])
    else:
        # Use default parameters
        initial_temp = 100.0
        cooling_rate = 0.995
        min_temp = 0.01
        iterations_per_temp = 100
    
    # Create solver
    solver = SimulatedAnnealingSolver(
        initial_temperature=initial_temp,
        cooling_rate=cooling_rate,
        min_temperature=min_temp,
        iterations_per_temp=iterations_per_temp
    )
    
    # Run multiple times and collect results
    tour_lengths = []
    execution_times = []
    
    for _ in range(n_runs):
        start_time = time.time()
        tour, length = solver.solve(tsp_instance)
        end_time = time.time()
        
        tour_lengths.append(length)
        execution_times.append(end_time - start_time)
    
    # Compile results
    results = {
        'method': 'NN-based' if use_nn else 'Default',
        'mean_length': np.mean(tour_lengths),
        'best_length': np.min(tour_lengths),
        'worst_length': np.max(tour_lengths),
        'std_length': np.std(tour_lengths),
        'mean_time': np.mean(execution_times),
        'parameters': {
            'initial_temp': initial_temp,
            'cooling_rate': cooling_rate,
            'min_temp': min_temp,
            'iterations_per_temp': iterations_per_temp
        },
        'all_lengths': tour_lengths
    }
    
    return results


def compare_on_instance(
    instance_path: str,
    predictor: ParameterPredictor,
    n_runs: int = 5
) -> Tuple[Dict, Dict]:
    """
    Compare both methods on a single TSPLIB instance.
    
    Args:
        instance_path: Path to TSPLIB file
        predictor: Trained parameter predictor
        n_runs: Number of runs per method
        
    Returns:
        Tuple of (nn_results, default_results)
    """
    print(f"\nBenchmarking: {os.path.basename(instance_path)}")
    print("-" * 60)
    
    # Load instance
    tsp_instance = parse_tsplib_file(instance_path)
    instance_name = instance_path if not hasattr(tsp_instance, 'name') else tsp_instance.name
    
    print(f"Instance: {instance_name}")
    print(f"Cities: {tsp_instance.n_cities}")
    
    # Run with NN
    print(f"\nRunning with NN-based parameters ({n_runs} runs)...")
    nn_results = run_benchmark(tsp_instance, use_nn=True, predictor=predictor, n_runs=n_runs)
    
    # Run with default
    print(f"Running with default parameters ({n_runs} runs)...")
    default_results = run_benchmark(tsp_instance, use_nn=False, n_runs=n_runs)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nNN-based parameters:")
    print(f"  Initial temp: {nn_results['parameters']['initial_temp']:.2f}")
    print(f"  Cooling rate: {nn_results['parameters']['cooling_rate']:.4f}")
    print(f"  Min temp: {nn_results['parameters']['min_temp']:.4f}")
    print(f"  Iterations: {nn_results['parameters']['iterations_per_temp']}")
    print(f"  Mean length: {nn_results['mean_length']:.2f}")
    print(f"  Best length: {nn_results['best_length']:.2f}")
    print(f"  Std dev: {nn_results['std_length']:.2f}")
    print(f"  Mean time: {nn_results['mean_time']:.3f}s")
    
    print("\nDefault parameters:")
    print(f"  Initial temp: {default_results['parameters']['initial_temp']:.2f}")
    print(f"  Cooling rate: {default_results['parameters']['cooling_rate']:.4f}")
    print(f"  Min temp: {default_results['parameters']['min_temp']:.4f}")
    print(f"  Iterations: {default_results['parameters']['iterations_per_temp']}")
    print(f"  Mean length: {default_results['mean_length']:.2f}")
    print(f"  Best length: {default_results['best_length']:.2f}")
    print(f"  Std dev: {default_results['std_length']:.2f}")
    print(f"  Mean time: {default_results['mean_time']:.3f}s")
    
    # Calculate improvement
    improvement = (default_results['mean_length'] - nn_results['mean_length']) / default_results['mean_length'] * 100
    print(f"\n{'Improvement' if improvement > 0 else 'Degradation'}: {abs(improvement):.2f}%")
    
    return nn_results, default_results


def run_full_benchmark_suite(
    instance_dir: str = 'tsplib_instances',
    n_runs: int = 10
):
    """
    Run comprehensive benchmark on all TSPLIB instances in directory.
    
    Args:
        instance_dir: Directory containing .tsp files
        n_runs: Number of runs per instance per method
    """
    print("=" * 70)
    print("TSPLIB BENCHMARK SUITE")
    print("=" * 70)
    
    # Check if model exists
    if not os.path.exists('parameter_model.pth'):
        print("\nError: Model not found. Please train the model first:")
        print("  python main.py --train --instances 200 --epochs 100")
        return
    
    # Load predictor
    print("\nLoading trained model...")
    predictor = ParameterPredictor(model_path='parameter_model.pth')
    try:
        predictor.load()
    except FileNotFoundError:
        print("\nError: Model file not found at 'parameter_model.pth'")
        print("Please train the model first using:")
        print("  python main.py --train --instances 200 --epochs 100")
        return
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("The model file may be corrupted or incompatible.")
        print("Please retrain the model using:")
        print("  python main.py --train --instances 200 --epochs 100")
        return
    
    # Find all .tsp files
    if not os.path.exists(instance_dir):
        print(f"\nCreating sample TSPLIB instances in {instance_dir}/...")
        create_sample_tsplib_instances()
    
    tsp_files = [f for f in os.listdir(instance_dir) if f.endswith('.tsp')]
    
    if not tsp_files:
        print(f"\nNo .tsp files found in {instance_dir}/")
        return
    
    print(f"\nFound {len(tsp_files)} TSPLIB instances")
    print(f"Running {n_runs} trials per instance per method")
    
    # Run benchmarks
    all_results = []
    
    for tsp_file in sorted(tsp_files):
        instance_path = os.path.join(instance_dir, tsp_file)
        try:
            nn_results, default_results = compare_on_instance(instance_path, predictor, n_runs)
            
            all_results.append({
                'instance': tsp_file,
                'nn': nn_results,
                'default': default_results
            })
        except Exception as e:
            print(f"Error processing {tsp_file}: {e}")
            continue
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    summary_data = []
    for result in all_results:
        instance = result['instance']
        nn = result['nn']
        default = result['default']
        
        improvement = (default['mean_length'] - nn['mean_length']) / default['mean_length'] * 100
        
        summary_data.append({
            'instance': instance,
            'nn_mean': nn['mean_length'],
            'default_mean': default['mean_length'],
            'improvement': improvement
        })
    
    # Print summary table
    print(f"\n{'Instance':<20} {'NN Mean':<12} {'Default Mean':<12} {'Improvement':<12}")
    print("-" * 70)
    for data in summary_data:
        print(f"{data['instance']:<20} {data['nn_mean']:<12.2f} {data['default_mean']:<12.2f} {data['improvement']:>+11.2f}%")
    
    # Overall statistics
    improvements = [d['improvement'] for d in summary_data]
    print("\n" + "=" * 70)
    print(f"Overall improvement (mean): {np.mean(improvements):+.2f}%")
    print(f"Best improvement: {np.max(improvements):+.2f}%")
    print(f"Worst improvement: {np.min(improvements):+.2f}%")
    print(f"Std dev: {np.std(improvements):.2f}%")
    print(f"NN wins: {sum(1 for i in improvements if i > 0)}/{len(improvements)}")
    
    # Generate visualization
    visualize_benchmark_results(all_results, summary_data)
    
    # Save detailed results
    save_results_to_file(all_results, summary_data)


def visualize_benchmark_results(all_results: List[Dict], summary_data: List[Dict]):
    """Create visualization of benchmark results."""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TSPLIB Benchmark Comparison: NN-based vs Default Parameters', fontsize=14, fontweight='bold')
    
    # 1. Bar chart comparing mean lengths
    ax1 = axes[0, 0]
    instances = [d['instance'].replace('.tsp', '') for d in summary_data]
    nn_means = [d['nn_mean'] for d in summary_data]
    default_means = [d['default_mean'] for d in summary_data]
    
    x = np.arange(len(instances))
    width = 0.35
    
    ax1.bar(x - width/2, nn_means, width, label='NN-based', color='#2ecc71')
    ax1.bar(x + width/2, default_means, width, label='Default', color='#e74c3c')
    
    ax1.set_xlabel('Instance')
    ax1.set_ylabel('Mean Tour Length')
    ax1.set_title('Mean Tour Length Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(instances, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement percentage bar chart
    ax2 = axes[0, 1]
    improvements = [d['improvement'] for d in summary_data]
    colors = ['#2ecc71' if i > 0 else '#e74c3c' for i in improvements]
    
    ax2.bar(instances, improvements, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.set_xlabel('Instance')
    ax2.set_ylabel('Improvement (%)')
    ax2.set_title('Performance Improvement (NN vs Default)')
    ax2.set_xticks(range(len(instances)))
    ax2.set_xticklabels(instances, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot of distributions
    ax3 = axes[1, 0]
    nn_all = []
    default_all = []
    labels = []
    
    for result in all_results:
        nn_all.append(result['nn']['all_lengths'])
        default_all.append(result['default']['all_lengths'])
        labels.append(result['instance'].replace('.tsp', ''))
    
    positions_nn = np.arange(len(labels)) * 2
    positions_default = positions_nn + 0.6
    
    bp1 = ax3.boxplot(nn_all, positions=positions_nn, widths=0.5, 
                      patch_artist=True, showfliers=False)
    bp2 = ax3.boxplot(default_all, positions=positions_default, widths=0.5,
                      patch_artist=True, showfliers=False)
    
    for patch in bp1['boxes']:
        patch.set_facecolor('#2ecc71')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)
    
    ax3.set_xlabel('Instance')
    ax3.set_ylabel('Tour Length')
    ax3.set_title('Distribution of Results (Multiple Runs)')
    ax3.set_xticks(positions_nn + 0.3)
    ax3.set_xticklabels(labels, rotation=45, ha='right')
    ax3.legend([bp1["boxes"][0], bp2["boxes"][0]], ['NN-based', 'Default'])
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    overall_improvement = np.mean(improvements)
    wins = sum(1 for i in improvements if i > 0)
    
    summary_text = f"""
    SUMMARY STATISTICS
    {'=' * 40}
    
    Total Instances: {len(all_results)}
    
    NN-based wins: {wins}/{len(all_results)} ({wins/len(all_results)*100:.1f}%)
    
    Mean Improvement: {overall_improvement:+.2f}%
    Best Improvement: {np.max(improvements):+.2f}%
    Worst Improvement: {np.min(improvements):+.2f}%
    Std Dev: {np.std(improvements):.2f}%
    
    {'=' * 40}
    
    The neural network learns to adapt
    SA parameters based on problem features,
    typically achieving better or more
    consistent results compared to fixed
    default parameters.
    """
    
    ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to benchmark_comparison.png")


def save_results_to_file(all_results: List[Dict], summary_data: List[Dict]):
    """Save detailed results to a text file."""
    
    with open('benchmark_results.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TSPLIB BENCHMARK COMPARISON RESULTS\n")
        f.write("=" * 70 + "\n\n")
        
        for result in all_results:
            f.write(f"\nInstance: {result['instance']}\n")
            f.write("-" * 70 + "\n")
            
            nn = result['nn']
            default = result['default']
            
            f.write("\nNN-based parameters:\n")
            f.write(f"  Mean: {nn['mean_length']:.2f}, Best: {nn['best_length']:.2f}, ")
            f.write(f"Worst: {nn['worst_length']:.2f}, Std: {nn['std_length']:.2f}\n")
            f.write(f"  Parameters: temp={nn['parameters']['initial_temp']:.2f}, ")
            f.write(f"cooling={nn['parameters']['cooling_rate']:.4f}, ")
            f.write(f"min_temp={nn['parameters']['min_temp']:.4f}, ")
            f.write(f"iters={nn['parameters']['iterations_per_temp']}\n")
            
            f.write("\nDefault parameters:\n")
            f.write(f"  Mean: {default['mean_length']:.2f}, Best: {default['best_length']:.2f}, ")
            f.write(f"Worst: {default['worst_length']:.2f}, Std: {default['std_length']:.2f}\n")
            
            improvement = (default['mean_length'] - nn['mean_length']) / default['mean_length'] * 100
            f.write(f"\nImprovement: {improvement:+.2f}%\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        improvements = [d['improvement'] for d in summary_data]
        f.write(f"Overall mean improvement: {np.mean(improvements):+.2f}%\n")
        f.write(f"Best improvement: {np.max(improvements):+.2f}%\n")
        f.write(f"Worst improvement: {np.min(improvements):+.2f}%\n")
        f.write(f"NN wins: {sum(1 for i in improvements if i > 0)}/{len(improvements)}\n")
    
    print("Detailed results saved to benchmark_results.txt")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark TSP solver on TSPLIB instances'
    )
    
    parser.add_argument('--instance-dir', type=str, default='tsplib_instances',
                        help='Directory containing TSPLIB .tsp files')
    parser.add_argument('--runs', type=int, default=10,
                        help='Number of runs per instance (default: 10)')
    parser.add_argument('--single', type=str, default=None,
                        help='Run on a single instance file')
    
    args = parser.parse_args()
    
    if args.single:
        # Run on single instance
        if not os.path.exists('parameter_model.pth'):
            print("Error: Model not found. Please train first.")
            return
        
        predictor = ParameterPredictor(model_path='parameter_model.pth')
        predictor.load()
        
        compare_on_instance(args.single, predictor, args.runs)
    else:
        # Run full benchmark suite
        run_full_benchmark_suite(args.instance_dir, args.runs)


if __name__ == "__main__":
    main()
