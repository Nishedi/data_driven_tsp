"""
Test script to verify TSPLIB benchmark implementation.

This script verifies:
1. TSPLIB parser works correctly
2. Benchmark comparison runs successfully  
3. Both NN-based and default approaches work
"""
import os
import sys


def test_tsplib_parser():
    """Test TSPLIB parser functionality."""
    print("Testing TSPLIB parser...")
    
    from tsplib_parser import parse_tsplib_file
    
    # Test parsing a sample file
    instance = parse_tsplib_file('tsplib_instances/berlin20.tsp')
    
    assert instance.n_cities == 20, f"Expected 20 cities, got {instance.n_cities}"
    assert hasattr(instance, 'name'), "Instance should have a name"
    
    features = instance.get_features()
    assert len(features) == 7, f"Expected 7 features, got {len(features)}"
    
    print("✓ TSPLIB parser test passed")
    return True


def test_benchmark_single():
    """Test single instance benchmark."""
    print("\nTesting single instance benchmark...")
    
    from benchmark_comparison import compare_on_instance
    from neural_network import ParameterPredictor
    
    # Load predictor
    if not os.path.exists('parameter_model.pth'):
        print("⚠ Model not found. Skipping benchmark test.")
        print("  Run: python main.py --train")
        return False
    
    predictor = ParameterPredictor(model_path='parameter_model.pth')
    predictor.load()
    
    # Run comparison on single instance
    nn_results, default_results = compare_on_instance(
        'tsplib_instances/simple10.tsp',
        predictor,
        n_runs=3
    )
    
    # Verify results structure
    assert 'mean_length' in nn_results, "NN results missing mean_length"
    assert 'mean_length' in default_results, "Default results missing mean_length"
    assert nn_results['mean_length'] > 0, "Invalid mean_length"
    assert default_results['mean_length'] > 0, "Invalid mean_length"
    
    print("✓ Single instance benchmark test passed")
    return True


def test_tsp_solver():
    """Test TSP solver with both approaches."""
    print("\nTesting TSP solver...")
    
    from tsp_solver import generate_random_tsp, SimulatedAnnealingSolver
    from neural_network import ParameterPredictor
    
    # Generate a small TSP instance
    tsp_instance = generate_random_tsp(15, seed=42)
    
    # Test with default parameters
    solver_default = SimulatedAnnealingSolver(
        initial_temperature=100.0,
        cooling_rate=0.995,
        min_temperature=0.01,
        iterations_per_temp=100
    )
    
    tour, length = solver_default.solve(tsp_instance)
    assert len(tour) == 15, f"Expected tour of length 15, got {len(tour)}"
    assert length > 0, "Invalid tour length"
    
    # Test with NN-predicted parameters (if model exists)
    if os.path.exists('parameter_model.pth'):
        predictor = ParameterPredictor(model_path='parameter_model.pth')
        predictor.load()
        
        features = tsp_instance.get_features()
        predicted_params = predictor.predict(features)
        
        solver_nn = SimulatedAnnealingSolver(
            initial_temperature=predicted_params[0],
            cooling_rate=predicted_params[1],
            min_temperature=predicted_params[2],
            iterations_per_temp=int(predicted_params[3])
        )
        
        tour_nn, length_nn = solver_nn.solve(tsp_instance)
        assert len(tour_nn) == 15, f"Expected tour of length 15, got {len(tour_nn)}"
        assert length_nn > 0, "Invalid tour length"
        
        print(f"  Default params: tour length = {length:.2f}")
        print(f"  NN params: tour length = {length_nn:.2f}")
    
    print("✓ TSP solver test passed")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("TSPLIB BENCHMARK IMPLEMENTATION TESTS")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test TSPLIB parser
    try:
        if test_tsplib_parser():
            tests_passed += 1
    except Exception as e:
        print(f"✗ TSPLIB parser test failed: {e}")
        tests_failed += 1
    
    # Test TSP solver
    try:
        if test_tsp_solver():
            tests_passed += 1
    except Exception as e:
        print(f"✗ TSP solver test failed: {e}")
        tests_failed += 1
    
    # Test benchmark
    try:
        if test_benchmark_single():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Benchmark test failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    
    if tests_failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {tests_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
