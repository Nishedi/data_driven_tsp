#!/usr/bin/env python3
"""
Demonstration script showing the complete data-driven TSP solver workflow.
"""
import os
import sys

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def main():
    print_section("DATA-DRIVEN TSP SOLVER DEMONSTRATION")
    
    print("This demonstration shows a complete TSP solver that uses a neural network")
    print("to select optimal Simulated Annealing parameters based on problem features.\n")
    
    # Check if model exists
    if not os.path.exists('parameter_model.pth'):
        print_section("STEP 1: Training the Neural Network")
        print("Training the parameter prediction model with 100 instances...\n")
        os.system('python main.py --train --instances 100 --epochs 50')
    else:
        print("Model already trained. Skipping training step.\n")
    
    print_section("STEP 2: Solving TSP with Neural Network Parameters")
    print("Solving a 25-city TSP using NN-predicted parameters...\n")
    os.system('python main.py --solve --cities 25')
    
    print_section("STEP 3: Solving TSP with Default Parameters")
    print("Solving the same size problem with default parameters for comparison...\n")
    os.system('python main.py --solve --cities 25 --no-nn')
    
    print_section("STEP 4: Statistical Comparison")
    print("Running multiple trials to compare both approaches...\n")
    os.system('python main.py --compare --cities 20')
    
    print_section("DEMONSTRATION COMPLETE")
    print("Check the generated PNG files to see the solution visualizations:")
    print("  - tsp_solution_nn.png      (using NN-predicted parameters)")
    print("  - tsp_solution_default.png (using default parameters)")
    print("\nThe neural network adapts the SA parameters based on:")
    print("  • Problem size (number of cities)")
    print("  • Distance distribution (mean, std dev, range)")
    print("  • Spatial characteristics (spread of cities)")
    print("\nThis allows the solver to perform better across diverse problem instances!")

if __name__ == "__main__":
    main()
