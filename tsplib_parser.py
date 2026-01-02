"""
TSPLIB parser for reading standard TSP benchmark instances.

Supports reading .tsp files in TSPLIB format with various specifications:
- NODE_COORD_SECTION: Euclidean 2D coordinates
- EDGE_WEIGHT_SECTION: Explicit distance matrix
- Different edge weight types (EUC_2D, EXPLICIT, etc.)
"""
import numpy as np
import math
import os
import urllib.request
from typing import Optional, Tuple
from tsp_solver import TSPInstance


def parse_tsplib_file(filepath: str) -> TSPInstance:
    """
    Parse a TSPLIB format TSP file and create a TSPInstance.
    
    Args:
        filepath: Path to the .tsp file
        
    Returns:
        TSPInstance object
        
    Raises:
        ValueError: If file format is not supported
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Parse header information
    name = None
    dimension = None
    edge_weight_type = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith('NAME'):
            name = line.split(':')[1].strip()
        elif line.startswith('DIMENSION'):
            dimension = int(line.split(':')[1].strip())
        elif line.startswith('EDGE_WEIGHT_TYPE'):
            edge_weight_type = line.split(':')[1].strip()
        elif line.startswith('NODE_COORD_SECTION'):
            # Read coordinates
            return _parse_coord_section(lines[i+1:], dimension, name)
        elif line.startswith('EDGE_WEIGHT_SECTION'):
            # Read explicit distance matrix
            return _parse_edge_weight_section(lines[i+1:], dimension, name)
        
        i += 1
    
    raise ValueError(f"Could not parse TSPLIB file: {filepath}")


def _parse_coord_section(lines: list, dimension: int, name: str) -> TSPInstance:
    """Parse NODE_COORD_SECTION to extract city coordinates."""
    cities = []
    
    for line in lines:
        if line.startswith('EOF') or line == '':
            break
        
        parts = line.split()
        if len(parts) >= 3:
            # Format: node_id x y
            x = float(parts[1])
            y = float(parts[2])
            cities.append([x, y])
    
    if len(cities) != dimension:
        raise ValueError(f"Expected {dimension} cities, found {len(cities)}")
    
    cities_array = np.array(cities, dtype=np.float64)
    instance = TSPInstance(cities_array)
    instance.name = name
    return instance


def _parse_edge_weight_section(lines: list, dimension: int, name: str) -> TSPInstance:
    """
    Parse EDGE_WEIGHT_SECTION for explicit distance matrix.
    Note: This creates approximate coordinates using MDS for visualization.
    The actual distances from the matrix are used for solving.
    """
    distances = []
    
    for line in lines:
        if line.startswith('EOF') or line == '':
            break
        
        parts = line.split()
        for part in parts:
            try:
                distances.append(float(part))
            except ValueError:
                pass
    
    # Convert to matrix
    if len(distances) == dimension * dimension:
        # Full matrix
        distance_matrix = np.array(distances).reshape(dimension, dimension)
    elif len(distances) == dimension * (dimension - 1) // 2:
        # Upper triangular
        distance_matrix = np.zeros((dimension, dimension))
        idx = 0
        for i in range(dimension):
            for j in range(i + 1, dimension):
                distance_matrix[i, j] = distances[idx]
                distance_matrix[j, i] = distances[idx]
                idx += 1
    else:
        raise ValueError(f"Unexpected number of distances: {len(distances)}")
    
    # Use simple MDS-like approach to generate approximate coordinates
    # This gives better visualization than random coordinates
    try:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        cities = mds.fit_transform(distance_matrix)
    except ImportError:
        # Fallback: Classical MDS (multidimensional scaling) implementation
        # This creates approximate 2D coordinates from the distance matrix
        # Algorithm steps:
        # 1. Square the distance matrix: D^2
        # 2. Create centering matrix: H = I - (1/n)J where J is all ones
        # 3. Compute B = -0.5 * H * D^2 * H (double-centered matrix)
        # 4. Eigen decomposition of B
        # 5. Take coordinates as eigenvectors * sqrt(eigenvalues) for top 2 eigenvalues
        n = distance_matrix.shape[0]
        D_squared = distance_matrix ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D_squared @ H
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Take the two largest eigenvalues (ensure they are positive)
        idx = eigenvalues.argsort()[-2:][::-1]
        selected_eigenvalues = np.maximum(eigenvalues[idx], 0)  # Ensure non-negative
        cities = eigenvectors[:, idx] * np.sqrt(selected_eigenvalues)
    
    instance = TSPInstance(cities)
    instance.distance_matrix = distance_matrix  # Override with explicit distances
    instance.name = name
    return instance


def download_tsplib_instance(url: str, filepath: str):
    """
    Download a TSPLIB instance from URL.
    
    Args:
        url: URL to the .tsp file
        filepath: Local path to save the file
    """
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Saved to {filepath}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise


def create_sample_tsplib_instances():
    """
    Create some sample TSPLIB-format files for testing.
    These are small instances suitable for quick testing.
    """
    # Create a directory for benchmark instances
    os.makedirs('tsplib_instances', exist_ok=True)
    
    # Berlin52 (simplified version with fewer cities for demo)
    berlin_sample = """NAME : berlin52_sample
COMMENT : Sample from Berlin52 (first 20 cities)
TYPE : TSP
DIMENSION : 20
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 565.0 575.0
2 25.0 185.0
3 345.0 750.0
4 945.0 685.0
5 845.0 655.0
6 880.0 660.0
7 25.0 230.0
8 525.0 1000.0
9 580.0 1175.0
10 650.0 1130.0
11 1605.0 620.0
12 1220.0 580.0
13 1465.0 200.0
14 1530.0 5.0
15 845.0 680.0
16 725.0 370.0
17 145.0 665.0
18 415.0 635.0
19 510.0 875.0
20 560.0 365.0
EOF
"""
    
    with open('tsplib_instances/berlin20.tsp', 'w') as f:
        f.write(berlin_sample)
    
    # Simple 10-city instance
    simple_10 = """NAME : simple10
COMMENT : Simple 10-city test instance
TYPE : TSP
DIMENSION : 10
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 10.0 10.0
2 20.0 15.0
3 30.0 20.0
4 40.0 25.0
5 50.0 30.0
6 60.0 25.0
7 70.0 20.0
8 80.0 15.0
9 90.0 10.0
10 50.0 50.0
EOF
"""
    
    with open('tsplib_instances/simple10.tsp', 'w') as f:
        f.write(simple_10)
    
    # Medium 30-city instance
    np.random.seed(42)
    coords_30 = np.random.rand(30, 2) * 1000
    
    with open('tsplib_instances/random30.tsp', 'w') as f:
        f.write("NAME : random30\n")
        f.write("COMMENT : Random 30-city instance\n")
        f.write("TYPE : TSP\n")
        f.write("DIMENSION : 30\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y) in enumerate(coords_30, 1):
            f.write(f"{i} {x:.1f} {y:.1f}\n")
        f.write("EOF\n")
    
    print("Created sample TSPLIB instances:")
    print("  - tsplib_instances/berlin20.tsp (20 cities)")
    print("  - tsplib_instances/simple10.tsp (10 cities)")
    print("  - tsplib_instances/random30.tsp (30 cities)")


if __name__ == "__main__":
    # Create sample instances and test parser
    create_sample_tsplib_instances()
    
    # Test parser
    print("\nTesting parser...")
    instance = parse_tsplib_file('tsplib_instances/berlin20.tsp')
    print(f"Loaded: {instance.name if hasattr(instance, 'name') else 'Unknown'}")
    print(f"Cities: {instance.n_cities}")
    print(f"Features: {instance.get_features()}")
