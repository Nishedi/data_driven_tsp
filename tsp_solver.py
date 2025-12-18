"""
TSP Solver using Simulated Annealing metaheuristic with configurable parameters.
"""
import numpy as np
import random
from typing import List, Tuple, Optional


class TSPInstance:
    """Represents a TSP problem instance."""
    
    def __init__(self, cities: np.ndarray):
        """
        Initialize TSP instance.
        
        Args:
            cities: Array of shape (n_cities, 2) with city coordinates
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise distances between all cities."""
        # Use broadcasting for vectorized distance computation
        diff = self.cities[:, np.newaxis, :] - self.cities[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        return distances
    
    def tour_length(self, tour: List[int]) -> float:
        """Calculate total length of a tour."""
        length = 0.0
        for i in range(len(tour)):
            length += self.distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
        return length
    
    def get_features(self) -> np.ndarray:
        """
        Extract features from the TSP instance for neural network input.
        
        Returns:
            Feature vector describing the problem characteristics
        """
        # Features that might influence optimal SA parameters:
        # 1. Number of cities
        # 2. Average distance
        # 3. Standard deviation of distances
        # 4. Convex hull ratio (spread of cities)
        # 5. Distance range (max - min)
        
        distances = self.distance_matrix[np.triu_indices(self.n_cities, k=1)]
        
        features = [
            self.n_cities,
            np.mean(distances),
            np.std(distances),
            np.max(distances) - np.min(distances),
            np.max(distances) / (np.min(distances) + 1e-8),
            # Spatial spread
            np.std(self.cities[:, 0]),
            np.std(self.cities[:, 1]),
        ]
        
        return np.array(features, dtype=np.float32)


class SimulatedAnnealingSolver:
    """Simulated Annealing solver for TSP with configurable parameters."""
    
    def __init__(self, 
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.995,
                 min_temperature: float = 0.01,
                 iterations_per_temp: int = 100):
        """
        Initialize SA solver with parameters.
        
        Args:
            initial_temperature: Starting temperature
            cooling_rate: Temperature reduction factor (0 < rate < 1)
            min_temperature: Stopping temperature
            iterations_per_temp: Number of iterations at each temperature
        """
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp
    
    def solve(self, tsp_instance: TSPInstance, 
              initial_solution: Optional[List[int]] = None) -> Tuple[List[int], float]:
        """
        Solve TSP using Simulated Annealing.
        
        Args:
            tsp_instance: TSP problem instance
            initial_solution: Optional starting tour
            
        Returns:
            Tuple of (best_tour, best_length)
        """
        # Initialize solution
        if initial_solution is None:
            current_tour = list(range(tsp_instance.n_cities))
            random.shuffle(current_tour)
        else:
            current_tour = initial_solution.copy()
        
        current_length = tsp_instance.tour_length(current_tour)
        best_tour = current_tour.copy()
        best_length = current_length
        
        temperature = self.initial_temperature
        
        while temperature > self.min_temperature:
            for _ in range(self.iterations_per_temp):
                # Generate neighbor solution using 2-opt
                new_tour = self._two_opt_swap(current_tour)
                new_length = tsp_instance.tour_length(new_tour)
                
                # Accept or reject
                delta = new_length - current_length
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_tour = new_tour
                    current_length = new_length
                    
                    # Update best solution
                    if current_length < best_length:
                        best_tour = current_tour.copy()
                        best_length = current_length
            
            # Cool down
            temperature *= self.cooling_rate
        
        return best_tour, best_length
    
    def _two_opt_swap(self, tour: List[int]) -> List[int]:
        """Perform a 2-opt swap on the tour."""
        new_tour = tour.copy()
        n = len(tour)
        i, j = sorted(random.sample(range(n), 2))
        new_tour[i:j+1] = reversed(new_tour[i:j+1])
        return new_tour


def generate_random_tsp(n_cities: int, seed: Optional[int] = None) -> TSPInstance:
    """
    Generate a random TSP instance.
    
    Args:
        n_cities: Number of cities
        seed: Random seed for reproducibility
        
    Returns:
        TSPInstance
    """
    if seed is not None:
        np.random.seed(seed)
    
    cities = np.random.rand(n_cities, 2) * 100
    return TSPInstance(cities)
