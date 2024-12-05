import numpy as np
import random

# Utility map: Example grid of utility values for wildfire coverage
# Higher values indicate areas of greater importance.
utility_map = np.array([
    [1, 2, 3, 4],
    [0, 1, 5, 3],
    [1, 4, 2, 6],
    [2, 5, 3, 1]
])

# Parameters
num_drones = 3
grid_size = utility_map.shape  # (rows, cols)
max_iterations = 100
neighborhoods = [1, 2, 3]  # Neighborhood structures

# Utility function
def compute_utility(solution, utility_map):
    """
    Computes the total utility of a given solution.
    Solution is a list of (row, col) positions representing drone locations.
    """
    total_utility = 0
    for drone_pos in solution:
        row, col = drone_pos
        total_utility += utility_map[row, col]
    return total_utility

import numpy as np

def initialize_utility_map(rate_of_spread, u_min=0.1):
    """
    Initialize the utility map based on the rate of spread (RoS) for each cell.
    Parameters:
        rate_of_spread (2D array): Matrix representing the rate of spread of the wildfire.
        u_min (float): Minimum utility constant.
    Returns:
        utility_map (2D array): Initialized utility map.
    """
    ros_min = np.min(rate_of_spread)
    ros_max = np.max(rate_of_spread)

    # Compute initial utility map
    utility_map = u_min + ((rate_of_spread - ros_min) / (ros_max - ros_min)) * (1 - u_min)
    return utility_map


def compute_indirect_utility(c_obs, c_ind, rate_of_spread, d_max, alpha=0.5):
    """
    Compute the indirect utility of observing a cell (c_obs) for another cell (c_ind).
    Parameters:
        c_obs (tuple): Coordinates of the observed cell (row, col).
        c_ind (tuple): Coordinates of the indirectly affected cell (row, col).
        rate_of_spread (2D array): Matrix representing the rate of spread.
        d_max (float): Maximum distance for indirect utility influence.
        alpha (float): Scaling constant for indirect information gain.
    Returns:
        u_ind (float): Computed indirect utility.
    """
    # Calculate distance between observed and indirect cells
    dist = np.sqrt((c_obs[0] - c_ind[0]) ** 2 + (c_obs[1] - c_ind[1]) ** 2)
    if dist > d_max:
        return 0  # No influence if beyond max distance

    # Base utility of the indirectly affected cell
    u_base = rate_of_spread[c_ind[0], c_ind[1]]
    u_ind = u_base * alpha * (1 - dist / d_max)

    return u_ind


def update_utility_map(c_obs, utility_map, rate_of_spread, d_max, alpha=0.5):
    """
    Update the utility map after observing a cell.
    Parameters:
        c_obs (tuple): Coordinates of the observed cell (row, col).
        utility_map (2D array): Current utility map to be updated.
        rate_of_spread (2D array): Matrix representing the rate of spread.
        d_max (float): Maximum distance for indirect utility influence.
        alpha (float): Scaling constant for indirect information gain.
    Returns:
        delta_u (float): Total utility gained from the observation.
    """
    rows, cols = utility_map.shape
    delta_u = utility_map[c_obs[0], c_obs[1]]
    utility_map[c_obs[0], c_obs[1]] = 0  # Reset direct utility to 0 after observation

    # Update indirect utility for nearby cells
    for r in range(max(0, c_obs[0] - int(d_max)), min(rows, c_obs[0] + int(d_max) + 1)):
        for c in range(max(0, c_obs[1] - int(d_max)), min(cols, c_obs[1] + int(d_max) + 1)):
            c_ind = (r, c)
            if c_ind == c_obs:
                continue

            # Compute indirect utility and update the map
            u_ind = compute_indirect_utility(c_obs, c_ind, rate_of_spread, d_max, alpha)
            u_ind = min(utility_map[r, c], u_ind)  # Limit indirect utility by current value
            utility_map[r, c] -= u_ind
            delta_u += u_ind

    return delta_u


def nearby_cells(c_obs, d_max):
    """Returns list of cell indices within d_max distance of c_obs."""
    row, col = c_obs
    nearby = []
    for i in range(max(0, row - d_max), min(utility_map.shape[0], row + d_max + 1)):
        for j in range(max(0, col - d_max), min(utility_map.shape[1], col + d_max + 1)):
            if (i, j) != c_obs:  # Exclude the observation cell itself
                nearby.append((i, j))
    return nearby

# Generate initial random solution
def generate_initial_solution(num_drones, grid_size):
    """
    Generates a random initial solution.
    Each drone is placed at a random location within the grid.
    """
    return [tuple(random.randint(0, grid_size[i] - 1) for i in range(2)) for _ in range(num_drones)]

# Neighborhood definition
def generate_neighbors(solution, neighborhood_size, grid_size):
    """
    Generates neighbors for a given solution by moving drones within a specified neighborhood size.
    """
    neighbors = []
    for i, (row, col) in enumerate(solution):
        for dr in range(-neighborhood_size, neighborhood_size + 1):
            for dc in range(-neighborhood_size, neighborhood_size + 1):
                if dr == 0 and dc == 0:
                    continue
                new_row = max(0, min(grid_size[0] - 1, row + dr))
                new_col = max(0, min(grid_size[1] - 1, col + dc))
                new_solution = solution.copy()
                new_solution[i] = (new_row, new_col)
                neighbors.append(new_solution)
    return neighbors

# VNS Algorithm
def vns(utility_map, num_drones, neighborhoods, max_iterations):
    """
    Implements the Variable Neighborhood Search (VNS) algorithm.
    """
    # Step 1: Generate initial solution
    current_solution = generate_initial_solution(num_drones, grid_size)
    best_solution = current_solution
    best_utility = compute_utility(best_solution, utility_map)
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration+1}: Best Utility = {best_utility}")
        k = 0
        
        # Step 2: Systematically explore neighborhoods
        while k < len(neighborhoods):
            # Generate neighbors
            neighbors = generate_neighbors(current_solution, neighborhoods[k], grid_size)
            
            # Evaluate neighbors
            improved = False
            for neighbor in neighbors:
                neighbor_utility = compute_utility(neighbor, utility_map)
                
                if neighbor_utility > best_utility:
                    # Update the best solution
                    best_solution = neighbor
                    best_utility = neighbor_utility
                    current_solution = neighbor
                    improved = True
                    k = 0  # Restart neighborhood exploration
                    break
            
            if not improved:
                # Move to the next neighborhood
                k += 1
        
        # Early stopping if no improvement
        if k == len(neighborhoods):
            break
    
    return best_solution, best_utility

# Run VNS
best_solution, best_utility = vns(utility_map, num_drones, neighborhoods, max_iterations)

# Output the result
print("Best Solution (Drone Positions):", best_solution)
print("Best Utility:", best_utility)
