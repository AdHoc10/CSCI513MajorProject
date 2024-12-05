from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config
import time
import numpy as np
import logging
import pygame
import sys

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add formatter to ch
ch.setFormatter(formatter)
# Add ch to logger
logger.addHandler(ch)

def calculate_gradients(ros_map):
    """Calculate gradients of the rate of spread map once per update"""
    grad_x, grad_y = np.gradient(ros_map)
    return grad_x, grad_y

def calculate_direction(grad_x, grad_y, position):
    """Calculate the direction of fire spread at a given position using precomputed gradients"""
    x, y = position
    direction = np.arctan2(grad_y[x, y], grad_x[x, y])
    return direction

def diversity_reward(directions):
    """Calculate a reward based on the diversity of directions"""
    unique_directions = np.unique(directions)
    return len(unique_directions) / len(directions)

def initialize_utility_map(fire_map, ros_map, u_min=0.1):
    """Initialize utility map based on rate of spread and fire state"""
    # Transpose maps to match coordinate system
    fire_map = fire_map.T
    ros_map = ros_map.T
    
    logger.info(f"Fire map unique values: {np.unique(fire_map)}")
    
    # Get rate of spread bounds for normalization
    ros_min = np.min(ros_map)
    ros_max = np.max(ros_map)
    
    # Initialize base utility using rate of spread
    utility_map = u_min + ((ros_map - ros_min) / (ros_max - ros_min)) * (1 - u_min)
    
    logger.info(f"Utility map range: {np.min(utility_map)}, {np.max(utility_map)}")
    return utility_map

def compute_indirect_utility(c_obs, c_ind, ros_map, d_max=5, alpha=0.5):
    """Compute indirect utility between observation point and affected cell"""
    dist = np.sqrt((c_obs[0] - c_ind[0])**2 + (c_obs[1] - c_ind[1])**2)
    if dist > d_max:
        return 0
    
    u_base = ros_map[c_ind[0], c_ind[1]]
    u_ind = u_base * alpha * (1 - dist/d_max)
    return u_ind

def update_utility_map(observed_positions, utility_map, ros_map, d_max=5, alpha=0.5):
    """Update utility map after drone observations"""
    rows, cols = utility_map.shape
    total_delta_u = 0
    
    for c_obs in observed_positions:
        # Record direct utility gained
        delta_u = utility_map[c_obs[0], c_obs[1]]
        utility_map[c_obs[0], c_obs[1]] *= 0.2  # Reduce utility of observed cell
        
        # Update indirect utility for nearby cells
        # for r in range(max(0, c_obs[0] - int(d_max)), min(rows, c_obs[0] + int(d_max) + 1)):
        #     for c in range(max(0, c_obs[1] - int(d_max)), min(cols, c_obs[1] + int(d_max) + 1)):
        #         if (r,c) == c_obs:
        #             continue
                    
        #         u_ind = compute_indirect_utility(c_obs, (r,c), ros_map, d_max, alpha)
        #         u_ind = min(utility_map[r,c], u_ind)
        #         utility_map[r,c] -= u_ind
        #         delta_u += u_ind
                
        total_delta_u += delta_u
    
    return total_delta_u, utility_map

def detect_fire_edges(fire_map):
    """Detect edges of the fire front and their directions"""
    # Transpose fire map to match coordinate system
    fire_map = fire_map.T
    edges = []
    edge_directions = []
    rows, cols = fire_map.shape
    
    # Find burning cells adjacent to unburned cells
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if fire_map[i, j] == 1:  # If cell is burning
                # Check 8 neighbors
                neighbors = [
                    (i-1, j), (i+1, j),  # North, South
                    (i, j-1), (i, j+1),  # West, East
                    (i-1, j-1), (i-1, j+1),  # NW, NE
                    (i+1, j-1), (i+1, j+1)   # SW, SE
                ]
                
                # If any neighbor is unburned (0), this is an edge
                for ni, nj in neighbors:
                    if fire_map[ni, nj] == 0:
                        # Calculate direction vector from burning to unburned
                        direction = np.array([ni - i, nj - j], dtype=float)
                        if not np.all(direction == 0):
                            direction = direction / np.linalg.norm(direction)
                            edges.append((j, i))  # Swap i,j to match coordinate system
                            edge_directions.append(direction)
                        break
    
    # Cluster similar directions
    if edge_directions:
        clustered_directions = cluster_directions(edge_directions)
        logger.info(f"Detected {len(clustered_directions)} distinct fire front directions")
        return edges, clustered_directions
    return edges, []

def cluster_directions(directions, angle_threshold=np.pi/4):
    """Cluster similar directions together"""
    clustered = []
    for direction in directions:
        # Check if direction is similar to any existing cluster
        found_cluster = False
        for i, existing in enumerate(clustered):
            angle = np.arccos(np.clip(np.dot(direction, existing), -1.0, 1.0))
            if angle < angle_threshold:
                # Update cluster with average direction
                clustered[i] = (existing + direction) / 2
                clustered[i] /= np.linalg.norm(clustered[i])
                found_cluster = True
                break
        if not found_cluster:
            clustered.append(direction)
    return clustered

def calculate_edge_coverage_reward(positions, edge_directions):
    """Calculate reward based on how well agents cover different fire front edges"""
    if not edge_directions or not positions:
        return 0.0
    
    # For each agent, find the closest edge direction they're following
    agent_directions = []
    for pos in positions:
        agent_vec = np.array(pos[:2], dtype=float)
        closest_direction = None
        min_angle = float('inf')
        
        for direction in edge_directions:
            angle = np.abs(np.arccos(np.clip(np.dot(agent_vec, direction), -1.0, 1.0)))
            if angle < min_angle:
                min_angle = angle
                closest_direction = direction
        
        if closest_direction is not None:
            agent_directions.append(closest_direction)
    
    # Calculate coverage score based on how many different directions are covered
    unique_directions = set(tuple(d) for d in agent_directions)
    coverage_score = len(unique_directions) / len(edge_directions)
    return coverage_score

def vns_step(current_positions, utility_map, fire_map, neighborhood_size=5):
    """Find better positions using VNS with improved utility calculation"""
    best_positions = current_positions.copy()
    best_utility = sum(utility_map[pos[0], pos[1]] for pos in current_positions)
    
    # Get fire front edges and directions
    edges, edge_directions = detect_fire_edges(fire_map)
    best_coverage = calculate_edge_coverage_reward(current_positions, edge_directions)
    
    # Try to improve each agent's position
    for agent_idx, pos in enumerate(current_positions):
        for size in range(1, neighborhood_size + 1):
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    new_x = min(max(0, pos[0] + dx), utility_map.shape[0]-1)
                    new_y = min(max(0, pos[1] + dy), utility_map.shape[1]-1)
                    
                    # Check minimum distance between agents
                    too_close = False
                    for other_idx, other_pos in enumerate(best_positions):
                        if other_idx != agent_idx:
                            dist = np.sqrt((new_x - other_pos[0])**2 + (new_y - other_pos[1])**2)
                            if dist < 2:  # Minimum separation distance
                                too_close = True
                                break
                    
                    if too_close:
                        continue
                    
                    # Try new position
                    new_positions = best_positions.copy()
                    new_positions[agent_idx] = (new_x, new_y)
                    
                    # Calculate utility including indirect effects
                    direct_utility = sum(utility_map[p[0], p[1]] for p in new_positions)
                    
                    # Calculate edge coverage reward
                    coverage_reward = calculate_edge_coverage_reward(new_positions, edge_directions)
                    
                    # Combine utilities with weights
                    combined_utility = (0.7 * direct_utility) + (0.3 * coverage_reward)
                    
                    if combined_utility > (0.7 * best_utility + 0.3 * best_coverage):
                        best_positions = new_positions.copy()
                        best_utility = direct_utility
                        best_coverage = coverage_reward
    
    return best_positions

def initialize_pygame_windows(width, height):
    """Initialize multiple Pygame windows"""
    pygame.init()
    # Create three windows
    utility_screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Utility Map")
    
    # Create additional windows using pygame.Surface
    ros_screen = pygame.display.set_mode((width, height), pygame.RESIZABLE, vsync=1)
    pygame.display.set_caption("Rate of Spread Map")
    
    fire_screen = pygame.display.set_mode((width, height), pygame.RESIZABLE, vsync=1)
    pygame.display.set_caption("Fire Map")
    
    return utility_screen, ros_screen, fire_screen

def update_displays(utility_screen, ros_screen, fire_screen, utility_map, ros_map, fire_map, agents):
    """Update all Pygame displays"""
    # Transpose arrays to match Pygame's coordinate system
    # utility_map = utility_map.T
    # ros_map = ros_map.T
    # fire_map = fire_map.T
    
    # Update utility map display
    normalized_utility = ((utility_map - np.min(utility_map)) / 
                        (np.max(utility_map) - np.min(utility_map)) * 255).astype(np.uint8)
    utility_surface = pygame.surfarray.make_surface(normalized_utility)
    utility_surface = pygame.transform.scale(utility_surface, utility_screen.get_size())
    utility_screen.fill((0, 0, 0))
    utility_screen.blit(utility_surface, (0, 0))
    
    # # Update ROS map display
    # normalized_ros = ((ros_map - np.min(ros_map)) / 
    #                  (np.max(ros_map) - np.min(ros_map)) * 255).astype(np.uint8)
    # ros_surface = pygame.surfarray.make_surface(normalized_ros)
    # ros_surface = pygame.transform.scale(ros_surface, ros_screen.get_size())
    # ros_screen.fill((0, 0, 0))
    # ros_screen.blit(ros_surface, (0, 0))
    
    # # Update fire map display (using different colors for fire states)
    # fire_display = np.zeros((fire_map.shape[0], fire_map.shape[1], 3), dtype=np.uint8)
    # fire_display[fire_map == 0] = [0, 0, 0]      # Unburned: black
    # fire_display[fire_map == 1] = [255, 0, 0]    # Burning: red
    # fire_display[fire_map == 2] = [128, 128, 128] # Burned out: gray
    # fire_surface = pygame.surfarray.make_surface(fire_display)
    # fire_surface = pygame.transform.scale(fire_surface, fire_screen.get_size())
    # fire_screen.fill((0, 0, 0))
    # fire_screen.blit(fire_surface, (0, 0))
    
    # Draw agents on all displays as white dots
    for agent in agents:
        x, y = agent[0], agent[1]  # No need to swap coordinates anymore
        pygame.draw.circle(utility_screen, (255, 255, 255), 
                         (int(x * utility_screen.get_width() / utility_map.shape[1]),
                          int(y * utility_screen.get_height() / utility_map.shape[0])), 3)
        # # Draw on ROS map
        # pygame.draw.circle(ros_screen, (255, 255, 255), 
        #                  (int(x * ros_screen.get_width() / ros_map.shape[1]),
        #                   int(x * ros_screen.get_height() / ros_map.shape[0])), 3)
        # # Draw on fire map
        # pygame.draw.circle(fire_screen, (255, 255, 255), 
        #                  (int(x * fire_screen.get_width() / fire_map.shape[1]),
        #                   int(y * fire_screen.get_height() / fire_map.shape[0])), 3)
    
    # Update all displays
    pygame.display.flip()

# Main simulation setup
config = Config("configs/operational_config.yml")
sim = FireSimulation(config)

# Initialize agents near fire
num_agents = 5
initial_fire_pos = eval(config.yaml_data["fire"]["fire_initial_position"]["static"]["position"])
initial_positions = [
    (initial_fire_pos[0] + 3, initial_fire_pos[1] + 3),
    (initial_fire_pos[0] + 3, initial_fire_pos[1] - 3),
    (initial_fire_pos[0] - 3, initial_fire_pos[1] + 3),
    (initial_fire_pos[0] - 3, initial_fire_pos[1] - 3),
    (initial_fire_pos[0], initial_fire_pos[1] + 4),
]
agents = [(pos[0], pos[1], i) for i, pos in enumerate(initial_positions)]

# Setup display
sim.rendering = True
sim.update_agent_positions(agents)

# Initialize Pygame windows
# utility_screen, ros_screen, fire_screen = initialize_pygame_windows(400, 400)

# Main simulation loop
start_time = time.time()
max_duration = 60
update_interval = 2
step_counter = 0

while time.time() - start_time < max_duration:
    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    
    # Run simulation step
    sim.run("5m")
    step_counter += 1
    logger.info(f"Step {step_counter}")
    
    if step_counter % update_interval == 0:
        # Get rate of spread from fire manager
        ros_map = sim.fire_manager.rate_of_spread
        
        # Create and update utility map
        utility_map = initialize_utility_map(sim.fire_map, ros_map)
        current_positions = [(agents[i][0], agents[i][1]) for i in range(num_agents)]
        
        # Update utility based on previous observations
        _, utility_map = update_utility_map(current_positions, utility_map, ros_map)
        
        # Find better positions using VNS with edge-based diversity
        new_positions = vns_step(current_positions, utility_map, sim.fire_map)
        
        # Update agent positions
        agents = [(pos[0], pos[1], i) for i, pos in enumerate(new_positions)]
        sim.update_agent_positions(agents)
        
        # Update all displays
        # update_displays(utility_screen, ros_screen, fire_screen, 
        #                utility_map, ros_map, sim.fire_map, agents)
        
        # Log positions
        logger.info("Agent positions updated:")
        for agent in agents:
            logger.info(f"Agent {agent[2]} at ({agent[0]}, {agent[1]})")
    
    time.sleep(0.1)

# Clean up Pygame
pygame.quit()
sim.save_gif()
sim.rendering = False
