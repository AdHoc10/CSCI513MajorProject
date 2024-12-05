from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config
import time

config = Config("configs/operational_config.yml")
sim = FireSimulation(config)

# Simple Simulation Rendering
# Render the next 2 hours of simulation
# sim.rendering = True
# sim.run("10h")

# Now save a GIF and fire spread graph from the last 2 hours of simulation
# sim.save_gif()
# sim.save_spread_graph()
# # Saved to the location specified in the config: simulation.sf_home

# Agent MovementRendering
# Update agents for display
# (x, y, agent_id)
agent_0 = (5, 5, 0)
agent_1 = (5, 5, 1)

agents = [agent_0, agent_1]

# Create the agents on the display
sim.rendering = True
sim.update_agent_positions(agents)

# Loop through to move agents with more visible steps
start_time = time.time()
max_duration = 300  # Maximum runtime in seconds
i = 0

# Get grid size
grid_size = sim.config.area.screen_size[0] - 1  # Subtract 1 since indices are 0-based
while time.time() - start_time < max_duration:
    # Ensure positions stay within bounds
    pos = min(5 + i, grid_size)
    agent_0 = (pos, pos, 0)
    agent_1 = (pos, pos, 1)
    sim.update_agent_positions([agent_0, agent_1])
    sim.run("5m")  # Run for 1 minute simulation time
    time.sleep(0.1)  # Add small delay between updates
    i += 1

# # Save the GIF while rendering is still on
# sim.save_gif()
# sim.save_spread_graph()

# Now turn off rendering if needed
sim.rendering = False
