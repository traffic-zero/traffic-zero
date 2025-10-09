from sim import run_interactive, run_automated, run_carla

# Call the runner for a specific simulation

# Use run_interactive for manual control and visual exploration (recommended)
run_interactive("simple4")

# Use run_automated for programmatic control and experiments (runs for 30 minutes)
# run_automated("simple4")

# Use run_carla for CARLA co-simulation
# run_carla("simple4")