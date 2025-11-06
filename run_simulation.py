from sim import run_interactive

# Call the runner for a specific simulation

# Use run_interactive for manual control and visual exploration (recommended)
# Basic usage with default XML files:
run_interactive("simple4")

# Use with experiment scenario (generates routes/tls from YAML):
# run_interactive("simple4", experiment_name="light_traffic")
# run_interactive("simple4", experiment_name="light_traffic_random")
# run_interactive("simple4", experiment_name="rush_hour")

# Use run_automated for programmatic control and experiments (runs for 30 minutes)
# Basic usage:
# run_automated("simple4")

# Use with experiment scenario:
# run_automated("simple4", experiment_name="light_traffic")
# run_automated("simple4", experiment_name="rush_hour")

# Use run_carla for CARLA co-simulation
# run_carla("simple4")