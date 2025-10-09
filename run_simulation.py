from sim import run_intersection, run_intersection_gui, run_in_carla

# Call the runner for a specific simulation
# Use run_intersection_gui for interactive SUMO-GUI (recommended)
run_intersection("simple4")
# run_intersection_gui("simple4")

# Use run_intersection for TraCI-controlled simulation (runs for 30 minutes)
# run_intersection("simple4")

# run_in_carla("simple4")