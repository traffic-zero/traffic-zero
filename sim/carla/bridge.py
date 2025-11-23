"""
CARLA-SUMO Co-Simulation Bridge

This module provides the bridge between CARLA and SUMO simulators,
allowing SUMO traffic scenarios to run in CARLA's 3D environment.
"""

import os
import sys
import time
import argparse
import random

# Add CARLA to Python path if CARLA_ROOT is set
carla_root = os.environ.get("CARLA_ROOT")
if carla_root:
    # Add PythonAPI/carla to path
    carla_egg_path = os.path.join(carla_root, "PythonAPI", "carla")
    if os.path.exists(carla_egg_path) and carla_egg_path not in sys.path:
        sys.path.append(carla_egg_path)

    # Also try to find and add the .egg file for specific Python version
    dist_path = os.path.join(carla_egg_path, "dist")
    if os.path.exists(dist_path):
        # Find .egg file matching current Python version
        py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
        for egg_file in os.listdir(dist_path):
            if egg_file.endswith(".egg") and py_version in egg_file:
                egg_full_path = os.path.join(dist_path, egg_file)
                if egg_full_path not in sys.path:
                    sys.path.append(egg_full_path)
                break

try:
    import carla
except ImportError:
    raise ImportError(
        "CARLA Python API not found. Please install CARLA and add "
        "PythonAPI to your PYTHONPATH. See CARLA.md for details.\n"
        f"CARLA_ROOT is set to: {carla_root}\n"
        "Make sure CARLA is installed at that location."
    )

try:
    import traci
except ImportError:
    raise ImportError("TraCI not found. Please install: pip install traci")


class CarlaSumoSync:
    """
    Synchronizes SUMO and CARLA simulations.

    This class manages the connection between SUMO and CARLA,
    synchronizing vehicles, traffic lights, and simulation steps.
    """

    def __init__(
        self,
        sumo_cfg_file: str,
        carla_host: str = "localhost",
        carla_port: int = 2000,
        step_length: float = 0.05,
        sync_vehicle_lights: bool = True,
        sync_vehicle_color: bool = False,
        sync_all: bool = True,
        tls_manager: str = "sumo",
        carla_map: str | None = None,
        auto_camera: bool = False,
        use_sumo_network: bool = False,
    ):
        """
        Initialize CARLA-SUMO synchronization.

        Args:
            sumo_cfg_file: Path to SUMO configuration file
            carla_host: CARLA server hostname
            carla_port: CARLA server port
            step_length: Simulation step length in seconds
            sync_vehicle_lights: Synchronize vehicle lights
            sync_vehicle_color: Synchronize vehicle colors
            sync_all: Sync all vehicles automatically
            tls_manager: Traffic light manager ('sumo', 'carla', or 'none')
            carla_map: CARLA map to load (None=current, 'empty'=empty map)
            auto_camera: Automatically move camera to follow vehicles
                         (default: False)
            use_sumo_network: Load SUMO network as OpenDRIVE in CARLA
                              (default: False)
        """
        self.sumo_cfg = sumo_cfg_file
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.step_length = step_length
        self.sync_vehicle_lights = sync_vehicle_lights
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_all = sync_all
        self.tls_manager = tls_manager
        self.carla_map = carla_map
        self.auto_camera = auto_camera
        self.use_sumo_network = use_sumo_network

        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle_actors = {}  # Maps SUMO vehicle ID to CARLA actor
        self.spawn_transforms = []  # Available spawn points
        self.spectator = None  # Camera controller
        self.offset_x = 0.0  # Coordinate offset for SUMO->CARLA
        self.offset_y = 0.0
        self.scale = 1.0  # Scale factor

    def _load_sumo_network_as_opendrive(self) -> bool:
        """Load SUMO network as OpenDRIVE map in CARLA.

        Returns True if successful.
        """
        if self.client is None:
            return False
        sumo_dir = os.path.dirname(self.sumo_cfg)
        xodr_file = os.path.join(sumo_dir, os.path.basename(sumo_dir) + ".xodr")

        if os.path.exists(xodr_file):
            print(f"Loading SUMO network as OpenDRIVE: {xodr_file}...")
            with open(xodr_file, "r") as f:
                opendrive_content = f.read()

            self.world = self.client.generate_opendrive_world(
                opendrive_content,
                carla.OpendriveGenerationParameters(
                    vertex_distance=2.0,
                    max_road_length=50.0,
                    wall_height=0.0,
                    additional_width=0.6,
                    smooth_junctions=True,
                    enable_mesh_visibility=True,
                ),
            )
            print("✓ SUMO network loaded as CARLA map!")
            print("  Generated procedural 3D mesh from OpenDRIVE")
            return True

        print(f"✗ OpenDRIVE file not found: {xodr_file}")
        print("  Generating from SUMO network...")
        net_file = os.path.join(sumo_dir, "network.net.xml")
        if os.path.exists(net_file):
            import subprocess

            try:
                subprocess.run(
                    [
                        "netconvert",
                        "--sumo-net-file",
                        net_file,
                        "--opendrive-output",
                        xodr_file,
                    ],
                    check=True,
                )
                print(f"✓ Generated OpenDRIVE: {xodr_file}")

                with open(xodr_file, "r") as f:
                    opendrive_content = f.read()

                if self.client is not None:
                    self.world = self.client.generate_opendrive_world(
                        opendrive_content,
                        carla.OpendriveGenerationParameters(
                            vertex_distance=2.0,
                            max_road_length=50.0,
                            wall_height=0.0,
                            additional_width=0.6,
                            smooth_junctions=True,
                            enable_mesh_visibility=True,
                        ),
                    )
                    print("✓ SUMO network loaded as CARLA map!")
                    return True
            except Exception as e:
                print(f"✗ Failed to generate OpenDRIVE: {e}")
                print("  Falling back to default map...")
        else:
            print(f"✗ Network file not found: {net_file}")

        return False

    def _load_empty_map(self):
        """Load a minimal layered map (empty environment)."""
        if self.client is None:
            return
        layered_maps = ["Town01_Opt", "Town02_Opt", "Town03_Opt"]

        for map_name in layered_maps:
            try:
                print(f"Loading minimal layered map: {map_name}...")
                self.world = self.client.load_world(map_name)
                print(f"✓ Loaded {map_name} with minimal layers")
                print(
                    "  All decorative layers (buildings, props, foliage) "
                    "disabled"
                )
                return
            except Exception:
                continue

        # Fallback: try non-layered Town01
        print("⚠ No layered map found, trying Town01...")
        try:
            self.world = self.client.load_world("Town01")
            print("✓ Loaded Town01")
        except Exception:
            print("⚠ Failed to load Town01, using current map")
            self.world = self.client.get_world()

    def _load_specific_map(self):
        """Load a specific CARLA map by name."""
        if self.client is None:
            return
        available_maps = self.client.get_available_maps()
        map_to_load = self.carla_map

        if map_to_load not in available_maps:
            map_to_load = f"/Game/Carla/Maps/{self.carla_map}"

        if map_to_load in available_maps:
            print(f"Loading map: {map_to_load}...")
            self.world = self.client.load_world(map_to_load)
        else:
            print(f"⚠ Map '{self.carla_map}' not found")
            print(f"Available maps: {', '.join(available_maps[:5])}...")
            self.world = self.client.get_world()

    def _setup_carla_world(self):
        """Configure CARLA world settings and get required objects."""
        if self.world is None:
            raise RuntimeError("Failed to connect to CARLA world")

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.step_length
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_transforms = self.world.get_map().get_spawn_points()
        self.spectator = self.world.get_spectator()

        print(f"✓ Connected to CARLA (map: {self.world.get_map().name})")

    def connect_carla(self):
        """Connect to CARLA server."""
        print(
            f"Connecting to CARLA server at "
            f"{self.carla_host}:{self.carla_port}..."
        )
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(10.0)

        # Load SUMO network as OpenDRIVE if requested
        if self.use_sumo_network:
            if not self._load_sumo_network_as_opendrive():
                if self.client is not None:
                    self.world = self.client.get_world()
        elif self.carla_map:
            if self.carla_map.lower() == "empty":
                self._load_empty_map()
            else:
                self._load_specific_map()
        else:
            # Use current world
            if self.client is not None:
                self.world = self.client.get_world()

        # Configure world settings
        self._setup_carla_world()

    def setup_sumo(self):
        """Start SUMO via TraCI."""
        try:
            # Start SUMO with TraCI
            sumo_cmd = [
                "sumo",  # Use 'sumo-gui' for GUI version
                "-c",
                self.sumo_cfg,
                "--step-length",
                str(self.step_length),
                "--quit-on-end",
                "true",
                "--start",
                "true",
                "--no-warnings",
                "true",
            ]

            print(f"Starting SUMO with {self.sumo_cfg}...")
            traci.start(sumo_cmd)

            print("✓ SUMO started successfully")

        except Exception as e:
            print(f"✗ Failed to start SUMO: {e}")
            print("\nPlease ensure:")
            print("1. SUMO is installed and in PATH")
            print("2. Configuration file exists: " + self.sumo_cfg)
            raise

    def spawn_vehicle_in_carla(self, sumo_vehicle_id: str, position: tuple):
        """
        Spawn a vehicle in CARLA corresponding to a SUMO vehicle.

        Args:
            sumo_vehicle_id: SUMO vehicle ID
            position: (x, y, z) position from SUMO
        """
        if sumo_vehicle_id in self.vehicle_actors:
            return  # Already spawned

        if self.blueprint_library is None:
            return
        try:
            # Get a vehicle blueprint
            vehicle_blueprints = self.blueprint_library.filter("vehicle.*")
            vehicle_bp = random.choice(
                [
                    bp
                    for bp in vehicle_blueprints
                    if int(bp.get_attribute("number_of_wheels")) == 4
                ]
            )

            # Set random color for visibility
            if vehicle_bp.has_attribute("color"):
                color = random.choice(
                    vehicle_bp.get_attribute("color").recommended_values
                )
                vehicle_bp.set_attribute("color", color)

            # Transform SUMO coordinates to CARLA coordinates
            # SUMO: x=east, y=north
            # CARLA: x=forward, y=right (rotated 90 degrees)
            carla_x = position[0] + self.offset_x
            carla_y = -position[1] + self.offset_y  # Flip Y axis
            carla_z = 0.5  # Spawn slightly above ground

            transform = carla.Transform(
                carla.Location(x=carla_x, y=carla_y, z=carla_z)
            )

            # Spawn vehicle
            if self.world is None:
                return
            actor = self.world.try_spawn_actor(vehicle_bp, transform)

            if actor:
                self.vehicle_actors[sumo_vehicle_id] = actor
                print(
                    f"✓ Spawned vehicle {sumo_vehicle_id} at CARLA "
                    f"({carla_x:.1f}, {carla_y:.1f})"
                )

        except Exception as e:
            print(f"⚠ Failed to spawn {sumo_vehicle_id}: {e}")

    def update_vehicle_in_carla(
        self, sumo_vehicle_id: str, position: tuple, angle: float
    ):
        """
        Update vehicle position in CARLA based on SUMO state.

        Args:
            sumo_vehicle_id: SUMO vehicle ID
            position: (x, y, z) position
            angle: Heading angle in degrees
        """
        if sumo_vehicle_id not in self.vehicle_actors:
            self.spawn_vehicle_in_carla(sumo_vehicle_id, position)
            return

        actor = self.vehicle_actors[sumo_vehicle_id]

        try:
            # Transform coordinates
            carla_x = position[0] + self.offset_x
            carla_y = -position[1] + self.offset_y  # Flip Y
            carla_z = 0.5

            # SUMO angle: 0=north, increases clockwise
            # CARLA yaw: 0=east, increases counter-clockwise
            # Convert: CARLA_yaw = 90 - SUMO_angle
            carla_yaw = 90.0 - angle

            # Update transform
            transform = carla.Transform(
                carla.Location(x=carla_x, y=carla_y, z=carla_z),
                carla.Rotation(yaw=carla_yaw),
            )

            actor.set_transform(transform)

        except Exception:
            # Actor may have been destroyed
            pass

    def remove_vehicle_from_carla(self, sumo_vehicle_id: str):
        """Remove a vehicle from CARLA."""
        if sumo_vehicle_id in self.vehicle_actors:
            try:
                self.vehicle_actors[sumo_vehicle_id].destroy()
            except Exception:
                # Actor may have already been destroyed, ignore
                pass
            del self.vehicle_actors[sumo_vehicle_id]

    def synchronize_vehicles(self):
        """Synchronize all vehicles between SUMO and CARLA."""
        # Get all vehicle IDs from SUMO
        sumo_vehicles = set(traci.vehicle.getIDList())
        carla_vehicles = set(self.vehicle_actors.keys())

        # Remove vehicles that are no longer in SUMO
        for vehicle_id in carla_vehicles - sumo_vehicles:
            self.remove_vehicle_from_carla(vehicle_id)

        # Update/spawn vehicles from SUMO
        for vehicle_id in sumo_vehicles:
            try:
                # Get position from SUMO (returns x, y)
                pos_2d = traci.vehicle.getPosition(vehicle_id)
                position = (pos_2d[0], pos_2d[1], 0.0)

                # Get angle (may return tuple or float)
                angle_raw = traci.vehicle.getAngle(vehicle_id)
                # Handle both tuple and float returns
                if isinstance(angle_raw, tuple):
                    angle = angle_raw[0] if len(angle_raw) > 0 else 0.0
                else:
                    angle = float(angle_raw)

                # Update in CARLA
                self.update_vehicle_in_carla(vehicle_id, position, angle)

            except traci.TraCIException:
                # Vehicle might have left the simulation
                continue

    def adjust_traffic_light_height(self, z_offset: float = -1.5):
        """
        Adjust the height of all traffic lights in the world.

        Args:
            z_offset: Vertical offset to apply
                      (negative = lower, positive = higher)
                     Default: -1.5 meters (lowers traffic lights)

        To change the height, modify the z_offset parameter in connect_carla()
        """
        if self.world is None:
            return
        traffic_lights = self.world.get_actors().filter(
            "traffic.traffic_light*"
        )

        if not traffic_lights:
            print("  ⚠ No traffic lights found in the world")
            return

        adjusted_count = 0
        for tl in traffic_lights:
            try:
                current_transform = tl.get_transform()
                new_transform = carla.Transform(
                    carla.Location(
                        x=current_transform.location.x,
                        y=current_transform.location.y,
                        z=current_transform.location.z
                        + z_offset,  # Apply offset
                    ),
                    current_transform.rotation,
                )
                tl.set_transform(new_transform)
                adjusted_count += 1
            except Exception:
                continue

        if adjusted_count > 0:
            print(
                f"  ✓ Adjusted {adjusted_count} traffic light(s) by {z_offset}m"
            )
        else:
            print("  ⚠ Could not adjust traffic lights")

    def set_initial_camera_view(self):
        """Position camera at start to view the simulation area."""
        # Position camera to view the area where SUMO vehicles will appear
        # Based on SUMO coordinates, vehicles spawn around (0, 0)
        # in custom network
        if self.spectator is None:
            return
        camera_transform = carla.Transform(
            carla.Location(x=-50, y=0, z=80),  # Above and behind
            carla.Rotation(
                pitch=-45, yaw=0
            ),  # Looking down at the intersection
        )

        self.spectator.set_transform(camera_transform)

        print("✓ Camera positioned to view simulation area")
        print("  🎮 Camera Controls:")
        print("     • Mouse: Look around")
        print("     • W/A/S/D: Move forward/left/back/right")
        print("     • Q/E: Move down/up")
        print("     • Scroll UP: Increase movement speed ⚡")
        print("     • Scroll DOWN: Decrease movement speed")
        print("")
        print("  💡 Tip: Scroll up several times for FAST camera movement!")

    def update_spectator_camera(self):
        """Move spectator camera to follow the action."""
        if not self.vehicle_actors:
            return

        # Get average position of all vehicles
        positions = []
        for actor in self.vehicle_actors.values():
            try:
                loc = actor.get_location()
                positions.append((loc.x, loc.y))
            except Exception:
                continue

        if positions:
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)

            # Position camera above and behind the action
            if self.spectator is None:
                return
            camera_transform = carla.Transform(
                carla.Location(x=avg_x - 30, y=avg_y, z=25),
                carla.Rotation(pitch=-20, yaw=0),
            )

            self.spectator.set_transform(camera_transform)

    def _check_simulation_done(
        self, duration: int | None, start_time: float
    ) -> tuple[bool, str]:
        """Check if simulation should stop. Returns (should_stop, reason)."""
        if duration and (time.time() - start_time) > duration:
            return True, f"\n✓ Simulation completed ({duration}s)"

        min_expected = traci.simulation.getMinExpectedNumber()
        if isinstance(min_expected, tuple):
            min_expected = min_expected[0] if len(min_expected) > 0 else 0
        if min_expected <= 0:
            return True, "\n✓ SUMO simulation finished (no more vehicles)"

        return False, ""

    def _run_simulation_step(self, step: int):
        """Run a single simulation step."""
        traci.simulationStep()
        self.synchronize_vehicles()

        if self.auto_camera and step % 5 == 0:
            self.update_spectator_camera()

        if self.world is None:
            return False
        self.world.tick()
        return True

    def _print_progress(self, step: int, start_time: float):
        """Print simulation progress indicator."""
        if step % 20 == 0:  # Update every second (20 steps * 0.05s)
            elapsed = time.time() - start_time
            num_vehicles = len(traci.vehicle.getIDList())
            print(
                f"Step {step:5d} | Elapsed: {elapsed:>6.1f}s | "
                f"Vehicles: {num_vehicles:3d}",
                end="\r",
            )

    def run_cosimulation(self, duration: int | None = None):
        """
        Run the co-simulation.

        Args:
            duration: Simulation duration in seconds (None = infinite)
        """
        try:
            print("\n" + "=" * 60)
            print("Starting CARLA-SUMO Co-Simulation")
            print("=" * 60)

            self.connect_carla()
            self.setup_sumo()

            # Position camera to view the simulation area
            self.set_initial_camera_view()

            print("\nSimulation parameters:")
            print(f"  Step length: {self.step_length}s")
            print(f"  Duration: {duration if duration else 'infinite'}")
            print(f"  TLS manager: {self.tls_manager}")
            print(f"  Sync vehicle lights: {self.sync_vehicle_lights}")

            # Simulation loop
            step = 0
            start_time = time.time()

            print("\n▶ Simulation running... (Press Ctrl+C to stop)\n")

            while True:
                # Check if simulation should stop
                should_stop, stop_reason = self._check_simulation_done(
                    duration, start_time
                )
                if should_stop:
                    print(stop_reason)
                    break

                # Run simulation step
                if not self._run_simulation_step(step):
                    break

                # Print progress
                self._print_progress(step, start_time)

                step += 1

        except KeyboardInterrupt:
            print("\n\n⚠ Simulation interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        # Destroy all spawned vehicles in CARLA
        num_vehicles = len(self.vehicle_actors)
        for vehicle_id in self.vehicle_actors.keys():
            self.remove_vehicle_from_carla(vehicle_id)
        if num_vehicles > 0:
            print(f"✓ Destroyed {num_vehicles} CARLA vehicles")
        else:
            print("⚠ No vehicles were spawned in CARLA")

        # Restore CARLA settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            print("✓ CARLA settings restored")

        # Close SUMO/TraCI
        try:
            traci.close()
            print("✓ SUMO closed")
        except Exception:
            # TraCI may already be closed, ignore
            pass

        print("Cleanup complete.\n")


def main():
    """Main entry point for CARLA-SUMO co-simulation."""
    parser = argparse.ArgumentParser(
        description="Run CARLA-SUMO co-simulation for traffic scenarios"
    )

    parser.add_argument(
        "sumo_cfg", type=str, help="Path to SUMO configuration file (.sumocfg)"
    )

    parser.add_argument(
        "--carla-host",
        default="localhost",
        help="CARLA server host (default: localhost)",
    )

    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA server port (default: 2000)",
    )

    parser.add_argument(
        "--step-length",
        type=float,
        default=0.05,
        help="Simulation step length in seconds (default: 0.05)",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Simulation duration in seconds (default: infinite)",
    )

    parser.add_argument(
        "--tls-manager",
        choices=["sumo", "carla", "none"],
        default="sumo",
        help="Traffic light manager (default: sumo)",
    )

    parser.add_argument(
        "--sync-vehicle-lights",
        action="store_true",
        default=True,
        help="Synchronize vehicle lights (default: True)",
    )

    parser.add_argument(
        "--map", default="empty", help="CARLA map to use (default: empty)"
    )

    parser.add_argument(
        "--auto-camera",
        action="store_true",
        help="Automatically move camera to follow vehicles",
    )

    parser.add_argument(
        "--use-sumo-network",
        action="store_true",
        help="Load SUMO network as OpenDRIVE map in CARLA",
    )

    args = parser.parse_args()

    # Create and run co-simulation
    cosim = CarlaSumoSync(
        sumo_cfg_file=args.sumo_cfg,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        step_length=args.step_length,
        tls_manager=args.tls_manager,
        sync_vehicle_lights=args.sync_vehicle_lights,
        carla_map=args.map,
        auto_camera=args.auto_camera,
        use_sumo_network=args.use_sumo_network,
    )

    cosim.run_cosimulation(duration=args.duration)


if __name__ == "__main__":
    main()
