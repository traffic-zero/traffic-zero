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
from typing import Any, Optional
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces

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


def _normalize_traci_value(value: float | tuple | None) -> float:
    """Normalize traci return value to float (handles tuple returns)."""
    if value is None:
        return 0.0
    if isinstance(value, tuple):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)


class CarlaSumoSync(gym.Env):
    """
    Synchronizes SUMO and CARLA simulations.

    This class manages the connection between SUMO and CARLA,
    synchronizing vehicles, traffic lights, and simulation steps.

    Can be used as a Gymnasium environment for reinforcement learning
    when enable_rl_control=True, or as a standard co-simulation tool
    when enable_rl_control=False (default, backward compatible).
    """

    def __init__(
        self,
        sumo_cfg_file: str,
        **kwargs,
    ):
        """
        Initialize CARLA-SUMO synchronization.

        Args:
            sumo_cfg_file: Path to SUMO configuration file
            **kwargs: Additional configuration parameters:
                - carla_host: CARLA server hostname (default: "localhost")
                - carla_port: CARLA server port (default: 2000)
                - step_length:
                    Simulation step length in seconds (default: 0.05)
                - sync_vehicle_lights:
                    Synchronize vehicle lights (default: True)
                - sync_vehicle_color:
                    Synchronize vehicle colors (default: False)
                - sync_all: Sync all vehicles automatically (default: True)
                - tls_manager:
                    Traffic light manager ('sumo', 'carla', or 'none')
                    (default: "sumo")
                - carla_map:
                    CARLA map to load (None=current, 'empty'=empty map)
                    (default: None)
                - auto_camera:
                    Automatically move camera to follow vehicles
                    (default: False)
                - use_sumo_network:
                    Load SUMO network as OpenDRIVE in CARLA
                    (default: False)
                - enable_rl_control:
                    Enable RL action space for traffic light control
                    (default: False, for backward compatibility)
                - observation_config:
                    Configuration dict for observation space.
                    Options:
                    - 'shape': tuple, observation shape (default: (10,))
                    - 'low': float, lower bound (default: 0.0)
                    - 'high': float, upper bound (default: inf)
                    - 'include_lane_metrics': bool, include lane data
                    - 'reward_waiting_time': bool, use waiting time in reward
                    - 'reward_throughput': bool, use throughput in reward
                    - 'max_duration': float, max episode duration in seconds
                    (None = use defaults)
                - action_config:
                    Configuration dict for action space.
                    Options:
                    - 'num_phases':
                        int, number of traffic light phases
                        (default: 4)
                    - 'num_traffic_lights':
                        int, number of TLS
                        (default: 1)
                    (None = use defaults)
        """
        super().__init__()

        # Extract parameters from kwargs with defaults
        self.sumo_cfg = sumo_cfg_file
        self.carla_host = kwargs.get("carla_host", "localhost")
        self.carla_port = kwargs.get("carla_port", 2000)
        self.step_length = kwargs.get("step_length", 0.05)
        self.sync_vehicle_lights = kwargs.get("sync_vehicle_lights", True)
        self.sync_vehicle_color = kwargs.get("sync_vehicle_color", False)
        self.sync_all = kwargs.get("sync_all", True)
        self.tls_manager = kwargs.get("tls_manager", "sumo")
        self.carla_map = kwargs.get("carla_map", None)
        self.auto_camera = kwargs.get("auto_camera", False)
        self.use_sumo_network = kwargs.get("use_sumo_network", False)
        self.enable_rl_control = kwargs.get("enable_rl_control", False)
        self.observation_config = kwargs.get("observation_config", {})
        self.action_config = kwargs.get("action_config", {})

        self.client = None
        self.world = None
        self.blueprint_library = None
        self.vehicle_actors = {}  # Maps SUMO vehicle ID to CARLA actor
        self.spawn_transforms = []  # Available spawn points
        self.spectator = None  # Camera controller
        self.offset_x = 0.0  # Coordinate offset for SUMO->CARLA
        self.offset_y = 0.0
        self.scale = 1.0  # Scale factor

        # RL state tracking
        self._initialized = False
        self._step_count = 0
        self._start_time: float | None = None
        self._tls_ids: list[str] = []
        self._tls_controller: Any | None = None

        # Initialize action and observation spaces
        self._initialize_spaces()

    def _initialize_spaces(self):
        """Initialize action and observation spaces based on configuration."""
        if self.enable_rl_control:
            # Get traffic light IDs (will be populated after SUMO setup)
            # For now, use default action space
            num_phases = self.action_config.get("num_phases", 4)
            num_tls = self.action_config.get("num_traffic_lights", 1)

            if num_tls == 1:
                # Single traffic light: Discrete action space
                self.action_space = spaces.Discrete(num_phases)
            else:
                # Multiple traffic lights: MultiDiscrete action space
                self.action_space = spaces.MultiDiscrete([num_phases] * num_tls)
        else:
            # No-op action space for backward compatibility
            self.action_space = spaces.Discrete(1)

        # Initialize observation space (will be updated after SUMO setup)
        obs_shape = self.observation_config.get("shape", (10,))
        obs_low = self.observation_config.get("low", 0.0)
        obs_high = self.observation_config.get("high", np.inf)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=np.float32,
        )

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

            # Increase timeout for world generation (can take 30+ seconds)
            # Store original timeout (default is 10.0 seconds)
            original_timeout = 10.0
            self.client.set_timeout(60.0)  # 60 seconds for world generation
            
            try:
                print("  Generating 3D world from OpenDRIVE (this may take 30-60 seconds)...")
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
            except RuntimeError as e:
                error_msg = str(e)
                if "time-out" in error_msg.lower() or "timeout" in error_msg.lower():
                    print(f"✗ Timeout while generating CARLA world: {error_msg}")
                    print("\n  This usually means:")
                    print("  1. CARLA server is not running or not ready")
                    print("  2. CARLA server is overloaded or slow")
                    print("  3. The OpenDRIVE file is too complex")
                    print("\n  Please ensure CARLA is running and try again.")
                else:
                    print(f"✗ Error generating CARLA world: {error_msg}")
                return False
            except Exception as e:
                print(f"✗ Unexpected error loading OpenDRIVE: {e}")
                return False
            finally:
                # Restore original timeout
                self.client.set_timeout(original_timeout)

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
                    # Increase timeout for world generation (can take 30+ seconds)
                    original_timeout = self.client.get_timeout()
                    self.client.set_timeout(60.0)  # 60 seconds for world generation
                    
                    try:
                        print("  Generating 3D world from OpenDRIVE (this may take 30-60 seconds)...")
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
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "time-out" in error_msg.lower() or "timeout" in error_msg.lower():
                            print(f"✗ Timeout while generating CARLA world: {error_msg}")
                            print("\n  This usually means:")
                            print("  1. CARLA server is not running or not ready")
                            print("  2. CARLA server is overloaded or slow")
                            print("  3. The OpenDRIVE file is too complex")
                            print("\n  Please ensure CARLA is running and try again.")
                        else:
                            print(f"✗ Error generating CARLA world: {error_msg}")
                        return False
                    except Exception as e:
                        print(f"✗ Unexpected error loading OpenDRIVE: {e}")
                        return False
                    finally:
                        # Restore original timeout
                        self.client.set_timeout(original_timeout)
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

        # Verify CARLA is responding before proceeding (with retries)
        max_retries = 5
        retry_delay = 2.0
        for attempt in range(max_retries):
            try:
                _ = self.client.get_world()
                print("✓ CARLA server is responding")
                break
            except RuntimeError as e:
                error_msg = str(e)
                if "time-out" in error_msg.lower() or "timeout" in error_msg.lower():
                    if attempt < max_retries - 1:
                        print(f"  Waiting for CARLA to be ready... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise RuntimeError(
                            f"CARLA server is not responding at {self.carla_host}:{self.carla_port} "
                            f"after {max_retries} attempts. "
                            "Please ensure CARLA is running and try again.\n"
                            f"Error: {error_msg}"
                        ) from e
                raise

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

    def spawn_vehicle_in_carla(self, sumo_vehicle_id: str, position: tuple, angle: float = 0.0):
        """
        Spawn a vehicle in CARLA corresponding to a SUMO vehicle.

        Args:
            sumo_vehicle_id: SUMO vehicle ID
            position: (x, y, z) position from SUMO
            angle: Heading angle in degrees from SUMO (0=north, increases clockwise)
        """
        if sumo_vehicle_id in self.vehicle_actors:
            return  # Already spawned

        if self.blueprint_library is None:
            return
        try:
            # Get vehicle blueprints - filter to only regular cars (exclude trucks, buses, etc.)
            vehicle_blueprints = self.blueprint_library.filter("vehicle.*")
            
            # Filter to only regular cars (exclude trucks, buses, emergency vehicles)
            car_blueprints = [
                bp
                for bp in vehicle_blueprints
                if int(bp.get_attribute("number_of_wheels")) == 4
                and "truck" not in bp.id.lower()
                and "bus" not in bp.id.lower()
                and "fire" not in bp.id.lower()
                and "police" not in bp.id.lower()
                and "ambulance" not in bp.id.lower()
                and "bicycle" not in bp.id.lower()
                and "motorcycle" not in bp.id.lower()
            ]
            
            # Fallback to any 4-wheel vehicle if no cars found
            if not car_blueprints:
                car_blueprints = [
                    bp
                    for bp in vehicle_blueprints
                    if int(bp.get_attribute("number_of_wheels")) == 4
                ]
            
            if not car_blueprints:
                print(f"⚠ No suitable vehicle blueprints found for {sumo_vehicle_id}")
                return
            
            vehicle_bp = random.choice(car_blueprints)

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

            # Convert SUMO angle to CARLA yaw
            # SUMO angle: 0=north, increases clockwise
            # CARLA yaw: 0=east, increases counter-clockwise
            # To fix opposite rotation: negate the conversion
            # Original: carla_yaw = 90 - angle (was causing opposite rotation)
            # Fixed: carla_yaw = -(90 - angle) = angle - 90
            carla_yaw = angle - 90.0

            transform = carla.Transform(
                carla.Location(x=carla_x, y=carla_y, z=carla_z),
                carla.Rotation(yaw=carla_yaw),
            )

            # Spawn vehicle
            if self.world is None:
                return
            actor = self.world.try_spawn_actor(vehicle_bp, transform)

            if actor:
                self.vehicle_actors[sumo_vehicle_id] = actor
                print(
                    f"✓ Spawned vehicle {sumo_vehicle_id} at CARLA "
                    f"({carla_x:.1f}, {carla_y:.1f}) with yaw {carla_yaw:.1f}°"
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
            self.spawn_vehicle_in_carla(sumo_vehicle_id, position, angle)
            return

        actor = self.vehicle_actors[sumo_vehicle_id]

        try:
            # Transform coordinates
            carla_x = position[0] + self.offset_x
            carla_y = -position[1] + self.offset_y  # Flip Y
            carla_z = 0.5

            # SUMO angle: 0=north, increases clockwise
            # CARLA yaw: 0=east, increases counter-clockwise
            # To fix opposite rotation: negate the conversion
            # Original: carla_yaw = 90 - angle (was causing opposite rotation)
            # Fixed: carla_yaw = -(90 - angle) = angle - 90
            carla_yaw = angle - 90.0

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
        # Ensure duration is a number, not a string
        if duration is not None:
            try:
                duration = int(duration)
            except (ValueError, TypeError):
                # If duration is not a valid number, ignore it
                duration = None
        
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

    def _get_traffic_light_phases(self, obs_features: list[float]) -> None:
        """Extract traffic light phases and append to observation features."""
        if not self._tls_ids:
            return

        for tls_id in self._tls_ids[:5]:  # Limit to first 5 TLS
            try:
                phase = traci.trafficlight.getPhase(tls_id)
                if isinstance(phase, tuple):
                    phase = phase[0] if len(phase) > 0 else 0
                obs_features.append(float(phase))
            except traci.TraCIException:
                obs_features.append(0.0)

    def _get_lane_metrics(self, obs_features: list[float]) -> None:
        """Extract lane metrics and append to observation features."""
        if not self.observation_config.get("include_lane_metrics", False):
            return

        try:
            lane_ids = traci.lane.getIDList()
            for lane_id in lane_ids[:5]:  # Limit to first 5 lanes
                try:
                    occupancy = traci.lane.getLastStepOccupancy(lane_id)
                    if isinstance(occupancy, tuple):
                        occupancy = occupancy[0] if len(occupancy) > 0 else 0.0
                    obs_features.append(float(occupancy))
                except traci.TraCIException:
                    obs_features.append(0.0)
        except traci.TraCIException:
            pass

    def _normalize_observation_size(
        self, obs_features: list[float]
    ) -> np.ndarray:
        """Normalize observation features to match observation space shape."""
        if (
            not hasattr(self.observation_space, "shape")
            or self.observation_space.shape is None
        ):
            # Fallback to default shape if shape is not available
            target_size = 10
        else:
            target_size = self.observation_space.shape[0]

        # Pad or truncate to match target size
        obs_features = obs_features + [0.0] * (target_size - len(obs_features))

        return np.array(obs_features, dtype=np.float32)

    def _get_observation(self) -> np.ndarray:
        """
        Extract observation from current simulation state.

        Returns:
            numpy array with observation values
        """
        try:
            obs_features = []

            # Get simulation time and step
            sim_time = _normalize_traci_value(traci.simulation.getTime())
            obs_features.append(float(sim_time))
            obs_features.append(float(self._step_count))

            # Get vehicle count
            vehicle_ids = traci.vehicle.getIDList()
            num_vehicles = len(vehicle_ids)
            obs_features.append(float(num_vehicles))

            # Get traffic light states if available
            self._get_traffic_light_phases(obs_features)

            # Get lane metrics if configured
            self._get_lane_metrics(obs_features)

            # Normalize size and return
            return self._normalize_observation_size(obs_features)

        except Exception:
            # Return zero observation on error
            if (
                hasattr(self.observation_space, "shape")
                and self.observation_space.shape is not None
            ):
                shape = self.observation_space.shape
            else:
                shape = (10,)
            return np.zeros(shape, dtype=np.float32)

    def _calculate_waiting_time_reward(self) -> float:
        """Calculate negative reward based on total waiting time."""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            total_waiting = 0.0
            for veh_id in vehicle_ids:
                try:
                    waiting = traci.vehicle.getWaitingTime(veh_id)
                    if isinstance(waiting, tuple):
                        waiting = waiting[0] if len(waiting) > 0 else 0.0
                    total_waiting += float(waiting)
                except traci.TraCIException:
                    continue
            return -total_waiting * 0.01  # Scale down
        except traci.TraCIException:
            return 0.0

    def _calculate_throughput_reward(self) -> float:
        """Calculate positive reward based on vehicle throughput."""
        try:
            vehicle_ids = traci.vehicle.getIDList()
            return len(vehicle_ids) * 0.1
        except traci.TraCIException:
            return 0.0

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on traffic metrics.

        Returns:
            Reward value (default: 0.0)
        """
        if not self.enable_rl_control:
            return 0.0

        try:
            reward = 0.0

            # Negative reward for waiting time (minimize waiting)
            if self.observation_config.get("reward_waiting_time", False):
                reward += self._calculate_waiting_time_reward()

            # Positive reward for throughput (maximize vehicles)
            if self.observation_config.get("reward_throughput", False):
                reward += self._calculate_throughput_reward()

            return float(reward)

        except Exception:
            return 0.0

    def _apply_action(self, action: Any):
        """
        Apply traffic light control action.

        Args:
            action: Action from action space
        """
        if not self.enable_rl_control or not self._tls_ids:
            return

        try:
            if isinstance(self.action_space, spaces.Discrete):
                # Single traffic light
                if len(self._tls_ids) > 0:
                    tls_id = self._tls_ids[0]
                    phase = int(action)
                    traci.trafficlight.setPhase(tls_id, phase)
            elif isinstance(self.action_space, spaces.MultiDiscrete):
                # Multiple traffic lights
                action_array = np.asarray(action)
                for i, tls_id in enumerate(self._tls_ids):
                    if i < len(action_array):
                        phase = int(action_array[i])
                        traci.trafficlight.setPhase(tls_id, phase)
        except Exception:
            # Silently ignore action errors
            pass

    def _get_num_phases_from_tls(self, tls_id: str) -> int:
        """Get number of phases for a traffic light."""
        try:
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                tls_id
            )
            if program and len(program) > 0:
                return len(program[0].phases)
        except Exception:
            pass
        return 4  # Default

    def _update_action_space_for_multiple_tls(self) -> None:
        """Update action space when multiple traffic lights are present."""
        if not self.enable_rl_control or not self._tls_ids:
            return

        num_tls = len(self._tls_ids)
        if num_tls <= 1:
            return

        num_phases = self._get_num_phases_from_tls(self._tls_ids[0])
        self.action_space = spaces.MultiDiscrete([num_phases] * num_tls)

    def _initialize_simulation(self) -> None:
        """Initialize CARLA and SUMO connections."""
        # Clean up existing simulation if running
        if self._initialized:
            try:
                self.cleanup()
            except Exception:
                pass

        # Initialize CARLA and SUMO
        self.connect_carla()
        self.setup_sumo()

        # Position camera
        self.set_initial_camera_view()

    def _collect_traffic_light_ids(self) -> None:
        """Collect and update traffic light IDs from SUMO."""
        try:
            self._tls_ids = list(traci.trafficlight.getIDList())
            self._update_action_space_for_multiple_tls()
        except traci.TraCIException:
            self._tls_ids = []

    def reset(  # type: ignore[override]
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset

        Returns:
            Tuple of (observation, info)
        """
        # Initialize simulation
        self._initialize_simulation()

        # Get traffic light IDs and update action space
        self._collect_traffic_light_ids()

        # Reset state tracking
        self._initialized = True
        self._step_count = 0
        self._start_time = time.time()

        # Get initial observation
        observation = self._get_observation()
        info = {
            "step": 0,
            "time": _normalize_traci_value(traci.simulation.getTime()),
            "num_vehicles": len(traci.vehicle.getIDList()),
            "tls_ids": self._tls_ids,
        }

        return observation, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self._initialized:
            raise RuntimeError(
                "Environment not initialized. Call reset() first."
            )

        # Use run_steps to execute one step
        result = self.run_steps(num_steps=1, action=action)

        return (
            result["observation"],
            result["reward"],
            result["done"],
            result["truncated"],
            result["info"],
        )

    def run_steps(
        self, num_steps: int = 1, action: Any | None = None
    ) -> dict[str, Any]:
        """
        Execute one or more simulation steps.

        Args:
            num_steps: Number of simulation steps to execute
            action: Optional action to apply (if enable_rl_control=True)

        Returns:
            Dictionary with step results:
                - observation: Current observation
                - reward: Reward value
                - done: Whether episode is done
                - truncated: Whether episode was truncated
                - info: Additional info dict
        """
        if not self._initialized:
            raise RuntimeError(
                "Simulation not initialized. "
                "Call reset() or run_cosimulation() first."
            )

        # Apply action if provided
        if action is not None and self.enable_rl_control:
            self._apply_action(action)

        # Execute steps
        for _ in range(num_steps):
            if not self._run_simulation_step(self._step_count):
                break
            self._step_count += 1

        # Get observation and reward
        observation = self._get_observation()
        reward = self._calculate_reward()

        # Check if done/truncated
        done = False
        truncated = False

        # Check if simulation finished
        try:
            min_expected = traci.simulation.getMinExpectedNumber()
            if isinstance(min_expected, tuple):
                min_expected = min_expected[0] if len(min_expected) > 0 else 0
            if min_expected <= 0:
                done = True
        except traci.TraCIException:
            pass

        # Check duration limit
        if self._start_time is not None:
            max_duration = self.observation_config.get("max_duration", None)
            if max_duration and (time.time() - self._start_time) > max_duration:
                truncated = True

        info = {
            "step": self._step_count,
            "time": _normalize_traci_value(traci.simulation.getTime()),
            "num_vehicles": len(traci.vehicle.getIDList()),
        }

        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": info,
        }

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

            # Initialize simulation (equivalent to reset)
            self.connect_carla()
            self.setup_sumo()

            # Position camera to view the simulation area
            self.set_initial_camera_view()

            # Initialize state tracking
            self._initialized = True
            self._step_count = 0
            self._start_time = time.time()

            # Get traffic light IDs
            try:
                self._tls_ids = list(traci.trafficlight.getIDList())
            except traci.TraCIException:
                self._tls_ids = []

            print("\nSimulation parameters:")
            print(f"  Step length: {self.step_length}s")
            print(f"  Duration: {duration if duration else 'infinite'}")
            print(f"  TLS manager: {self.tls_manager}")
            print(f"  Sync vehicle lights: {self.sync_vehicle_lights}")

            print("\n▶ Simulation running... (Press Ctrl+C to stop)\n")

            # Use run_steps in a loop
            while True:
                # Check if simulation should stop
                should_stop, stop_reason = self._check_simulation_done(
                    duration, self._start_time
                )
                if should_stop:
                    print(stop_reason)
                    break

                # Run one step using run_steps
                try:
                    result = self.run_steps(num_steps=1, action=None)
                    if result["done"] or result["truncated"]:
                        break
                except Exception as e:
                    print(f"\n⚠ Error during simulation step: {e}")
                    break

                # Print progress
                self._print_progress(self._step_count, self._start_time)

        except KeyboardInterrupt:
            print("\n\n⚠ Simulation interrupted by user")

        finally:
            self.cleanup()

    def render(self):
        """
        Render the environment (delegates to camera logic).

        For CARLA-SUMO, rendering is handled by the CARLA viewer.
        This method can be used to update camera position if needed.
        """
        if self.auto_camera and self._step_count % 5 == 0:
            self.update_spectator_camera()

    def close(self):
        """
        Close the environment and clean up resources.

        This method is called by Gymnasium and delegates to cleanup().
        """
        self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        # Destroy all spawned vehicles in CARLA
        num_vehicles = len(self.vehicle_actors)
        # Note: we'll modify the dict during iteration
        vehicle_ids = list(self.vehicle_actors.keys())
        for vehicle_id in vehicle_ids:
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

        # Reset state tracking
        self._initialized = False
        self._step_count = 0
        self._start_time = None
        self._tls_ids = []

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
