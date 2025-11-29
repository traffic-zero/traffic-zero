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
import subprocess
from pathlib import Path
from typing import Any
import numpy as np
import traci
import gymnasium as gym
from gymnasium import spaces

# Add CARLA to Python path if CARLA_ROOT is set
carla_root = os.environ.get("CARLA_ROOT")
if carla_root:
    carla_egg_path = os.path.join(carla_root, "PythonAPI", "carla")
    if os.path.exists(carla_egg_path) and carla_egg_path not in sys.path:
        sys.path.append(carla_egg_path)

    dist_path = os.path.join(carla_egg_path, "dist")
    if os.path.exists(dist_path):
        py_version = f"py{sys.version_info.major}{sys.version_info.minor}"
        for egg_file in os.listdir(dist_path):
            if egg_file.endswith(".egg") and py_version in egg_file:
                egg_full_path = os.path.join(dist_path, egg_file)
                if egg_full_path not in sys.path:
                    sys.path.append(egg_full_path)
                break

# Import CARLA with proper error message
try:
    import carla
except ImportError:
    error_msg = (
        "CARLA Python API not found. Please install CARLA and add "
        "PythonAPI to your PYTHONPATH. See CARLA.md for details.\n"
        f"CARLA_ROOT is set to: {carla_root}\n"
        "Make sure CARLA is installed at that location."
    )
    raise ImportError(error_msg)


# Error message constants
CARLA_CLIENT_ERROR_MSG = "CARLA client must be initialized"
CARLA_WORLD_ERROR_MSG = "CARLA world must be initialized"
SPECTATOR_ERROR_MSG = "Spectator must be initialized"
GYMNASIUM_SPACES_ERROR_MSG = "gymnasium.spaces must be available"
BLUEPRINT_ERROR_MSG = "Blueprint library must be initialized"


def _normalize_traci_value(value: float | tuple | None) -> float:
    """Normalize traci return value to float (handles tuple returns)."""
    if value is None:
        return 0.0
    if isinstance(value, tuple):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)


class CarlaSumoSync:
    """
    Base class for CARLA-SUMO synchronization.

    Handles connection between SUMO and CARLA simulators,
    synchronizing vehicles, traffic lights, and simulation steps.
    This class does not depend on Gymnasium and provides core
    synchronization functionality.
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
            use_sumo_network: Load SUMO network as OpenDRIVE in CARLA
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

        self.client: carla.Client | None = None
        self.world: carla.World | None = None
        self.blueprint_library: carla.BlueprintLibrary | None = None
        self.vehicle_actors: dict[str, carla.Actor] = {}
        self.spawn_transforms: list[carla.Transform] = []
        self.spectator: carla.Actor | None = None
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.scale = 1.0

        self._initialized = False
        self._step_count = 0
        self._start_time: float | None = None
        self._tls_ids: list[str] = []

    def _get_opendrive_parameters(self) -> carla.OpendriveGenerationParameters:
        """Get OpenDRIVE generation parameters for CARLA."""
        return carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=50.0,
            wall_height=0.0,
            additional_width=0.6,
            smooth_junctions=True,
            enable_mesh_visibility=True,
        )

    def _load_existing_opendrive_file(self, xodr_file: str) -> bool:
        """
        Load existing OpenDRIVE file into CARLA.

        Args:
            xodr_file: Path to OpenDRIVE file

        Returns:
            True if successfully loaded, False otherwise
        """
        if not os.path.exists(xodr_file):
            return False

        print(f"Loading SUMO network as OpenDRIVE: {xodr_file}...")
        with open(xodr_file, "r", encoding="utf-8") as f:
            opendrive_content = f.read()

        assert self.client is not None, CARLA_CLIENT_ERROR_MSG
        self.world = self.client.generate_opendrive_world(
            opendrive_content, self._get_opendrive_parameters()
        )
        print("✓ SUMO network loaded as CARLA map!")
        print("  Generated procedural 3D mesh from OpenDRIVE")
        return True

    def _generate_opendrive_from_network(
        self, net_file: str, xodr_file: str
    ) -> bool:
        """
        Generate OpenDRIVE file from SUMO network file.

        Args:
            net_file: Path to SUMO network file
            xodr_file: Path to output OpenDRIVE file

        Returns:
            True if successfully generated, False otherwise
        """
        assert os.path.exists(net_file), (
            f"Network file not found: {net_file}. "
            "Please generate network files first."
        )

        print(f"✗ OpenDRIVE file not found: {xodr_file}")
        print("  Generating from SUMO network...")

        result = subprocess.run(
            [
                "netconvert",
                "--sumo-net-file",
                net_file,
                "--opendrive-output",
                xodr_file,
            ],
            check=False,
            capture_output=True,
        )

        assert result.returncode == 0, (
            f"Failed to generate OpenDRIVE: {result.stderr.decode()}. "
            "Make sure netconvert is in PATH."
        )

        print(f"✓ Generated OpenDRIVE: {xodr_file}")
        return True

    def _load_sumo_network_as_opendrive(self) -> None:
        """
        Load SUMO network as OpenDRIVE map in CARLA.

        Attempts to load existing OpenDRIVE file, or generates it from
        SUMO network if not found.
        """
        assert self.client is not None, CARLA_CLIENT_ERROR_MSG

        sumo_dir = os.path.dirname(self.sumo_cfg)
        xodr_file = os.path.join(sumo_dir, os.path.basename(sumo_dir) + ".xodr")

        if self._load_existing_opendrive_file(xodr_file):
            return

        net_file = os.path.join(sumo_dir, "network.net.xml")
        if not self._generate_opendrive_from_network(net_file, xodr_file):
            return

        with open(xodr_file, "r", encoding="utf-8") as f:
            opendrive_content = f.read()

        assert self.client is not None, CARLA_CLIENT_ERROR_MSG
        self.world = self.client.generate_opendrive_world(
            opendrive_content, self._get_opendrive_parameters()
        )
        print("✓ SUMO network loaded as CARLA map!")

    def _load_empty_map(self) -> None:
        """Load a minimal layered map (empty environment)."""
        assert self.client is not None, CARLA_CLIENT_ERROR_MSG

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
            except RuntimeError:
                continue

        print("⚠ No layered map found, trying Town01...")
        try:
            self.world = self.client.load_world("Town01")
            print("✓ Loaded Town01")
        except RuntimeError:
            print("⚠ Failed to load Town01, using current map")
            self.world = self.client.get_world()

    def _load_specific_map(self) -> None:
        """Load a specific CARLA map by name."""
        assert self.client is not None, CARLA_CLIENT_ERROR_MSG

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

    def _setup_carla_world(self) -> None:
        """Configure CARLA world settings and get required objects."""
        assert self.world is not None, (
            "CARLA world must be initialized. "
            "Failed to connect to CARLA world."
        )

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.step_length
        self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_transforms = self.world.get_map().get_spawn_points()
        self.spectator = self.world.get_spectator()

        map_name = self.world.get_map().name
        print(f"✓ Connected to CARLA (map: {map_name})")

    def connect_carla(self) -> None:
        """Connect to CARLA server."""
        print(
            f"Connecting to CARLA server at "
            f"{self.carla_host}:{self.carla_port}..."
        )
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(10.0)

        if self.use_sumo_network:
            self._load_sumo_network_as_opendrive()
            if self.world is None:
                assert self.client is not None, CARLA_CLIENT_ERROR_MSG
                self.world = self.client.get_world()
        elif self.carla_map:
            if self.carla_map.lower() == "empty":
                self._load_empty_map()
            else:
                self._load_specific_map()
        else:
            assert self.client is not None, CARLA_CLIENT_ERROR_MSG
            self.world = self.client.get_world()

        self._setup_carla_world()

    def setup_sumo(self) -> None:
        """Start SUMO via TraCI."""
        sumo_cmd = [
            "sumo",
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

    def spawn_vehicle_in_carla(
        self, sumo_vehicle_id: str, position: tuple[float, float, float]
    ) -> None:
        """
        Spawn a vehicle in CARLA corresponding to a SUMO vehicle.

        Args:
            sumo_vehicle_id: SUMO vehicle ID
            position: (x, y, z) position from SUMO
        """
        if sumo_vehicle_id in self.vehicle_actors:
            return

        assert self.blueprint_library is not None, BLUEPRINT_ERROR_MSG
        assert self.world is not None, CARLA_WORLD_ERROR_MSG

        vehicle_blueprints = self.blueprint_library.filter("vehicle.*")
        vehicle_bp = random.choice(
            [
                bp
                for bp in vehicle_blueprints
                if int(bp.get_attribute("number_of_wheels")) == 4
            ]
        )

        if vehicle_bp.has_attribute("color"):
            color = random.choice(
                vehicle_bp.get_attribute("color").recommended_values
            )
            vehicle_bp.set_attribute("color", color)

        carla_x = position[0] + self.offset_x
        carla_y = -position[1] + self.offset_y
        carla_z = 0.5

        transform = carla.Transform(
            carla.Location(x=carla_x, y=carla_y, z=carla_z)
        )

        actor = self.world.try_spawn_actor(vehicle_bp, transform)

        if actor:
            self.vehicle_actors[sumo_vehicle_id] = actor
            print(
                f"✓ Spawned vehicle {sumo_vehicle_id} at CARLA "
                f"({carla_x:.1f}, {carla_y:.1f})"
            )

    def update_vehicle_in_carla(
        self,
        sumo_vehicle_id: str,
        position: tuple[float, float, float],
        angle: float,
    ) -> None:
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

        carla_x = position[0] + self.offset_x
        carla_y = -position[1] + self.offset_y
        carla_z = 0.5

        carla_yaw = 90.0 - angle

        transform = carla.Transform(
            carla.Location(x=carla_x, y=carla_y, z=carla_z),
            carla.Rotation(yaw=carla_yaw),
        )

        actor.set_transform(transform)

    def remove_vehicle_from_carla(self, sumo_vehicle_id: str) -> None:
        """Remove a vehicle from CARLA."""
        if sumo_vehicle_id in self.vehicle_actors:
            actor = self.vehicle_actors[sumo_vehicle_id]
            actor.destroy()
            del self.vehicle_actors[sumo_vehicle_id]

    def synchronize_vehicles(self) -> None:
        """Synchronize all vehicles between SUMO and CARLA."""
        sumo_vehicles = set(traci.vehicle.getIDList())
        carla_vehicles = set(self.vehicle_actors.keys())

        for vehicle_id in carla_vehicles - sumo_vehicles:
            self.remove_vehicle_from_carla(vehicle_id)

        for vehicle_id in sumo_vehicles:
            pos_2d = traci.vehicle.getPosition(vehicle_id)
            position = (pos_2d[0], pos_2d[1], 0.0)

            angle_raw = traci.vehicle.getAngle(vehicle_id)
            if isinstance(angle_raw, tuple):
                angle = angle_raw[0] if len(angle_raw) > 0 else 0.0
            else:
                angle = float(angle_raw)

            self.update_vehicle_in_carla(vehicle_id, position, angle)

    def adjust_traffic_light_height(self, z_offset: float = -1.5) -> None:
        """
        Adjust the height of all traffic lights in the world.

        Args:
            z_offset: Vertical offset to apply
                      (negative = lower, positive = higher)
        """
        assert self.world is not None, CARLA_WORLD_ERROR_MSG

        traffic_lights = self.world.get_actors().filter(
            "traffic.traffic_light*"
        )

        if not traffic_lights:
            print("  ⚠ No traffic lights found in the world")
            return

        adjusted_count = 0
        for tl in traffic_lights:
            current_transform = tl.get_transform()
            new_transform = carla.Transform(
                carla.Location(
                    x=current_transform.location.x,
                    y=current_transform.location.y,
                    z=current_transform.location.z + z_offset,
                ),
                current_transform.rotation,
            )
            tl.set_transform(new_transform)
            adjusted_count += 1

        if adjusted_count > 0:
            print(
                f"  ✓ Adjusted {adjusted_count} traffic light(s) by {z_offset}m"
            )
        else:
            print("  ⚠ Could not adjust traffic lights")

    def set_initial_camera_view(self) -> None:
        """Position camera at start to view the simulation area."""
        assert self.spectator is not None, SPECTATOR_ERROR_MSG

        camera_transform = carla.Transform(
            carla.Location(x=-50, y=0, z=80),
            carla.Rotation(pitch=-45, yaw=0),
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

    def update_spectator_camera(self) -> None:
        """Move spectator camera to follow the action."""
        if not self.vehicle_actors:
            return

        assert self.spectator is not None, SPECTATOR_ERROR_MSG

        positions = []
        for actor in self.vehicle_actors.values():
            loc = actor.get_location()
            positions.append((loc.x, loc.y))

        if positions:
            avg_x = sum(p[0] for p in positions) / len(positions)
            avg_y = sum(p[1] for p in positions) / len(positions)

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

    def _run_simulation_step(self, step: int) -> bool:
        """Run a single simulation step."""
        traci.simulationStep()
        self.synchronize_vehicles()

        if self.auto_camera and step % 5 == 0:
            self.update_spectator_camera()

        assert self.world is not None, CARLA_WORLD_ERROR_MSG
        self.world.tick()
        return True

    def _print_progress(self, step: int, start_time: float) -> None:
        """Print simulation progress indicator."""
        if step % 20 == 0:
            elapsed = time.time() - start_time
            num_vehicles = len(traci.vehicle.getIDList())
            print(
                f"Step {step:5d} | Elapsed: {elapsed:>6.1f}s | "
                f"Vehicles: {num_vehicles:3d}",
                end="\r",
            )

    def _get_intersection_observation(self, tls_id: str) -> np.ndarray:
        """
        Extract per-intersection observation features.

        Args:
            tls_id: Traffic light ID

        Returns:
            numpy array with intersection-specific features
        """
        features = []

        try:
            phase = traci.trafficlight.getPhase(tls_id)
            if isinstance(phase, tuple):
                phase = phase[0] if len(phase) > 0 else 0
            features.append(float(phase))

            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            for lane_id in controlled_lanes[:4]:
                occupancy = traci.lane.getLastStepOccupancy(lane_id)
                if isinstance(occupancy, tuple):
                    occupancy = occupancy[0] if len(occupancy) > 0 else 0.0
                features.append(float(occupancy))

                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                if isinstance(vehicle_count, tuple):
                    vehicle_count = (
                        vehicle_count[0] if len(vehicle_count) > 0 else 0
                    )
                features.append(float(vehicle_count))

        except traci.TraCIException:
            pass

        return np.array(features, dtype=np.float32)

    def _get_neighbor_intersections(self, tls_id: str) -> list[str]:
        """
        Identify adjacent intersections for a given traffic light.

        Uses spatial proximity based on controlled lanes to find neighbors.

        Args:
            tls_id: Traffic light ID

        Returns:
            List of neighbor traffic light IDs
        """
        all_tls = traci.trafficlight.getIDList()
        neighbors = []

        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            controlled_edges = set()
            for lane_id in controlled_lanes:
                edge_id = traci.lane.getEdgeID(lane_id)
                controlled_edges.add(edge_id)

            for other_tls in all_tls:
                if other_tls == tls_id:
                    continue

                other_lanes = traci.trafficlight.getControlledLanes(other_tls)
                other_edges = {
                    traci.lane.getEdgeID(lane) for lane in other_lanes
                }

                if controlled_edges.intersection(other_edges):
                    neighbors.append(other_tls)

        except traci.TraCIException:
            pass

        return neighbors

    def _get_intersection_reward(self, tls_id: str) -> float:
        """
        Calculate per-intersection reward.

        Args:
            tls_id: Traffic light ID

        Returns:
            Reward value for this intersection
        """
        reward = 0.0

        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            total_waiting = 0.0
            vehicle_count = 0

            for lane_id in controlled_lanes:
                vehicles = traci.lane.getLastStepVehicleIDs(lane_id)
                vehicle_count += len(vehicles)

                for veh_id in vehicles:
                    waiting = traci.vehicle.getWaitingTime(veh_id)
                    if isinstance(waiting, tuple):
                        waiting = waiting[0] if len(waiting) > 0 else 0.0
                    total_waiting += float(waiting)

            reward = -total_waiting * 0.01 + vehicle_count * 0.1

        except traci.TraCIException:
            pass

        return float(reward)

    def run_cosimulation(self, duration: int | None = None) -> None:
        """
        Run the co-simulation.

        Args:
            duration: Simulation duration in seconds (None = infinite)
        """
        print("\n" + "=" * 60)
        print("Starting CARLA-SUMO Co-Simulation")
        print("=" * 60)

        self.connect_carla()
        self.setup_sumo()
        self.set_initial_camera_view()

        self._initialized = True
        self._step_count = 0
        self._start_time = time.time()

        self._tls_ids = list(traci.trafficlight.getIDList())

        print("\nSimulation parameters:")
        print(f"  Step length: {self.step_length}s")
        print(f"  Duration: {duration if duration else 'infinite'}")
        print(f"  TLS manager: {self.tls_manager}")
        print(f"  Sync vehicle lights: {self.sync_vehicle_lights}")

        print("\n▶ Simulation running... (Press Ctrl+C to stop)\n")

        while True:
            should_stop, stop_reason = self._check_simulation_done(
                duration, self._start_time
            )
            if should_stop:
                print(stop_reason)
                break

            self._run_simulation_step(self._step_count)
            self._step_count += 1
            self._print_progress(self._step_count, self._start_time)

        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        print("\nCleaning up...")

        num_vehicles = len(self.vehicle_actors)
        vehicle_ids = list(self.vehicle_actors.keys())
        for vehicle_id in vehicle_ids:
            self.remove_vehicle_from_carla(vehicle_id)

        if num_vehicles > 0:
            print(f"✓ Destroyed {num_vehicles} CARLA vehicles")
        else:
            print("⚠ No vehicles were spawned in CARLA")

        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            print("✓ CARLA settings restored")

        traci.close()
        print("✓ SUMO closed")

        self._initialized = False
        self._step_count = 0
        self._start_time = None
        self._tls_ids = []

        print("Cleanup complete.\n")


class CarlaSumoGymEnv(gym.Env, CarlaSumoSync):
    """
    Gymnasium environment wrapper for CARLA-SUMO co-simulation.

    Extends CarlaSumoSync to provide a Gymnasium-compatible interface
    for reinforcement learning. Handles observation spaces, action spaces,
    rewards, and episode termination.
    """

    def __init__(
        self,
        sumo_cfg_file: str,
        enable_rl_control: bool = False,
        observation_config: dict[str, Any] | None = None,
        action_config: dict[str, Any] | None = None,
        video_config: dict[str, Any] | None = None,
        device: str | None = None,
        **kwargs,
    ):
        """
        Initialize CARLA-SUMO Gymnasium environment.

        Args:
            sumo_cfg_file: Path to SUMO configuration file
            enable_rl_control: Enable RL action space for traffic light control
            observation_config: Configuration dict for observation space
            action_config: Configuration dict for action space
            video_config: Configuration dict for video recording
            device: Compute device ('cuda', 'npu', 'cpu', or None for auto)
            **kwargs: Additional parameters passed to CarlaSumoSync
        """
        CarlaSumoSync.__init__(self, sumo_cfg_file, **kwargs)

        self.enable_rl_control = enable_rl_control
        self.observation_config = observation_config or {}
        self.action_config = action_config or {}
        self.video_config = video_config or {}
        self.device = self._detect_device(device)

        self._tls_controller: Any | None = None

        self._initialize_spaces()
        self._initialize_video_recording()

    def _detect_device(self, device: str | None) -> str:
        """
        Detect available compute device (GPU/NPU/CPU).

        Args:
            device: Preferred device or None for auto-detect

        Returns:
            Device string ('cuda', 'npu', or 'cpu')
        """
        if device:
            return device.lower()

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        try:
            import intel_extension_for_pytorch as ipex

            if hasattr(ipex, "xpu") and ipex.xpu.is_available():
                return "npu"
        except ImportError:
            pass

        return "cpu"

    def _initialize_spaces(self) -> None:
        """Initialize action and observation spaces based on configuration."""
        assert spaces is not None, GYMNASIUM_SPACES_ERROR_MSG

        if self.enable_rl_control:
            num_phases = self.action_config.get("num_phases") or 4
            num_tls = self.action_config.get("num_traffic_lights") or 1

            if num_tls == 1:
                self.action_space = spaces.Discrete(num_phases)
            else:
                self.action_space = spaces.MultiDiscrete([num_phases] * num_tls)
        else:
            self.action_space = spaces.Discrete(1)

        obs_shape = self.observation_config.get("shape", (10,))
        obs_low = self.observation_config.get("low", 0.0)
        obs_high = self.observation_config.get("high", np.inf)

        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=np.float32,
        )

    def _initialize_video_recording(self) -> None:
        """Initialize video recording if enabled."""
        self._video_enabled = self.video_config.get("enabled", False)
        self._video_output_path = self.video_config.get("output_path", "videos")
        self._video_format = self.video_config.get("format", "mp4")
        self._video_writer = None
        self._video_frames = []

        if self._video_enabled:
            Path(self._video_output_path).mkdir(parents=True, exist_ok=True)

    def _capture_frame(self) -> None:
        """Capture a frame from CARLA for video recording."""
        if not self._video_enabled or self.world is None:
            return

        assert self.spectator is not None, SPECTATOR_ERROR_MSG

        # Get camera view (simplified - in practice would use CARLA camera)
        # For now, we'll use a placeholder that can be enhanced
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self._video_frames.append(frame)

    def _save_video(self, episode: int) -> None:
        """
        Save recorded frames as video file.

        Note: cv2 (OpenCV) is imported inline here because it's an optional
        dependency for video recording. If OpenCV is not installed, video
        recording will be silently skipped.
        """
        if not self._video_enabled or not self._video_frames:
            return

        try:
            import cv2
        except ImportError:
            return

        output_file = Path(self._video_output_path) / (
            f"episode_{episode}.{self._video_format}"
        )

        if not self._video_frames:
            return

        height, width = self._video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_file), fourcc, 20.0, (width, height))

        for frame in self._video_frames:
            out.write(frame)

        out.release()
        self._video_frames = []
        print(f"✓ Saved video: {output_file}")

    def _get_traffic_light_phases(self, obs_features: list[float]) -> None:
        """Extract traffic light phases and append to observation features."""
        if not self._tls_ids:
            return

        for tls_id in self._tls_ids[:5]:
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
            for lane_id in lane_ids[:5]:
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
        """Normalize observation features to match observation space."""
        has_shape = hasattr(self.observation_space, "shape")
        if has_shape and self.observation_space.shape is not None:
            target_size = self.observation_space.shape[0]
        else:
            target_size = 10

        obs_features = obs_features + [0.0] * (target_size - len(obs_features))

        obs_array = np.array(obs_features, dtype=np.float32)

        if self.device != "cpu":
            obs_array = self._move_observation_to_device(obs_array)

        return obs_array

    def _move_observation_to_device(self, obs: np.ndarray) -> np.ndarray:
        """
        Move observation to compute device (GPU/NPU) if available.

        Note: torch is imported inline here because it's an optional
        dependency. If PyTorch is not installed, this function returns
        the observation unchanged.

        Args:
            obs: Observation array

        Returns:
            Observation array (possibly moved to device and back)
        """
        try:
            import torch

            obs_tensor = torch.from_numpy(obs)
            if self.device == "cuda":
                obs_tensor = obs_tensor.cuda()
            elif self.device == "npu":
                obs_tensor = obs_tensor.to("xpu")
            return obs_tensor.cpu().numpy()
        except ImportError:
            return obs

    def _get_observation(self) -> np.ndarray:
        """
        Extract observation from current simulation state.

        Returns:
            numpy array with observation values
        """
        obs_features = []

        sim_time = _normalize_traci_value(traci.simulation.getTime())
        obs_features.append(float(sim_time))
        obs_features.append(float(self._step_count))

        vehicle_ids = traci.vehicle.getIDList()
        num_vehicles = len(vehicle_ids)
        obs_features.append(float(num_vehicles))

        self._get_traffic_light_phases(obs_features)
        self._get_lane_metrics(obs_features)

        return self._normalize_observation_size(obs_features)

    def _calculate_waiting_time_reward(self) -> float:
        """Calculate negative reward based on total waiting time."""
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
        return -total_waiting * 0.01

    def _calculate_throughput_reward(self) -> float:
        """Calculate positive reward based on vehicle throughput."""
        vehicle_ids = traci.vehicle.getIDList()
        return len(vehicle_ids) * 0.1

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on traffic metrics.

        Returns:
            Reward value (default: 0.0)
        """
        if not self.enable_rl_control:
            return 0.0

        reward = 0.0

        if self.observation_config.get("reward_waiting_time", False):
            reward += self._calculate_waiting_time_reward()

        if self.observation_config.get("reward_throughput", False):
            reward += self._calculate_throughput_reward()

        return float(reward)

    def _apply_action(self, action: Any) -> None:
        """
        Apply traffic light control action.

        Args:
            action: Action from action space
        """
        if not self.enable_rl_control or not self._tls_ids:
            return

        assert spaces is not None, GYMNASIUM_SPACES_ERROR_MSG

        if isinstance(self.action_space, spaces.Discrete):
            if len(self._tls_ids) > 0:
                tls_id = self._tls_ids[0]
                phase = int(action)
                traci.trafficlight.setPhase(tls_id, phase)
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            action_array = np.asarray(action)
            for i, tls_id in enumerate(self._tls_ids):
                if i < len(action_array):
                    phase = int(action_array[i])
                    traci.trafficlight.setPhase(tls_id, phase)

    def _get_num_phases_from_tls(self, tls_id: str) -> int:
        """Get number of phases for a traffic light."""
        try:
            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(
                tls_id
            )
            if program and len(program) > 0:
                return len(program[0].phases)
        except traci.TraCIException:
            pass
        return 4

    def _update_action_space_for_multiple_tls(self) -> None:
        """Update action space when multiple traffic lights are present."""
        if not self.enable_rl_control or not self._tls_ids:
            return

        num_tls = len(self._tls_ids)
        if num_tls <= 1:
            return

        assert spaces is not None, GYMNASIUM_SPACES_ERROR_MSG
        num_phases = self._get_num_phases_from_tls(self._tls_ids[0])
        self.action_space = spaces.MultiDiscrete([num_phases] * num_tls)

    def _initialize_simulation(self) -> None:
        """Initialize CARLA and SUMO connections."""
        if self._initialized:
            self.cleanup()

        self.connect_carla()
        self.setup_sumo()
        self.set_initial_camera_view()

    def _collect_traffic_light_ids(self) -> None:
        """Collect and update traffic light IDs from SUMO."""
        self._tls_ids = list(traci.trafficlight.getIDList())
        self._update_action_space_for_multiple_tls()

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
        self._initialize_simulation()
        self._collect_traffic_light_ids()

        self._initialized = True
        self._step_count = 0
        self._start_time = time.time()

        observation = self._get_observation()
        info = {
            "step": 0,
            "time": _normalize_traci_value(traci.simulation.getTime()),
            "num_vehicles": len(traci.vehicle.getIDList()),
            "tls_ids": self._tls_ids,
        }

        return observation, info

    def step(  # type: ignore[override]
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert (
            self._initialized
        ), "Environment not initialized. Call reset() first."

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
            Dictionary with step results
        """
        assert self._initialized, (
            "Simulation not initialized. "
            "Call reset() or run_cosimulation() first."
        )

        if action is not None and self.enable_rl_control:
            self._apply_action(action)

        for _ in range(num_steps):
            self._run_simulation_step(self._step_count)
            self._step_count += 1
            if self._video_enabled:
                self._capture_frame()

        observation = self._get_observation()
        reward = self._calculate_reward()

        done = False
        truncated = False

        try:
            min_expected = traci.simulation.getMinExpectedNumber()
            if isinstance(min_expected, tuple):
                min_expected = min_expected[0] if len(min_expected) > 0 else 0
            if min_expected <= 0:
                done = True
        except traci.TraCIException:
            pass

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

    def render(self) -> None:
        """
        Render the environment (delegates to camera logic).

        For CARLA-SUMO, rendering is handled by the CARLA viewer.
        This method can be used to update camera position if needed.
        """
        if self.auto_camera and self._step_count % 5 == 0:
            self.update_spectator_camera()

    def close(self) -> None:
        """
        Close the environment and clean up resources.

        This method is called by Gymnasium and delegates to cleanup().
        """
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources including video recording."""
        CarlaSumoSync.cleanup(self)

        if self._video_enabled and self._video_frames:
            self._save_video(self._step_count)


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
