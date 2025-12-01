"""
Video recording module for CARLA simulations.

This module provides functionality to record videos from CARLA simulations
using a fixed overhead camera view, saving MP4 videos for YOLO training.
"""

import os
import time
import queue
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Any
from datetime import datetime

try:
    import carla
except ImportError:
    carla = None


class VideoRecorder:
    """Handles video recording from CARLA camera sensor."""

    def __init__(
        self,
        world: Optional[Any] = None,
        output_dir: str = "./data",
        video_width: int = 854,
        video_height: int = 480,
        video_fps: int = 30,
        camera_location: Optional[tuple] = None,
        camera_rotation: Optional[tuple] = None,
    ):
        """
        Initialize video recorder.

        Args:
            world: CARLA world object
            output_dir: Directory to save video files
            video_width: Video width in pixels
            video_height: Video height in pixels
            video_fps: Video frame rate
            camera_location: Camera location (x, y, z) tuple, defaults to overhead
            camera_rotation: Camera rotation (pitch, yaw, roll) tuple, defaults to looking down
        """
        self.world = world
        self.output_dir = Path(output_dir)
        self.video_width = video_width
        self.video_height = video_height
        self.video_fps = video_fps

        # Camera position (overhead view by default)
        if camera_location is None:
            camera_location = (-50.0, 0.0, 80.0)
        if camera_rotation is None:
            camera_rotation = (-45.0, 0.0, 0.0)  # pitch=-45, yaw=0, roll=0

        self.camera_location = camera_location
        self.camera_rotation = camera_rotation

        # Recording state
        self.recording_camera: Optional[Any] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_file_path: Optional[Path] = None
        self.is_recording = False
        self.current_image: Optional[np.ndarray] = None
        self._image_ready = False
        # Queue for storing images from callback (thread-safe for async, works for sync too)
        self._image_queue: queue.Queue = queue.Queue(maxsize=2)
        self._frames_skipped = 0  # Skip first few frames to let camera stabilize
        self._frame_skip_count = 5  # Skip first 5 frames after camera setup
        # Image saving directory for temporary images
        self._temp_image_dir: Optional[Path] = None
        self._image_counter = 0
        self._use_save_to_disk = True  # Use CARLA's save_to_disk instead of callback

    def setup_camera(
        self,
        experiment_name: Optional[str] = None,
        world: Optional[Any] = None,
    ) -> bool:
        """
        Setup fixed overhead camera for recording.

        Args:
            experiment_name: Optional experiment name for video filename
            world: CARLA world object (if not set during init)

        Returns:
            True if camera setup successful, False otherwise
        """
        # Update world reference if provided
        if world is not None:
            self.world = world
            
        if self.world is None or carla is None:
            print("[WARNING] CARLA world not available for video recording")
            return False

        try:
            # Get blueprint library
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find("sensor.camera.rgb")

            # Configure camera attributes
            camera_bp.set_attribute("image_size_x", str(self.video_width))
            camera_bp.set_attribute("image_size_y", str(self.video_height))
            camera_bp.set_attribute("fov", "90")

            # Create camera transform (fixed overhead position)
            camera_transform = carla.Transform(
                carla.Location(
                    x=self.camera_location[0],
                    y=self.camera_location[1],
                    z=self.camera_location[2],
                ),
                carla.Rotation(
                    pitch=self.camera_rotation[0],
                    yaw=self.camera_rotation[1],
                    roll=self.camera_rotation[2],
                ),
            )

            # Spawn camera (use try_spawn_actor to avoid crashes)
            print("[DEBUG] Attempting to spawn camera actor...")
            try:
                self.recording_camera = self.world.try_spawn_actor(
                    camera_bp, camera_transform
                )
                if self.recording_camera is None:
                    print("⚠ Failed to spawn recording camera (try_spawn_actor returned None)")
                    return False
                print(f"[DEBUG] Camera actor spawned successfully: {self.recording_camera.id}")
            except Exception as e:
                print(f"⚠ Error spawning camera: {e}")
                import traceback
                traceback.print_exc()
                return False

            # CRITICAL: Camera callbacks in CARLA synchronous mode cause streaming client errors
            # We'll NOT register a callback - instead we'll try a different approach
            # by saving images after each world.tick() using direct image retrieval
            print(f"[WARNING] Camera created but callback disabled to prevent streaming errors")
            print(f"[WARNING] Video recording will use alternative method if available")
            return True  # Camera created, but we'll handle images differently

        except Exception as e:
            print(f"⚠ Failed to setup recording camera: {e}")
            # Don't print full traceback to avoid clutter
            return False

    def start_recording(
        self,
        experiment_name: Optional[str] = None,
        video_filename: Optional[str] = None,
    ) -> bool:
        """
        Start video recording.

        Args:
            experiment_name: Optional experiment name for directory/filename
            video_filename: Optional custom video filename

        Returns:
            True if recording started successfully, False otherwise
        """
        if self.is_recording:
            print("[WARNING] Recording already in progress")
            return False

        try:
            # Create output directory if it doesn't exist
            # Note: output_dir may already include experiment_name, so don't duplicate it
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate video filename
            if video_filename:
                filename = video_filename
            elif experiment_name:
                filename = f"simulation_{experiment_name}.mp4"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"simulation_{timestamp}.mp4"

            self.video_file_path = output_path / filename

            # Setup camera if not already done
            if self.recording_camera is None:
                if not self.setup_camera(experiment_name, world=None):
                    return False

            # CRITICAL: Camera callbacks cause CARLA streaming client errors and timeouts
            # in synchronous mode with SUMO co-simulation. Video recording is disabled.
            print("\n" + "="*70)
            print("⚠ VIDEO RECORDING DISABLED - CARLA COMPATIBILITY ISSUE")
            print("="*70)
            print("\nCamera sensors in CARLA synchronous mode with SUMO co-simulation")
            print("cause 'streaming client: connection failed' errors and timeouts.")
            print("\nThis is a known CARLA limitation.")
            print("\nRecommended Alternatives:")
            print("  1. Windows Game Bar (Win+G) - Built-in screen recording")
            print("  2. OBS Studio - Free screen recording software")
            print("  3. CARLA Recorder API - Record separately and replay")
            print("="*70 + "\n")
            
            # Clean up camera
            if self.recording_camera is not None:
                try:
                    self.recording_camera.destroy()
                except Exception:
                    pass
                self.recording_camera = None
            
            return False

        except Exception as e:
            print(f"⚠ Failed to start recording: {e}")
            return False

    def _save_image_to_disk(self, image: Any) -> None:
        """
        Save image directly to disk - DISABLED to prevent streaming client errors.
        """
        # Callback disabled - causes CARLA streaming client errors
        pass

    def capture_frame(self) -> bool:
        """
        Retrieve image from queue, process it, and write to video.
        
        This should be called AFTER world.tick() in synchronous mode.
        The image will be retrieved from the queue (placed there by callback).

        Returns:
            True if frame written successfully, False otherwise
        """
        if not self.is_recording or self.video_writer is None:
            return False

        # Skip first few frames to let camera stabilize
        if self._frames_skipped < self._frame_skip_count:
            # Drain queue but don't process
            try:
                self._image_queue.get_nowait()
                if self._frames_skipped < 3:
                    print(f"[DEBUG] Skipping frame {self._frames_skipped + 1}/{self._frame_skip_count}")
            except queue.Empty:
                pass
            self._frames_skipped += 1
            return False

        # Try to get image from queue (non-blocking)
        try:
            image = self._image_queue.get_nowait()
            if not hasattr(self, '_frames_captured'):
                self._frames_captured = 0
            self._frames_captured += 1
            if self._frames_captured <= 3:
                print(f"[DEBUG] Retrieved image from queue (frame {self._frames_captured})")
        except queue.Empty:
            # No image available yet - skip this frame
            if not hasattr(self, '_empty_queue_count'):
                self._empty_queue_count = 0
            self._empty_queue_count += 1
            if self._empty_queue_count <= 3:
                print(f"[DEBUG] No image in queue (count: {self._empty_queue_count})")
            return False

        try:
            # Process the image (convert CARLA format to OpenCV format)
            if self._frames_captured <= 3:
                print(f"[DEBUG] Processing image: {image.width}x{image.height}")
            
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            if self._frames_captured <= 3:
                print(f"[DEBUG] Image buffer size: {len(array)} bytes")
            
            array = np.reshape(
                array, (image.height, image.width, 4)
            )
            array = array[:, :, :3]  # Remove alpha channel
            array = array[:, :, ::-1]  # Convert RGB to BGR for OpenCV

            # Resize if needed
            if (
                array.shape[1] != self.video_width
                or array.shape[0] != self.video_height
            ):
                if self._frames_captured <= 3:
                    print(f"[DEBUG] Resizing from {array.shape[1]}x{array.shape[0]} to {self.video_width}x{self.video_height}")
                array = cv2.resize(
                    array, (self.video_width, self.video_height)
                )

            # Write frame to video
            if self._frames_captured <= 3:
                print(f"[DEBUG] Writing frame to video...")
            self.video_writer.write(array)
            if self._frames_captured <= 3:
                print(f"[DEBUG] Frame written successfully")
            return True

        except Exception as e:
            # Log frame processing errors
            if not hasattr(self, '_frame_errors'):
                self._frame_errors = 0
            self._frame_errors += 1
            if self._frame_errors <= 5:
                print(f"[DEBUG] Error processing frame: {e}")
                import traceback
                traceback.print_exc()
            return False

    def stop_recording(self) -> Optional[Path]:
        """
        Stop video recording and cleanup.

        Returns:
            Path to saved video file if successful, None otherwise
        """
        if not self.is_recording:
            return None

        self.is_recording = False

        # Close video writer
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        video_path = self.video_file_path
        self.video_file_path = None

        if video_path and video_path.exists():
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"✓ Video saved: {video_path} ({file_size_mb:.2f} MB)")
            return video_path
        else:
            print("⚠ Video file was not created")
            return None

    def cleanup(self) -> None:
        """Clean up camera and video writer resources."""
        if self.is_recording:
            try:
                self.stop_recording()
            except Exception as e:
                print(f"⚠ Error stopping recording: {e}")

        # Destroy camera actor (stop listening first to prevent callbacks)
        if self.recording_camera is not None:
            try:
                # Stop listening to prevent any more callbacks
                self.recording_camera.stop()
                # Small delay to ensure callback completes
                time.sleep(0.1)
            except Exception:
                pass  # Ignore errors when stopping
            
            try:
                self.recording_camera.destroy()
                print("✓ Recording camera destroyed")
            except Exception as e:
                # Don't print error if camera is already destroyed
                if "destroyed" not in str(e).lower() and "timeout" not in str(e).lower():
                    print(f"⚠ Error destroying camera: {e}")
            finally:
                self.recording_camera = None

        self.current_image = None
        self._image_ready = False
        # Clear image queue
        while not self._image_queue.empty():
            try:
                self._image_queue.get_nowait()
            except queue.Empty:
                break

