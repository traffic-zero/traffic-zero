"""
CARLA Recorder-based video generation module.

Uses CARLA's Recorder API to record simulations and generate videos.
This avoids camera sensor callbacks which cause issues in synchronous mode.
"""

import os
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Any

try:
    import carla
except ImportError:
    carla = None


class RecorderVideoGenerator:
    """
    Generate videos using CARLA Recorder API.
    
    Records simulation state, then replays and captures frames to create video.
    """
    
    def __init__(
        self,
        client: Any,
        output_dir: str = "./data",
        video_width: int = 854,
        video_height: int = 480,
        video_fps: int = 30,
        camera_location: Optional[tuple] = None,
        camera_rotation: Optional[tuple] = None,
    ):
        """
        Initialize recorder-based video generator.
        
        Args:
            client: CARLA client object
            output_dir: Directory to save video files
            video_width: Video width in pixels
            video_height: Video height in pixels
            video_fps: Video frame rate
            camera_location: Camera location (x, y, z) tuple, defaults to overhead
            camera_rotation: Camera rotation (pitch, yaw, roll) tuple, defaults to looking down
        """
        self.client = client
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
        self.recording_file: Optional[Path] = None
        self.is_recording = False
        self.video_file_path: Optional[Path] = None
        
    def start_recording(
        self,
        experiment_name: Optional[str] = None,
        video_filename: Optional[str] = None,
    ) -> bool:
        """
        Start CARLA recorder for simulation recording.
        
        Args:
            experiment_name: Optional experiment name for filename
            video_filename: Optional custom video filename
            
        Returns:
            True if recording started successfully, False otherwise
        """
        if self.is_recording:
            return False
            
        if self.client is None or carla is None:
            print("⚠ CARLA client not available for recording")
            return False
            
        try:
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate recording filename
            if video_filename:
                # Use video filename but with .log extension for recording
                recording_filename = video_filename.replace(".mp4", ".log")
            elif experiment_name:
                recording_filename = f"recording_{experiment_name}.log"
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                recording_filename = f"recording_{timestamp}.log"
            
            self.recording_file = self.output_dir / recording_filename
            
            # Generate video filename
            if video_filename:
                self.video_file_path = self.output_dir / video_filename
            elif experiment_name:
                self.video_file_path = self.output_dir / f"simulation_{experiment_name}.mp4"
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.video_file_path = self.output_dir / f"simulation_{timestamp}.mp4"
            
            # Start CARLA recorder
            self.client.start_recorder(str(self.recording_file))
            self.is_recording = True
            
            print(f"✓ Started CARLA recorder: {self.recording_file.name}")
            print(f"  Video will be generated after recording: {self.video_file_path.name}")
            
            return True
            
        except Exception as e:
            print(f"⚠ Failed to start CARLA recorder: {e}")
            return False
    
    def stop_recording(self) -> Optional[Path]:
        """
        Stop CARLA recorder.
        
        Returns:
            Path to recording file if successful, None otherwise
        """
        if not self.is_recording:
            return None
            
        if self.client is None or carla is None:
            return None
            
        try:
            self.client.stop_recorder()
            self.is_recording = False
            
            print(f"✓ Stopped CARLA recorder")
            
            if self.recording_file and self.recording_file.exists():
                return self.recording_file
            return None
            
        except Exception as e:
            print(f"⚠ Failed to stop CARLA recorder: {e}")
            return None
    
    def generate_video_from_recording(
        self,
        world: Any,
        recording_file: Optional[Path] = None,
        duration: Optional[float] = None,
    ) -> Optional[Path]:
        """
        Replay recording and generate video file.
        
        Args:
            world: CARLA world object
            recording_file: Path to recording file (uses self.recording_file if None)
            duration: Maximum duration to record (None = entire recording)
            
        Returns:
            Path to generated video file, or None if failed
        """
        if recording_file is None:
            recording_file = self.recording_file
            
        if recording_file is None or not recording_file.exists():
            print("⚠ Recording file not found")
            return None
            
        if world is None or carla is None:
            print("⚠ CARLA world not available for video generation")
            return None
            
        print(f"\n[INFO] Generating video from recording: {recording_file.name}")
        print("  This may take a few minutes...")
        
        try:
            # Create temporary camera for replay
            camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", str(self.video_width))
            camera_bp.set_attribute("image_size_y", str(self.video_height))
            camera_bp.set_attribute("fov", "90")
            
            # Position camera overhead
            camera_transform = carla.Transform(
                carla.Location(*self.camera_location),
                carla.Rotation(*self.camera_rotation)
            )
            
            # Spawn camera
            camera = world.spawn_actor(camera_bp, camera_transform)
            
            # Setup video writer
            codecs = ["mp4v", "XVID", "MJPG"]
            video_writer = None
            
            for codec_str in codecs:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_str)  # type: ignore[attr-defined]
                    writer = cv2.VideoWriter(
                        str(self.video_file_path),
                        fourcc,
                        float(self.video_fps),
                        (self.video_width, self.video_height),
                    )
                    if writer.isOpened():
                        video_writer = writer
                        break
                except Exception:
                    continue
            
            if video_writer is None or not video_writer.isOpened():
                print(f"⚠ Failed to create video writer: {self.video_file_path}")
                camera.destroy()
                return None
            
            # Replay recording and capture frames
            print("  Replaying recording and capturing frames...")
            
            # Read recording to get duration
            try:
                recording_info = self.client.show_recorder_file_info(str(recording_file), True)
                recording_duration = self._parse_recording_duration(recording_info)
            except Exception:
                # Default duration if we can't parse
                recording_duration = duration if duration else 60.0
            
            if duration:
                recording_duration = min(recording_duration, duration)
            
            # Switch to asynchronous mode for replay (better for camera callbacks)
            settings = world.get_settings()
            original_sync_mode = settings.synchronous_mode
            original_fixed_delta = settings.fixed_delta_seconds
            
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = 1.0 / self.video_fps
            world.apply_settings(settings)
            
            # Set up replay (replay from start to end, speed factor = 1.0)
            self.client.replay_file(str(recording_file), 0.0, recording_duration, 1.0)
            
            # Wait for replay to start
            time.sleep(0.5)
            
            # Capture frames during replay
            frame_count = 0
            frames_to_capture = int(recording_duration * self.video_fps)
            max_frames = frames_to_capture + (self.video_fps * 5)  # Extra buffer
            
            # Use callback to capture frames (thread-safe list)
            import threading
            image_queue = []
            queue_lock = threading.Lock()
            
            def capture_image(image):
                """Capture image from camera during replay."""
                try:
                    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                    array = np.reshape(array, (image.height, image.width, 4))
                    array = array[:, :, :3]  # Remove alpha
                    array = array[:, :, ::-1]  # RGB to BGR
                    with queue_lock:
                        image_queue.append(array)
                except Exception:
                    pass
            
            camera.listen(capture_image)
            
            # Replay and capture in async mode
            step = 0
            start_time = time.time()
            last_progress = 0
            
            while step < max_frames:
                # In async mode, we just wait for frames
                time.sleep(1.0 / self.video_fps)
                
                # Process captured images
                with queue_lock:
                    while image_queue and frame_count < frames_to_capture:
                        img = image_queue.pop(0)
                        # Resize if needed
                        if img.shape[1] != self.video_width or img.shape[0] != self.video_height:
                            img = cv2.resize(img, (self.video_width, self.video_height))
                        video_writer.write(img)
                        frame_count += 1
                
                step += 1
                
                # Progress update
                progress = int((frame_count / frames_to_capture) * 100)
                if progress >= last_progress + 10:
                    print(f"  Progress: {progress}% ({frame_count}/{frames_to_capture} frames)")
                    last_progress = progress
                
                # Stop if we have enough frames
                if frame_count >= frames_to_capture:
                    break
                
                # Timeout after reasonable time
                if time.time() - start_time > recording_duration * 2:
                    print(f"  Timeout after {recording_duration * 2}s - captured {frame_count} frames")
                    break
            
            # Restore original settings
            settings.synchronous_mode = original_sync_mode
            settings.fixed_delta_seconds = original_fixed_delta
            world.apply_settings(settings)
            
            # Stop camera
            camera.stop()
            camera.destroy()
            
            # Release video writer
            video_writer.release()
            
            # Clean up recording file
            try:
                recording_file.unlink()
            except Exception:
                pass
            
            if self.video_file_path and self.video_file_path.exists():
                file_size_mb = self.video_file_path.stat().st_size / (1024 * 1024)
                print(f"✓ Video generated: {self.video_file_path} ({file_size_mb:.2f} MB)")
                return self.video_file_path
            else:
                print("⚠ Video file was not created")
                return None
                
        except Exception as e:
            print(f"⚠ Error generating video from recording: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _parse_recording_duration(self, recording_info: str) -> float:
        """Parse duration from CARLA recorder file info."""
        try:
            # Recording info format varies, try to extract duration
            # Example: "Recording info: Duration: 60.5s ..."
            lines = recording_info.split('\n')
            for line in lines:
                if 'Duration' in line or 'duration' in line:
                    # Try to extract number
                    import re
                    match = re.search(r'[\d.]+', line)
                    if match:
                        return float(match.group())
        except Exception:
            pass
        return 60.0  # Default to 60 seconds
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.is_recording:
            self.stop_recording()

