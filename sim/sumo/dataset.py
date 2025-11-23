"""
Dataset generation with realism/noise modes for SUMO simulations.

This module provides functionality to simulate real-world data collection
by introducing sensor noise, errors, and inconsistencies into SUMO TraCI data.
"""

import os
import random
import numpy as np
from enum import Enum


class RealismLevel(Enum):
    """Realism levels for sensor noise simulation."""

    NONE = "none"  # No noise - perfect sensors
    LOW = "low"  # Minor jitter and occasional missing data
    MED = "med"  # Moderate noise, some sensor failures, errors
    HIGH = "high"  # High noise, frequent failures, significant errors


class RealismMode:
    """
    Configures realism/noise parameters for sensor data collection.

    Simulates real-world sensor imperfections:
    - Random jitters in measurements
    - Sensor failures (missing data)
    - Errors and inconsistencies
    - Outliers and anomalous readings
    """

    def __init__(self, level: RealismLevel = RealismLevel.NONE):
        """
        Initialize realism mode.

        Args:
            level: Realism level (none/low/med/high)
        """
        self.level = level
        self._configure_parameters()

    def _configure_parameters(self):
        """Configure noise parameters based on realism level."""
        if self.level == RealismLevel.NONE:
            self.jitter_std = 0.0
            self.sensor_failure_rate = 0.0
            self.error_rate = 0.0
            self.outlier_rate = 0.0
            self.max_jitter_percent = 0.0
        elif self.level == RealismLevel.LOW:
            self.jitter_std = 0.02  # 2% standard deviation
            self.sensor_failure_rate = 0.01  # 1% of readings fail
            self.error_rate = 0.005  # 0.5% errors
            self.outlier_rate = 0.001  # 0.1% outliers
            self.max_jitter_percent = 0.05  # Max 5% jitter
        elif self.level == RealismLevel.MED:
            self.jitter_std = 0.05  # 5% standard deviation
            self.sensor_failure_rate = 0.05  # 5% of readings fail
            self.error_rate = 0.02  # 2% errors
            self.outlier_rate = 0.005  # 0.5% outliers
            self.max_jitter_percent = 0.10  # Max 10% jitter
        elif self.level == RealismLevel.HIGH:
            self.jitter_std = 0.10  # 10% standard deviation
            self.sensor_failure_rate = 0.15  # 15% of readings fail
            self.error_rate = 0.05  # 5% errors
            self.outlier_rate = 0.02  # 2% outliers
            self.max_jitter_percent = 0.20  # Max 20% jitter

        # Track failed sensors (persistent failures)
        self.failed_sensors = set()
        self._init_failed_sensors()

    def _init_failed_sensors(self):
        """Initialize some sensors as permanently failed based on realism
        level.
        """
        if self.level == RealismLevel.NONE:
            return

        # This will be populated as sensors are encountered
        # For now, we'll mark sensors as failed on-the-fly

    def _should_fail_sensor(self, sensor_id: str) -> bool:
        """
        Check if a sensor should fail (either permanently or temporarily).

        Args:
            sensor_id: Unique identifier for the sensor
                       (e.g., lane_id, vehicle_id)

        Returns:
            True if sensor reading should fail/return None
        """
        # Check if sensor is permanently failed
        if sensor_id in self.failed_sensors:
            return True

        # Random chance of permanent failure (first encounter)
        if random.random() < {
            RealismLevel.LOW: 0.001,
            RealismLevel.MED: 0.01,
            RealismLevel.HIGH: 0.02,
        }.get(self.level, 0.0):
            self.failed_sensors.add(sensor_id)
            return True

        # Temporary sensor failure
        if random.random() < self.sensor_failure_rate:
            return True

        return False

    def _apply_jitter(
        self,
        value: float,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> float:
        """
        Apply random jitter to a measurement value.

        Args:
            value: Original measurement value
            min_value: Minimum allowed value (None = no limit)
            max_value: Maximum allowed value (None = no limit)

        Returns:
            Value with jitter applied
        """
        if abs(self.jitter_std) < 1e-9 or abs(value) < 1e-9:
            return value

        # Calculate jitter as percentage of value
        rng = np.random.default_rng(
            seed=int.from_bytes(os.urandom(16), "little")
        )
        jitter_percent = rng.normal(0, self.jitter_std)
        jitter_percent = np.clip(
            jitter_percent, -self.max_jitter_percent, self.max_jitter_percent
        )

        noisy_value = value * (1 + jitter_percent)

        # Clamp to valid range if provided
        if min_value is not None:
            noisy_value = max(noisy_value, min_value)
        if max_value is not None:
            noisy_value = min(noisy_value, max_value)

        return noisy_value

    def _apply_error(self, value: float) -> float:
        """
        Apply systematic error or inconsistency to a measurement.

        Args:
            value: Original measurement value

        Returns:
            Value with error applied (or original if no error)
        """
        if random.random() < self.error_rate:
            # Apply error: scale by wrong factor or add offset
            error_type = random.choice(["scale", "offset", "negate"])
            if error_type == "scale":
                # Wrong scaling factor
                scale = random.uniform(0.5, 1.5)
                return value * scale
            elif error_type == "offset":
                # Add/subtract offset
                offset = value * random.uniform(-0.3, 0.3)
                return value + offset
            elif error_type == "negate":
                # Sometimes readings are inverted
                return -value if abs(value) > 1e-6 else value

        return value

    def _apply_outlier(self, value: float) -> float:
        """
        Apply outlier/anomalous reading.

        Args:
            value: Original measurement value

        Returns:
            Value with outlier applied (or original if no outlier)
        """
        if random.random() < self.outlier_rate:
            # Generate outlier: 2x to 10x the value or near-zero
            outlier_type = random.choice(["high", "low", "zero"])
            if outlier_type == "high":
                return value * random.uniform(2.0, 10.0)
            elif outlier_type == "low":
                return value * random.uniform(0.01, 0.1)
            elif outlier_type == "zero":
                return 0.0

        return value

    def apply_noise_to_value(
        self,
        value: float,
        sensor_id: str,
        min_value: float | None = None,
        max_value: float | None = None,
        default_value: float | None = None,
    ) -> float | None:
        """
        Apply all noise effects to a sensor reading.

        Args:
            value: Original sensor reading
            sensor_id: Unique sensor identifier
            min_value: Minimum valid value
            max_value: Maximum valid value
            default_value: Value to return if sensor fails (default: None)

        Returns:
            Noisy measurement value, or None/default_value if sensor failed
        """
        # Check for sensor failure first
        if self._should_fail_sensor(sensor_id):
            return default_value

        # Apply effects in order: error -> outlier -> jitter
        noisy_value = value
        noisy_value = self._apply_error(noisy_value)
        noisy_value = self._apply_outlier(noisy_value)
        noisy_value = self._apply_jitter(noisy_value, min_value, max_value)

        return noisy_value

    def apply_noise_to_position(
        self, position: tuple[float, float], sensor_id: str
    ) -> tuple[float, float] | None:
        """
        Apply noise to position coordinates.

        Args:
            position: (x, y) position tuple
            sensor_id: Unique sensor identifier

        Returns:
            Noisy position tuple, or None if sensor failed
        """
        if self._should_fail_sensor(sensor_id):
            return None

        x, y = position
        noisy_x = self.apply_noise_to_value(
            x, f"{sensor_id}_x", default_value=x
        )
        noisy_y = self.apply_noise_to_value(
            y, f"{sensor_id}_y", default_value=y
        )

        if noisy_x is None or noisy_y is None:
            return None

        return (noisy_x, noisy_y)


def get_realism_level(level_str: str) -> RealismLevel:
    """
    Parse realism level string to enum.

    Args:
        level_str: String representation ('none', 'low', 'med', 'high')

    Returns:
        RealismLevel enum value

    Raises:
        ValueError: If level_str is invalid
    """
    level_str = level_str.lower().strip()
    for level in RealismLevel:
        if level.value == level_str:
            return level
    raise ValueError(
        f"Invalid realism level: {level_str}. "
        f"Must be one of: {', '.join([level.value for level in RealismLevel])}"
    )
