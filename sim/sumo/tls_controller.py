"""
Traffic light controller module for SUMO.

Provides interface for dynamic traffic light control and logs all
traffic light instructions/changes during simulation.
"""

import traci
import pandas as pd
from pathlib import Path


def _normalize_traci_value(value: float | tuple | None) -> float:
    """Normalize traci return value to float (handles tuple returns)."""
    if value is None:
        return 0.0
    if isinstance(value, tuple):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)


def _normalize_traci_str(value: str | tuple | None) -> str:
    """Normalize traci return value to str (handles tuple returns)."""
    if value is None:
        return ""
    if isinstance(value, tuple):
        return str(value[0]) if len(value) > 0 else ""
    return str(value)


class TLSController:
    """
    Controller for traffic lights with comprehensive action logging.

    Tracks all traffic light control actions (phase changes, program changes,
    duration modifications) and exports log to CSV.
    """

    def __init__(self, output_dir: str | None = None):
        """
        Initialize traffic light controller.

        Args:
            output_dir: Directory to save action log CSV (None = no file output)
        """
        self.output_dir = Path(output_dir) if output_dir else None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Action log storage
        self.action_log = []
        self.current_step = 0

    def set_step(self, step: int):
        """Set current simulation step for logging."""
        self.current_step = step

    def set_phase(self, tls_id: str, phase: int) -> bool:
        """
        Set traffic light phase.

        Args:
            tls_id: Traffic light ID
            phase: Phase number to set

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current phase before change
            old_phase_raw = traci.trafficlight.getPhase(tls_id)
            if isinstance(old_phase_raw, tuple):
                old_phase = (
                    int(old_phase_raw[0]) if len(old_phase_raw) > 0 else 0
                )
            else:
                old_phase = int(old_phase_raw)
            current_time_raw = traci.simulation.getTime()
            current_time = _normalize_traci_value(current_time_raw)

            # Set new phase
            traci.trafficlight.setPhase(tls_id, phase)

            # Get controlled lanes
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                controlled_lanes_str = (
                    ",".join(controlled_lanes) if controlled_lanes else None
                )
            except traci.TraCIException:
                controlled_lanes_str = None

            # Log action
            self._log_action(
                tls_id=tls_id,
                action_type="phase_change",
                old_value=str(old_phase),
                new_value=str(phase),
                controlled_lanes=controlled_lanes_str,
                time=current_time,
            )

            return True
        except traci.TraCIException as e:
            print(f"Warning: Failed to set phase for {tls_id}: {e}")
            return False

    def set_program(self, tls_id: str, program: str) -> bool:
        """
        Set traffic light program.

        Args:
            tls_id: Traffic light ID
            program: Program name/ID to set

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current program before change
            old_program_raw = traci.trafficlight.getProgram(tls_id)
            old_program = _normalize_traci_str(old_program_raw)
            current_time_raw = traci.simulation.getTime()
            current_time = _normalize_traci_value(current_time_raw)

            # Set new program
            traci.trafficlight.setProgram(tls_id, program)

            # Get controlled lanes
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                controlled_lanes_str = (
                    ",".join(controlled_lanes) if controlled_lanes else None
                )
            except traci.TraCIException:
                controlled_lanes_str = None

            # Log action
            self._log_action(
                tls_id=tls_id,
                action_type="program_change",
                old_value=old_program,
                new_value=program,
                controlled_lanes=controlled_lanes_str,
                time=current_time,
            )

            return True
        except traci.TraCIException as e:
            print(f"Warning: Failed to set program for {tls_id}: {e}")
            return False

    def set_phase_duration(
        self, tls_id: str, phase: int, duration: float
    ) -> bool:
        """
        Set duration for a specific phase.

        Args:
            tls_id: Traffic light ID
            phase: Phase number
            duration: Duration in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get current phase duration before change
            try:
                old_duration = traci.trafficlight.getPhaseDuration(tls_id)
            except traci.TraCIException:
                old_duration = None

            current_time_raw = traci.simulation.getTime()
            current_time = _normalize_traci_value(current_time_raw)

            # Set new duration
            traci.trafficlight.setPhaseDuration(tls_id, duration)

            # Get controlled lanes
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
                controlled_lanes_str = (
                    ",".join(controlled_lanes) if controlled_lanes else None
                )
            except traci.TraCIException:
                controlled_lanes_str = None

            # Log action
            self._log_action(
                tls_id=tls_id,
                action_type="duration_change",
                old_value=(
                    str(old_duration) if old_duration is not None else "N/A"
                ),
                new_value=str(duration),
                controlled_lanes=controlled_lanes_str,
                time=current_time,
            )

            return True
        except traci.TraCIException as e:
            print(f"Warning: Failed to set phase duration for {tls_id}: {e}")
            return False

    def get_current_state(self, tls_id: str) -> dict | None:
        """
        Get current state of traffic light.

        Args:
            tls_id: Traffic light ID

        Returns:
            Dictionary with current state info, or None if error
        """
        try:
            phase = traci.trafficlight.getPhase(tls_id)
            phase_duration = traci.trafficlight.getPhaseDuration(tls_id)
            program = traci.trafficlight.getProgram(tls_id)
            state = traci.trafficlight.getRedYellowGreenState(tls_id)

            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            except traci.TraCIException:
                controlled_lanes = []

            return {
                "phase": phase,
                "phase_duration": phase_duration,
                "program": program,
                "state": state,
                "controlled_lanes": controlled_lanes,
            }
        except traci.TraCIException:
            return None

    def get_all_tls_ids(self) -> list[str]:
        """Get list of all traffic light IDs."""
        try:
            return list(traci.trafficlight.getIDList())
        except traci.TraCIException:
            return []

    def _log_action(
        self,
        tls_id: str,
        action_type: str,
        old_value: str,
        new_value: str,
        controlled_lanes: str | None,
        time: float,
    ):
        """Log a traffic light control action."""
        self.action_log.append(
            {
                "step": self.current_step,
                "time": time,
                "tls_id": tls_id,
                "action_type": action_type,
                "old_value": old_value,
                "new_value": new_value,
                "controlled_lanes": controlled_lanes,
            }
        )

    def get_action_log(self) -> pd.DataFrame:
        """
        Get action log as pandas DataFrame.

        Returns:
            DataFrame with action log entries
        """
        if not self.action_log:
            # Create empty DataFrame with correct columns
            return pd.DataFrame(
                {
                    "step": [],
                    "time": [],
                    "tls_id": [],
                    "action_type": [],
                    "old_value": [],
                    "new_value": [],
                    "controlled_lanes": [],
                }
            )
        return pd.DataFrame(self.action_log)

    def export_action_log(self):
        """Export action log to CSV file."""
        if not self.output_dir:
            return

        df = self.get_action_log()

        if not df.empty:
            filepath = self.output_dir / "traffic_light_actions.csv"
            df.to_csv(filepath, index=False)
            print(f">>> Exported traffic_light_actions.csv to {filepath}")
        else:
            print(">>> No traffic light actions to export")

    def reset(self):
        """Reset action log."""
        self.action_log = []
        self.current_step = 0
