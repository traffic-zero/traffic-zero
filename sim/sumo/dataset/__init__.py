"""
Dataset generation module with realism/noise modes for SUMO simulations.

This module provides the CLI interface for dataset generation.
The actual RealismMode implementation is in the parent dataset.py module.
"""

# Import from parent module (sim/sumo/dataset.py) using importlib
# to avoid circular imports
import importlib.util
from pathlib import Path

# Load the parent dataset.py module directly
_dataset_module_path = Path(__file__).parent.parent / "dataset.py"
spec = importlib.util.spec_from_file_location(
    "sumo_dataset", _dataset_module_path
)
_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_dataset_module)

# Re-export the classes/functions
RealismLevel = _dataset_module.RealismLevel
RealismMode = _dataset_module.RealismMode
get_realism_level = _dataset_module.get_realism_level

__all__ = ["RealismLevel", "RealismMode", "get_realism_level"]
