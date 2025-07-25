"""
Autonomous Vehicle Simulation Data Analysis Package
"""

__version__ = "1.0.0"
__author__ = "alaminxtration"
__email__ = "your-email@example.com"
__description__ = "A comprehensive simulation and analysis framework for autonomous vehicle sensor data"

# Import main components for easy access
from .simulation.vehicle_simulation import VehicleSimulation
from .processing.sensor_data_processor import RiskAssessment
from .storage.storage_manager import StorageManager

__all__ = [
    "VehicleSimulation",
    "RiskAssessment", 
    "StorageManager",
]
