"""
Simplified Sensor Data Processor - No Kafka Version
Real-time sensor data processing for autonomous vehicles
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessedSensorData:
    """Processed sensor data structure"""
    vehicle_id: str
    timestamp: float
    location: Dict[str, float]
    velocity: Dict[str, float]
    calculated_speed: float
    risk_level: str
    anomalies: List[str]
    lidar_summary: Dict[str, Any]
    imu_summary: Dict[str, Any]
    processing_timestamp: float

class SensorDataProcessor:
    """No-Kafka version of sensor data processor"""
    
    def __init__(self):
        self.vehicle_histories = {}
        self.anomaly_thresholds = {
            'speed_limit': 30.0,  # m/s
            'acceleration_limit': 5.0,  # m/s²
            'lidar_min_points': 10,
            'gps_min_satellites': 4
        }
        logger.info("Sensor data processor initialized (no Kafka)")
    
    def validate_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Validate incoming sensor data"""
        required_fields = [
            'vehicle_id', 'timestamp', 'location', 'velocity',
            'lidar_points', 'imu_data', 'gps_data'
        ]
        
        for field in required_fields:
            if field not in data:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate location data
        location = data.get('location', {})
        if not all(k in location for k in ['latitude', 'longitude']):
            logger.warning("Invalid location data")
            return False
        
        # Validate velocity data
        velocity = data.get('velocity', {})
        if not all(k in velocity for k in ['x', 'y', 'z']):
            logger.warning("Invalid velocity data")
            return False
        
        return True
    
    def calculate_speed(self, velocity: Dict[str, float]) -> float:
        """Calculate speed from velocity components"""
        vx, vy, vz = velocity['x'], velocity['y'], velocity['z']
        return np.sqrt(vx**2 + vy**2 + vz**2)
    
    def detect_anomalies(self, data: Dict[str, Any], calculated_speed: float) -> List[str]:
        """Detect anomalies in sensor data"""
        anomalies = []
        
        # Speed anomaly
        if calculated_speed > self.anomaly_thresholds['speed_limit']:
            anomalies.append(f"excessive_speed_{calculated_speed:.2f}")
        
        # Acceleration anomaly (if we have history)
        vehicle_id = data['vehicle_id']
        if vehicle_id in self.vehicle_histories:
            prev_speed = self.vehicle_histories[vehicle_id].get('speed', 0)
            prev_time = self.vehicle_histories[vehicle_id].get('timestamp', 0)
            
            time_diff = data['timestamp'] - prev_time
            if time_diff > 0:
                acceleration = abs(calculated_speed - prev_speed) / time_diff
                if acceleration > self.anomaly_thresholds['acceleration_limit']:
                    anomalies.append(f"high_acceleration_{acceleration:.2f}")
        
        # LiDAR anomaly
        lidar_points = data.get('lidar_points', [])
        if len(lidar_points) < self.anomaly_thresholds['lidar_min_points']:
            anomalies.append(f"insufficient_lidar_points_{len(lidar_points)}")
        
        # GPS anomaly
        gps_data = data.get('gps_data', {})
        satellites = gps_data.get('satellites', 0)
        if satellites < self.anomaly_thresholds['gps_min_satellites']:
            anomalies.append(f"low_gps_satellites_{satellites}")
        
        return anomalies
    
    def calculate_risk_level(self, calculated_speed: float, anomalies: List[str]) -> str:
        """Calculate risk level based on speed and anomalies"""
        if len(anomalies) >= 3:
            return "high"
        elif len(anomalies) >= 1 or calculated_speed > 20:
            return "medium"
        else:
            return "low"
    
    def process_lidar_data(self, lidar_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process LiDAR point cloud data"""
        if not lidar_points:
            return {
                'point_count': 0,
                'avg_distance': 0,
                'min_distance': 0,
                'max_distance': 0,
                'intensity_avg': 0
            }
        
        distances = []
        intensities = []
        
        for point in lidar_points:
            x, y, z = point.get('x', 0), point.get('y', 0), point.get('z', 0)
            distance = np.sqrt(x**2 + y**2 + z**2)
            distances.append(distance)
            intensities.append(point.get('intensity', 0))
        
        return {
            'point_count': len(lidar_points),
            'avg_distance': np.mean(distances) if distances else 0,
            'min_distance': np.min(distances) if distances else 0,
            'max_distance': np.max(distances) if distances else 0,
            'intensity_avg': np.mean(intensities) if intensities else 0
        }
    
    def process_imu_data(self, imu_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process IMU sensor data"""
        # Calculate total acceleration and angular velocity
        accel_x = imu_data.get('acceleration_x', 0)
        accel_y = imu_data.get('acceleration_y', 0)
        accel_z = imu_data.get('acceleration_z', 9.8)  # Include gravity
        
        total_acceleration = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        angular_x = imu_data.get('angular_velocity_x', 0)
        angular_y = imu_data.get('angular_velocity_y', 0)
        angular_z = imu_data.get('angular_velocity_z', 0)
        
        total_angular_velocity = np.sqrt(angular_x**2 + angular_y**2 + angular_z**2)
        
        return {
            'total_acceleration': total_acceleration,
            'total_angular_velocity': total_angular_velocity,
            'lateral_acceleration': np.sqrt(accel_x**2 + accel_y**2)
        }
    
    def process_sensor_data(self, data: Dict[str, Any]) -> ProcessedSensorData:
        """Process complete sensor data package"""
        try:
            # Calculate derived metrics
            calculated_speed = self.calculate_speed(data['velocity'])
            anomalies = self.detect_anomalies(data, calculated_speed)
            risk_level = self.calculate_risk_level(calculated_speed, anomalies)
            
            # Process subsystem data
            lidar_summary = self.process_lidar_data(data.get('lidar_points', []))
            imu_summary = self.process_imu_data(data.get('imu_data', {}))
            
            # Update vehicle history
            vehicle_id = data['vehicle_id']
            self.vehicle_histories[vehicle_id] = {
                'speed': calculated_speed,
                'timestamp': data['timestamp'],
                'location': data['location']
            }
            
            # Create processed data object
            processed_data = ProcessedSensorData(
                vehicle_id=vehicle_id,
                timestamp=data['timestamp'],
                location=data['location'],
                velocity=data['velocity'],
                calculated_speed=calculated_speed,
                risk_level=risk_level,
                anomalies=anomalies,
                lidar_summary=lidar_summary,
                imu_summary=imu_summary,
                processing_timestamp=datetime.now().timestamp()
            )
            
            logger.info(f"Processed sensor data for vehicle {vehicle_id}, risk: {risk_level}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing sensor data: {e}")
            raise
    
    def process_data_batch(self, data_batch: List[Dict[str, Any]]) -> List[ProcessedSensorData]:
        """Process a batch of sensor data"""
        processed_batch = []
        
        for data in data_batch:
            if self.validate_sensor_data(data):
                try:
                    processed = self.process_sensor_data(data)
                    processed_batch.append(processed)
                except Exception as e:
                    logger.error(f"Failed to process data for vehicle {data.get('vehicle_id')}: {e}")
        
        return processed_batch

# Test function
def test_processor():
    """Test the sensor data processor"""
    print("Testing sensor data processor...")
    
    # Create test data
    test_data = {
        'vehicle_id': 'test_vehicle_1',
        'timestamp': datetime.now().timestamp(),
        'location': {'latitude': 37.7749, 'longitude': -122.4194, 'altitude': 10},
        'velocity': {'x': 5.0, 'y': 2.0, 'z': 0.0},
        'lidar_points': [
            {'x': 1.0, 'y': 2.0, 'z': 0.5, 'intensity': 100, 'timestamp': datetime.now().timestamp()}
            for _ in range(50)
        ],
        'camera_frames': [],
        'imu_data': {
            'acceleration_x': 0.5, 'acceleration_y': 0.2, 'acceleration_z': 9.8,
            'angular_velocity_x': 0.1, 'angular_velocity_y': 0.0, 'angular_velocity_z': 0.0
        },
        'gps_data': {'latitude': 37.7749, 'longitude': -122.4194, 'speed': 7.0, 'satellites': 12},
        'weather_conditions': {'temperature': 20, 'precipitation': 'none', 'visibility': 10000}
    }
    
    # Create processor and test
    processor = SensorDataProcessor()
    
    # Test validation
    is_valid = processor.validate_sensor_data(test_data)
    print(f"Data validation: {is_valid}")
    
    # Test processing
    if is_valid:
        processed = processor.process_sensor_data(test_data)
        print(f"Processing successful: {processed.vehicle_id}")
        print(f"Risk level: {processed.risk_level}")
        print(f"Speed: {processed.calculated_speed:.2f} m/s")
        print(f"Anomalies: {processed.anomalies}")
    
    print("Processor test completed")

if __name__ == "__main__":
    test_processor()
