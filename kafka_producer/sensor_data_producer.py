import json
import random
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any
import logging
from dataclasses import dataclass, asdict
import numpy as np
from kafka import KafkaProducer
from kafka.errors import KafkaError
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LiDARPoint:
    """Individual LiDAR point representation"""
    x: float
    y: float
    z: float
    intensity: float
    timestamp: float

@dataclass
class CameraFrame:
    """Camera frame representation"""
    frame_id: str
    timestamp: float
    width: int
    height: int
    format: str
    data_size: int
    exposure: float
    iso: int

@dataclass
class SensorData:
    """Complete sensor data package"""
    vehicle_id: str
    timestamp: float
    location: Dict[str, float]
    velocity: Dict[str, float]
    lidar_points: List[Dict[str, float]]
    camera_frames: List[Dict[str, Any]]
    imu_data: Dict[str, float]
    gps_data: Dict[str, float]
    weather_conditions: Dict[str, Any]

class SensorDataGenerator:
    """Generate realistic sensor data for autonomous vehicles"""
    
    def __init__(self, vehicle_id: str = None):
        self.vehicle_id = vehicle_id or str(uuid.uuid4())
        self.current_location = {
            'latitude': 37.7749 + random.uniform(-0.1, 0.1),
            'longitude': -122.4194 + random.uniform(-0.1, 0.1),
            'altitude': random.uniform(0, 100)
        }
        self.current_velocity = {'x': 0, 'y': 0, 'z': 0}
        
    def generate_lidar_points(self, num_points: int = 64000) -> List[Dict[str, float]]:
        """Generate realistic LiDAR point cloud data"""
        points = []
        timestamp = time.time()
        
        for _ in range(num_points):
            # Generate points in spherical coordinates then convert to cartesian
            range_val = random.uniform(0.5, 120.0)  # 0.5m to 120m range
            azimuth = random.uniform(-np.pi, np.pi)
            elevation = random.uniform(-0.4, 0.4)  # Typical LiDAR elevation range
            
            x = range_val * np.cos(elevation) * np.cos(azimuth)
            y = range_val * np.cos(elevation) * np.sin(azimuth)
            z = range_val * np.sin(elevation)
            
            intensity = random.uniform(0, 255)
            
            points.append({
                'x': round(x, 3),
                'y': round(y, 3),
                'z': round(z, 3),
                'intensity': round(intensity, 1),
                'timestamp': timestamp
            })
        
        return points
    
    def generate_camera_frames(self, num_cameras: int = 6) -> List[Dict[str, Any]]:
        """Generate camera frame metadata"""
        frames = []
        timestamp = time.time()
        
        camera_positions = [
            'front', 'rear', 'left', 'right', 'front_left', 'front_right'
        ]
        
        for i in range(min(num_cameras, len(camera_positions))):
            frame = {
                'frame_id': f"{self.vehicle_id}_{camera_positions[i]}_{int(timestamp)}",
                'timestamp': timestamp,
                'camera_position': camera_positions[i],
                'width': 1920,
                'height': 1080,
                'format': 'RGB24',
                'data_size': 1920 * 1080 * 3,  # RGB 24-bit
                'exposure': random.uniform(0.001, 0.033),  # 1ms to 33ms
                'iso': random.choice([100, 200, 400, 800]),
                'focal_length': random.uniform(18, 85),
                'aperture': random.uniform(1.4, 5.6)
            }
            frames.append(frame)
        
        return frames
    
    def generate_imu_data(self) -> Dict[str, float]:
        """Generate IMU (Inertial Measurement Unit) data"""
        return {
            'acceleration_x': random.uniform(-2.0, 2.0),
            'acceleration_y': random.uniform(-2.0, 2.0),
            'acceleration_z': random.uniform(9.0, 10.0),  # Include gravity
            'angular_velocity_x': random.uniform(-0.5, 0.5),
            'angular_velocity_y': random.uniform(-0.5, 0.5),
            'angular_velocity_z': random.uniform(-0.5, 0.5),
            'magnetic_field_x': random.uniform(-50, 50),
            'magnetic_field_y': random.uniform(-50, 50),
            'magnetic_field_z': random.uniform(-50, 50),
            'temperature': random.uniform(20, 40)
        }
    
    def generate_gps_data(self) -> Dict[str, float]:
        """Generate GPS data with realistic movement"""
        # Simulate vehicle movement
        speed = random.uniform(0, 30)  # 0-30 m/s
        direction = random.uniform(0, 2 * np.pi)
        
        # Update velocity
        self.current_velocity = {
            'x': speed * np.cos(direction),
            'y': speed * np.sin(direction),
            'z': random.uniform(-0.1, 0.1)
        }
        
        # Update location (simplified movement)
        lat_change = self.current_velocity['y'] / 111000  # Rough lat conversion
        lon_change = self.current_velocity['x'] / (111000 * np.cos(np.radians(self.current_location['latitude'])))
        
        self.current_location['latitude'] += lat_change
        self.current_location['longitude'] += lon_change
        self.current_location['altitude'] += self.current_velocity['z']
        
        return {
            'latitude': self.current_location['latitude'],
            'longitude': self.current_location['longitude'],
            'altitude': max(0, self.current_location['altitude']),
            'speed': speed,
            'heading': np.degrees(direction),
            'hdop': random.uniform(0.5, 2.0),
            'vdop': random.uniform(0.5, 2.0),
            'satellites': random.randint(8, 24)
        }
    
    def generate_weather_conditions(self) -> Dict[str, Any]:
        """Generate weather condition data"""
        return {
            'temperature': random.uniform(-10, 40),
            'humidity': random.uniform(20, 95),
            'pressure': random.uniform(980, 1040),
            'wind_speed': random.uniform(0, 20),
            'wind_direction': random.uniform(0, 360),
            'visibility': random.uniform(100, 10000),
            'precipitation': random.choice(['none', 'light_rain', 'heavy_rain', 'snow', 'fog']),
            'cloud_cover': random.uniform(0, 100)
        }
    
    def generate_complete_sensor_data(self) -> SensorData:
        """Generate a complete sensor data package"""
        timestamp = time.time()
        
        return SensorData(
            vehicle_id=self.vehicle_id,
            timestamp=timestamp,
            location=self.current_location.copy(),
            velocity=self.current_velocity.copy(),
            lidar_points=self.generate_lidar_points(1000),  # Reduced for testing
            camera_frames=self.generate_camera_frames(),
            imu_data=self.generate_imu_data(),
            gps_data=self.generate_gps_data(),
            weather_conditions=self.generate_weather_conditions()
        )

class KafkaSensorProducer:
    """Kafka producer for sensor data"""
    
    def __init__(self, bootstrap_servers: str = 'localhost:9092', topic: str = 'sensor_data'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self.connect()
        
    def connect(self):
        """Connect to Kafka cluster"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                retries=5,
                retry_backoff_ms=300,
                request_timeout_ms=60000,
                acks='all'
            )
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            raise
    
    def send_sensor_data(self, sensor_data: SensorData):
        """Send sensor data to Kafka topic"""
        try:
            # Convert dataclass to dict
            data_dict = asdict(sensor_data)
            
            # Use vehicle_id and timestamp as key for partitioning
            key = f"{sensor_data.vehicle_id}_{int(sensor_data.timestamp)}"
            
            # Send to Kafka
            future = self.producer.send(
                self.topic,
                key=key,
                value=data_dict,
                timestamp_ms=int(sensor_data.timestamp * 1000)
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            logger.info(
                f"Sent data to topic {record_metadata.topic} "
                f"partition {record_metadata.partition} "
                f"offset {record_metadata.offset}"
            )
            
        except KafkaError as e:
            logger.error(f"Failed to send data to Kafka: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def close(self):
        """Close the producer"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            logger.info("Kafka producer closed")

def main():
    """Main function to run the sensor data producer"""
    # Configuration
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    TOPIC = os.getenv('KAFKA_TOPIC', 'sensor_data')
    VEHICLE_COUNT = int(os.getenv('VEHICLE_COUNT', '5'))
    SEND_INTERVAL = float(os.getenv('SEND_INTERVAL', '1.0'))  # seconds
    
    logger.info(f"Starting sensor data producer for {VEHICLE_COUNT} vehicles")
    logger.info(f"Kafka servers: {KAFKA_SERVERS}")
    logger.info(f"Topic: {TOPIC}")
    logger.info(f"Send interval: {SEND_INTERVAL}s")
    
    # Create producers and generators
    producer = KafkaSensorProducer(KAFKA_SERVERS, TOPIC)
    generators = [SensorDataGenerator() for _ in range(VEHICLE_COUNT)]
    
    try:
        while True:
            for generator in generators:
                # Generate and send sensor data
                sensor_data = generator.generate_complete_sensor_data()
                producer.send_sensor_data(sensor_data)
                
                logger.info(f"Sent data for vehicle {sensor_data.vehicle_id}")
            
            # Wait before next batch
            time.sleep(SEND_INTERVAL)
            
    except KeyboardInterrupt:
        logger.info("Stopping producer...")
    except Exception as e:
        logger.error(f"Producer error: {e}")
    finally:
        producer.close()

if __name__ == "__main__":
    main()
