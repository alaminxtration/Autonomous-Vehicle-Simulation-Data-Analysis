import json
import logging
import time
from typing import Dict, List, Any
from dataclasses import dataclass
import numpy as np
# Using Kafka instead of PyFlink for simplified development
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import os
import threading
from datetime import datetime, timedelta
import statistics

# Configure logging
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
    lidar_stats: Dict[str, float]
    lidar_density: float
    weather_severity: float
    speed_variance: float
    hour_of_day: int
    anomaly_count: int
    risk_level: str

class SensorDataProcessor:
    """Simplified sensor data processor using Kafka Python"""
    
    def __init__(self, 
                 bootstrap_servers: str = 'localhost:9092',
                 input_topic: str = 'sensor_data',
                 output_topic: str = 'processed_sensor_data'):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer = None
        self.producer = None
        self.vehicle_histories = {}  # Store vehicle history for analysis
        self.setup_kafka()
    
    def setup_kafka(self):
        """Setup Kafka consumer and producer"""
        try:
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                group_id='sensor_processor_group',
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8')
            )
            
            logger.info(f"Connected to Kafka at {self.bootstrap_servers}")
            
        except Exception as e:
            logger.error(f"Failed to setup Kafka: {e}")
            raise
    
    def validate_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Validate incoming sensor data"""
        required_fields = ['vehicle_id', 'timestamp', 'location', 'velocity', 
                          'lidar_points', 'imu_data', 'gps_data']
        
        # Check required fields
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
        if not all(k in velocity for k in ['x', 'y']):
            logger.warning("Invalid velocity data")
            return False
        
        # Validate timestamp
        try:
            timestamp = float(data['timestamp'])
            current_time = time.time()
            if abs(timestamp - current_time) > 3600:  # 1 hour tolerance
                logger.warning(f"Timestamp too old or in future: {timestamp}")
                return False
        except (ValueError, TypeError):
            logger.warning("Invalid timestamp")
            return False
        
        return True
    
    def calculate_lidar_statistics(self, lidar_points: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate statistics from LiDAR point cloud"""
        if not lidar_points:
            return {
                'point_count': 0,
                'avg_intensity': 0,
                'max_distance': 0,
                'min_distance': 0,
                'avg_distance': 0
            }
        
        try:
            intensities = [p.get('intensity', 0) for p in lidar_points]
            distances = [
                np.sqrt(p.get('x', 0)**2 + p.get('y', 0)**2 + p.get('z', 0)**2) 
                for p in lidar_points
            ]
            
            return {
                'point_count': len(lidar_points),
                'avg_intensity': statistics.mean(intensities) if intensities else 0,
                'max_distance': max(distances) if distances else 0,
                'min_distance': min(distances) if distances else 0,
                'avg_distance': statistics.mean(distances) if distances else 0
            }
        except Exception as e:
            logger.error(f"Error calculating LiDAR stats: {e}")
            return {
                'point_count': len(lidar_points),
                'avg_intensity': 0,
                'max_distance': 0,
                'min_distance': 0,
                'avg_distance': 0
            }
    
    def calculate_speed(self, velocity: Dict[str, float]) -> float:
        """Calculate speed from velocity components"""
        try:
            vx = velocity.get('x', 0)
            vy = velocity.get('y', 0)
            return np.sqrt(vx**2 + vy**2)
        except Exception:
            return 0.0
    
    def calculate_weather_severity(self, weather: Dict[str, Any]) -> float:
        """Calculate weather severity score (0-1)"""
        try:
            severity = 0.0
            
            # Precipitation impact
            precipitation = weather.get('precipitation', 'none')
            if precipitation in ['heavy_rain', 'snow']:
                severity += 0.4
            elif precipitation in ['light_rain', 'fog']:
                severity += 0.2
            
            # Visibility impact
            visibility = weather.get('visibility', 10000)
            if visibility < 1000:
                severity += 0.3
            elif visibility < 5000:
                severity += 0.1
            
            # Wind impact
            wind_speed = weather.get('wind_speed', 0)
            if wind_speed > 15:
                severity += 0.2
            elif wind_speed > 10:
                severity += 0.1
            
            return min(severity, 1.0)
            
        except Exception:
            return 0.0
    
    def detect_anomalies(self, data: Dict[str, Any]) -> int:
        """Detect anomalies in sensor data"""
        anomaly_count = 0
        
        try:
            # Check LiDAR point count
            lidar_points = data.get('lidar_points', [])
            if len(lidar_points) < 100:  # Very low point count
                anomaly_count += 1
            
            # Check IMU data for extreme values
            imu = data.get('imu_data', {})
            accel_x = abs(imu.get('acceleration_x', 0))
            accel_y = abs(imu.get('acceleration_y', 0))
            if accel_x > 5 or accel_y > 5:  # High acceleration
                anomaly_count += 1
            
            # Check GPS data
            gps = data.get('gps_data', {})
            satellites = gps.get('satellites', 12)
            if satellites < 4:  # Poor GPS
                anomaly_count += 1
            
            # Check velocity consistency
            velocity = data.get('velocity', {})
            speed = self.calculate_speed(velocity)
            if speed > 50:  # Very high speed (m/s)
                anomaly_count += 1
                
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            anomaly_count += 1
        
        return anomaly_count
    
    def calculate_risk_level(self, processed_data: ProcessedSensorData) -> str:
        """Calculate risk level based on processed data"""
        risk_score = 0
        
        # Weather impact
        risk_score += processed_data.weather_severity * 30
        
        # Speed impact
        if processed_data.calculated_speed > 25:  # High speed
            risk_score += 20
        elif processed_data.calculated_speed > 15:
            risk_score += 10
        
        # Anomaly impact
        risk_score += processed_data.anomaly_count * 15
        
        # LiDAR density impact
        if processed_data.lidar_density < 0.5:
            risk_score += 15
        
        # Time of day impact (night driving)
        if processed_data.hour_of_day < 6 or processed_data.hour_of_day > 20:
            risk_score += 10
        
        # Determine risk level
        if risk_score > 60:
            return 'high'
        elif risk_score > 30:
            return 'medium'
        else:
            return 'low'
    
    def update_vehicle_history(self, vehicle_id: str, speed: float):
        """Update vehicle history for variance calculation"""
        if vehicle_id not in self.vehicle_histories:
            self.vehicle_histories[vehicle_id] = []
        
        # Keep last 10 speed measurements
        self.vehicle_histories[vehicle_id].append(speed)
        if len(self.vehicle_histories[vehicle_id]) > 10:
            self.vehicle_histories[vehicle_id].pop(0)
    
    def calculate_speed_variance(self, vehicle_id: str) -> float:
        """Calculate speed variance for a vehicle"""
        history = self.vehicle_histories.get(vehicle_id, [])
        if len(history) < 2:
            return 0.0
        
        try:
            return statistics.variance(history)
        except Exception:
            return 0.0
    
    def process_sensor_data(self, raw_data: Dict[str, Any]) -> ProcessedSensorData:
        """Process raw sensor data"""
        vehicle_id = raw_data['vehicle_id']
        timestamp = float(raw_data['timestamp'])
        
        # Calculate derived metrics
        calculated_speed = self.calculate_speed(raw_data['velocity'])
        lidar_stats = self.calculate_lidar_statistics(raw_data['lidar_points'])
        weather_severity = self.calculate_weather_severity(raw_data['weather_conditions'])
        anomaly_count = self.detect_anomalies(raw_data)
        
        # Update vehicle history
        self.update_vehicle_history(vehicle_id, calculated_speed)
        speed_variance = self.calculate_speed_variance(vehicle_id)
        
        # Calculate LiDAR density
        lidar_density = lidar_stats['point_count'] / 1000.0  # Normalize
        
        # Extract hour of day
        hour_of_day = datetime.fromtimestamp(timestamp).hour
        
        # Create processed data
        processed_data = ProcessedSensorData(
            vehicle_id=vehicle_id,
            timestamp=timestamp,
            location=raw_data['location'].copy(),
            velocity=raw_data['velocity'].copy(),
            calculated_speed=calculated_speed,
            lidar_stats=lidar_stats,
            lidar_density=lidar_density,
            weather_severity=weather_severity,
            speed_variance=speed_variance,
            hour_of_day=hour_of_day,
            anomaly_count=anomaly_count,
            risk_level='low'  # Will be calculated next
        )
        
        # Calculate risk level
        processed_data.risk_level = self.calculate_risk_level(processed_data)
        
        return processed_data
    
    def send_processed_data(self, processed_data: ProcessedSensorData):
        """Send processed data to output topic"""
        try:
            # Convert to dict
            data_dict = {
                'vehicle_id': processed_data.vehicle_id,
                'timestamp': processed_data.timestamp,
                'location': processed_data.location,
                'velocity': processed_data.velocity,
                'calculated_speed': processed_data.calculated_speed,
                'lidar_stats': processed_data.lidar_stats,
                'lidar_density': processed_data.lidar_density,
                'weather_severity': processed_data.weather_severity,
                'speed_variance': processed_data.speed_variance,
                'hour_of_day': processed_data.hour_of_day,
                'anomaly_count': processed_data.anomaly_count,
                'risk_level': processed_data.risk_level,
                'processed_at': time.time()
            }
            
            # Send to Kafka
            key = f"{processed_data.vehicle_id}_{int(processed_data.timestamp)}"
            self.producer.send(
                self.output_topic,
                key=key,
                value=data_dict
            )
            
            logger.debug(f"Sent processed data for vehicle {processed_data.vehicle_id}")
            
        except Exception as e:
            logger.error(f"Failed to send processed data: {e}")
    
    def run(self):
        """Main processing loop"""
        logger.info("Starting sensor data processor...")
        
        try:
            for message in self.consumer:
                try:
                    raw_data = message.value
                    
                    # Validate data
                    if not self.validate_sensor_data(raw_data):
                        logger.warning("Invalid sensor data received, skipping")
                        continue
                    
                    # Process data
                    processed_data = self.process_sensor_data(raw_data)
                    
                    # Send processed data
                    self.send_processed_data(processed_data)
                    
                    logger.info(f"Processed data for vehicle {processed_data.vehicle_id}, "
                              f"risk level: {processed_data.risk_level}")
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Stopping processor...")
        except Exception as e:
            logger.error(f"Processor error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.consumer:
            self.consumer.close()
        if self.producer:
            self.producer.flush()
            self.producer.close()
        logger.info("Processor cleanup completed")

def main():
    """Main function"""
    # Configuration
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    INPUT_TOPIC = os.getenv('KAFKA_TOPIC', 'sensor_data')
    OUTPUT_TOPIC = os.getenv('PROCESSED_TOPIC', 'processed_sensor_data')
    
    logger.info(f"Starting sensor data processor")
    logger.info(f"Kafka servers: {KAFKA_SERVERS}")
    logger.info(f"Input topic: {INPUT_TOPIC}")
    logger.info(f"Output topic: {OUTPUT_TOPIC}")
    
    # Create and run processor
    processor = SensorDataProcessor(
        bootstrap_servers=KAFKA_SERVERS,
        input_topic=INPUT_TOPIC,
        output_topic=OUTPUT_TOPIC
    )
    
    processor.run()

if __name__ == "__main__":
    main()
