import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource, KafkaSink
from pyflink.datastream.formats.json import JsonRowSerializationSchema, JsonRowDeserializationSchema
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import MapFunction, FilterFunction, ProcessFunction
from pyflink.datastream.state import ValueStateDescriptor
from pyflink.common.time import Time
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.datastream.functions import WindowFunction
from pyflink.common.watermark_strategy import WatermarkStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataValidator(FilterFunction):
    """Validate incoming sensor data"""
    
    def filter(self, value: Dict[str, Any]) -> bool:
        try:
            # Check required fields
            required_fields = ['vehicle_id', 'timestamp', 'location', 'velocity']
            for field in required_fields:
                if field not in value:
                    logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate timestamp
            timestamp = float(value['timestamp'])
            current_time = datetime.now().timestamp()
            if abs(timestamp - current_time) > 300:  # 5 minutes tolerance
                logger.warning(f"Timestamp out of range: {timestamp}")
                return False
            
            # Validate location
            location = value['location']
            if not isinstance(location, dict):
                return False
            
            lat = location.get('latitude', 0)
            lon = location.get('longitude', 0)
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                logger.warning(f"Invalid coordinates: lat={lat}, lon={lon}")
                return False
            
            # Validate LiDAR points count
            lidar_points = value.get('lidar_points', [])
            if len(lidar_points) > 100000:  # Reasonable upper limit
                logger.warning(f"Too many LiDAR points: {len(lidar_points)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

class SensorDataEnricher(MapFunction):
    """Enrich sensor data with additional computed fields"""
    
    def map(self, value: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Add processing timestamp
            value['processing_timestamp'] = datetime.now().isoformat()
            
            # Calculate speed from velocity
            velocity = value.get('velocity', {})
            speed = np.sqrt(
                velocity.get('x', 0)**2 + 
                velocity.get('y', 0)**2 + 
                velocity.get('z', 0)**2
            )
            value['calculated_speed'] = round(speed, 2)
            
            # Add LiDAR statistics
            lidar_points = value.get('lidar_points', [])
            if lidar_points:
                intensities = [point.get('intensity', 0) for point in lidar_points]
                distances = [
                    np.sqrt(point.get('x', 0)**2 + point.get('y', 0)**2 + point.get('z', 0)**2)
                    for point in lidar_points
                ]
                
                value['lidar_stats'] = {
                    'point_count': len(lidar_points),
                    'avg_intensity': round(np.mean(intensities), 2) if intensities else 0,
                    'max_distance': round(max(distances), 2) if distances else 0,
                    'min_distance': round(min(distances), 2) if distances else 0,
                    'avg_distance': round(np.mean(distances), 2) if distances else 0
                }
            
            # Add risk assessment
            weather = value.get('weather_conditions', {})
            visibility = weather.get('visibility', 10000)
            precipitation = weather.get('precipitation', 'none')
            
            risk_score = 0
            if visibility < 1000:
                risk_score += 0.4
            elif visibility < 5000:
                risk_score += 0.2
            
            if precipitation in ['heavy_rain', 'snow']:
                risk_score += 0.3
            elif precipitation in ['light_rain', 'fog']:
                risk_score += 0.1
            
            if speed > 25:  # High speed
                risk_score += 0.2
            
            value['risk_assessment'] = {
                'risk_score': min(1.0, risk_score),
                'risk_level': 'high' if risk_score > 0.6 else 'medium' if risk_score > 0.3 else 'low'
            }
            
            return value
            
        except Exception as e:
            logger.error(f"Enrichment error: {e}")
            return value

class LiDARPointFilter(MapFunction):
    """Filter and downsample LiDAR points for efficiency"""
    
    def map(self, value: Dict[str, Any]) -> Dict[str, Any]:
        try:
            lidar_points = value.get('lidar_points', [])
            
            if not lidar_points:
                return value
            
            # Filter points by distance and intensity
            filtered_points = []
            for point in lidar_points:
                x, y, z = point.get('x', 0), point.get('y', 0), point.get('z', 0)
                distance = np.sqrt(x**2 + y**2 + z**2)
                intensity = point.get('intensity', 0)
                
                # Keep points within reasonable range and intensity
                if 0.5 <= distance <= 100 and intensity > 10:
                    filtered_points.append(point)
            
            # Downsample if too many points (keep every nth point)
            if len(filtered_points) > 10000:
                step = len(filtered_points) // 10000
                filtered_points = filtered_points[::step]
            
            value['lidar_points'] = filtered_points
            value['lidar_filtered_count'] = len(filtered_points)
            value['lidar_original_count'] = len(lidar_points)
            
            return value
            
        except Exception as e:
            logger.error(f"LiDAR filtering error: {e}")
            return value

class AnomalyDetector(ProcessFunction):
    """Detect anomalies in sensor data"""
    
    def __init__(self):
        self.speed_state = None
        self.location_state = None
    
    def open(self, runtime_context):
        # Initialize state for storing historical values
        speed_descriptor = ValueStateDescriptor("speed_history", Types.LIST(Types.FLOAT()))
        self.speed_state = runtime_context.get_state(speed_descriptor)
        
        location_descriptor = ValueStateDescriptor("location_history", Types.LIST(Types.MAP(Types.STRING(), Types.FLOAT())))
        self.location_state = runtime_context.get_state(location_descriptor)
    
    def process_element(self, value, ctx, out):
        try:
            vehicle_id = value['vehicle_id']
            current_speed = value.get('calculated_speed', 0)
            current_location = value.get('location', {})
            
            # Get historical data
            speed_history = self.speed_state.value() or []
            location_history = self.location_state.value() or []
            
            anomalies = []
            
            # Speed anomaly detection
            if len(speed_history) >= 5:
                avg_speed = np.mean(speed_history[-5:])
                if abs(current_speed - avg_speed) > 15:  # 15 m/s deviation
                    anomalies.append({
                        'type': 'speed_anomaly',
                        'current_speed': current_speed,
                        'average_speed': round(avg_speed, 2),
                        'deviation': round(abs(current_speed - avg_speed), 2)
                    })
            
            # Location jump detection
            if location_history:
                last_location = location_history[-1]
                lat_diff = abs(current_location.get('latitude', 0) - last_location.get('latitude', 0))
                lon_diff = abs(current_location.get('longitude', 0) - last_location.get('longitude', 0))
                
                # Check for impossible location jumps (> 0.01 degrees ~ 1km in 1 second)
                if lat_diff > 0.01 or lon_diff > 0.01:
                    anomalies.append({
                        'type': 'location_jump',
                        'lat_diff': round(lat_diff, 6),
                        'lon_diff': round(lon_diff, 6)
                    })
            
            # Update historical data
            speed_history.append(current_speed)
            if len(speed_history) > 10:
                speed_history.pop(0)
            
            location_history.append(current_location)
            if len(location_history) > 10:
                location_history.pop(0)
            
            # Update state
            self.speed_state.update(speed_history)
            self.location_state.update(location_history)
            
            # Add anomalies to output
            value['anomalies'] = anomalies
            value['anomaly_count'] = len(anomalies)
            
            out.collect(value)
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            out.collect(value)

def create_kafka_source(bootstrap_servers: str, topic: str, group_id: str):
    """Create Kafka source"""
    return KafkaSource.builder() \
        .set_bootstrap_servers(bootstrap_servers) \
        .set_topics(topic) \
        .set_group_id(group_id) \
        .set_starting_offsets_earliest() \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()

def create_kafka_sink(bootstrap_servers: str, topic: str):
    """Create Kafka sink"""
    return KafkaSink.builder() \
        .set_bootstrap_servers(bootstrap_servers) \
        .set_record_serializer(
            JsonRowSerializationSchema.builder()
            .with_type_info(Types.ROW_NAMED(['data'], [Types.STRING()]))
            .build()
        ) \
        .set_delivery_guarantee_exactly_once() \
        .set_transactional_id_prefix("flink-sensor-processor") \
        .build()

def main():
    """Main Flink job for sensor data processing"""
    
    # Configuration
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    INPUT_TOPIC = os.getenv('INPUT_TOPIC', 'sensor_data')
    OUTPUT_TOPIC = os.getenv('OUTPUT_TOPIC', 'processed_sensor_data')
    GROUP_ID = os.getenv('CONSUMER_GROUP_ID', 'flink-sensor-processor')
    
    # Create execution environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(2)
    env.enable_checkpointing(60000)  # Checkpoint every 60 seconds
    
    logger.info("Starting Flink sensor data processing job...")
    logger.info(f"Input topic: {INPUT_TOPIC}")
    logger.info(f"Output topic: {OUTPUT_TOPIC}")
    logger.info(f"Kafka servers: {KAFKA_SERVERS}")
    
    try:
        # Create Kafka source
        kafka_source = create_kafka_source(KAFKA_SERVERS, INPUT_TOPIC, GROUP_ID)
        
        # Create data stream
        raw_stream = env.from_source(
            kafka_source,
            WatermarkStrategy.no_watermarks(),
            "Kafka Source"
        )
        
        # Parse JSON
        parsed_stream = raw_stream.map(
            lambda x: json.loads(x),
            output_type=Types.PYTHON_OBJECT()
        )
        
        # Processing pipeline
        processed_stream = parsed_stream \
            .filter(SensorDataValidator()) \
            .map(LiDARPointFilter()) \
            .map(SensorDataEnricher()) \
            .process(AnomalyDetector())
        
        # Convert back to JSON strings for Kafka sink
        output_stream = processed_stream.map(
            lambda x: json.dumps(x, default=str),
            output_type=Types.STRING()
        )
        
        # Print to console for debugging
        output_stream.print("Processed sensor data")
        
        # Create Kafka sink and add to stream
        kafka_sink = KafkaSink.builder() \
            .set_bootstrap_servers(KAFKA_SERVERS) \
            .set_record_serializer(
                SimpleStringSchema()
            ) \
            .set_delivery_guarantee_at_least_once() \
            .build()
        
        output_stream.sink_to(kafka_sink)
        
        # Execute the job
        env.execute("Sensor Data Processing Job")
        
    except Exception as e:
        logger.error(f"Job execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
