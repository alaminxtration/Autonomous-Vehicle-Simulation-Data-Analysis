"""
Vehicle Simulation Data Generator
Generates realistic autonomous vehicle sensor data for testing
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import os
import sys

# Add storage utils to path
sys.path.append('storage_utils')
sys.path.append('flink_processor')

from storage_manager_nokafka import UnifiedStorageManager
from sensor_data_processor_nokafka import SensorDataProcessor

class VehicleSimulation:
    """Simulates autonomous vehicle sensor data"""
    
    def __init__(self, num_vehicles: int = 5, simulation_duration: int = 300):
        self.num_vehicles = num_vehicles
        self.simulation_duration = simulation_duration  # seconds
        self.vehicles = [f"AV_{i:03d}" for i in range(1, num_vehicles + 1)]
        
        # Initialize components
        self.storage_manager = UnifiedStorageManager(enable_s3=False, enable_hdfs=True)
        self.data_processor = SensorDataProcessor()
        
        # Vehicle states
        self.vehicle_states = {}
        self._initialize_vehicles()
    
    def _initialize_vehicles(self):
        """Initialize vehicle starting states"""
        base_lat, base_lon = 37.7749, -122.4194  # San Francisco
        
        for vehicle_id in self.vehicles:
            self.vehicle_states[vehicle_id] = {
                'location': {
                    'latitude': base_lat + random.uniform(-0.01, 0.01),
                    'longitude': base_lon + random.uniform(-0.01, 0.01),
                    'altitude': random.uniform(5, 50)
                },
                'velocity': {
                    'x': random.uniform(-5, 5),
                    'y': random.uniform(-5, 5),
                    'z': 0
                },
                'heading': random.uniform(0, 360),
                'speed': random.uniform(5, 25),  # m/s
                'last_update': datetime.now()
            }
    
    def _generate_lidar_points(self, num_points: int = None) -> list:
        """Generate realistic LiDAR point cloud data"""
        if num_points is None:
            num_points = random.randint(50, 150)
        
        points = []
        for _ in range(num_points):
            # Generate points in a realistic pattern around the vehicle
            distance = np.random.exponential(10)  # Most points close, some far
            angle = random.uniform(0, 2 * np.pi)
            height = random.uniform(-2, 2)
            
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            z = height
            
            points.append({
                'x': x,
                'y': y,
                'z': z,
                'intensity': random.randint(50, 255),
                'timestamp': datetime.now().timestamp()
            })
        
        return points
    
    def _generate_imu_data(self) -> dict:
        """Generate IMU sensor data"""
        return {
            'acceleration_x': random.gauss(0, 0.5),
            'acceleration_y': random.gauss(0, 0.5),
            'acceleration_z': random.gauss(9.8, 0.1),  # Gravity + noise
            'angular_velocity_x': random.gauss(0, 0.1),
            'angular_velocity_y': random.gauss(0, 0.1),
            'angular_velocity_z': random.gauss(0, 0.1)
        }
    
    def _generate_gps_data(self, vehicle_state: dict) -> dict:
        """Generate GPS data with realistic accuracy"""
        location = vehicle_state['location']
        return {
            'latitude': location['latitude'] + random.gauss(0, 0.00001),  # GPS noise
            'longitude': location['longitude'] + random.gauss(0, 0.00001),
            'altitude': location['altitude'] + random.gauss(0, 1),
            'speed': vehicle_state['speed'] + random.gauss(0, 0.5),
            'heading': vehicle_state['heading'] + random.gauss(0, 1),
            'satellites': random.randint(8, 16),
            'hdop': random.uniform(0.5, 2.0)  # Horizontal dilution of precision
        }
    
    def _generate_weather_conditions(self) -> dict:
        """Generate weather conditions"""
        conditions = ['clear', 'cloudy', 'rain', 'fog']
        return {
            'temperature': random.uniform(10, 35),  # Celsius
            'humidity': random.uniform(30, 90),  # Percentage
            'precipitation': random.choice(['none', 'light', 'moderate', 'heavy']),
            'visibility': random.uniform(100, 10000),  # meters
            'wind_speed': random.uniform(0, 15),  # m/s
            'condition': random.choice(conditions)
        }
    
    def _update_vehicle_state(self, vehicle_id: str):
        """Update vehicle position and state"""
        state = self.vehicle_states[vehicle_id]
        dt = 1.0  # 1 second time step
        
        # Update position based on velocity
        state['location']['latitude'] += state['velocity']['y'] * dt / 111000  # Rough conversion
        state['location']['longitude'] += state['velocity']['x'] * dt / (111000 * np.cos(np.radians(state['location']['latitude'])))
        
        # Add some randomness to movement
        state['velocity']['x'] += random.gauss(0, 0.5)
        state['velocity']['y'] += random.gauss(0, 0.5)
        
        # Keep velocity reasonable
        state['velocity']['x'] = np.clip(state['velocity']['x'], -15, 15)
        state['velocity']['y'] = np.clip(state['velocity']['y'], -15, 15)
        
        # Update speed
        state['speed'] = np.sqrt(state['velocity']['x']**2 + state['velocity']['y']**2)
        
        # Update heading
        if state['speed'] > 0:
            state['heading'] = np.degrees(np.arctan2(state['velocity']['y'], state['velocity']['x']))
        
        state['last_update'] = datetime.now()
    
    def generate_sensor_data(self, vehicle_id: str) -> dict:
        """Generate complete sensor data package for a vehicle"""
        self._update_vehicle_state(vehicle_id)
        state = self.vehicle_states[vehicle_id]
        
        sensor_data = {
            'vehicle_id': vehicle_id,
            'timestamp': datetime.now().timestamp(),
            'location': state['location'].copy(),
            'velocity': state['velocity'].copy(),
            'lidar_points': self._generate_lidar_points(),
            'camera_frames': [],  # Placeholder for camera data
            'imu_data': self._generate_imu_data(),
            'gps_data': self._generate_gps_data(state),
            'weather_conditions': self._generate_weather_conditions(),
            'system_status': {
                'battery_level': random.uniform(20, 100),
                'cpu_usage': random.uniform(10, 90),
                'memory_usage': random.uniform(20, 80),
                'disk_usage': random.uniform(10, 70),
                'network_latency': random.uniform(1, 50)  # ms
            }
        }
        
        return sensor_data
    
    def run_simulation(self, output_file: str = None, real_time: bool = False):
        """Run the vehicle simulation"""
        print(f"\n🚗 Starting Autonomous Vehicle Simulation")
        print(f"📊 Vehicles: {self.num_vehicles}")
        print(f"⏱️  Duration: {self.simulation_duration} seconds")
        print(f"💾 Storage: HDFS simulation")
        print("-" * 50)
        
        all_data = []
        processed_data = []
        start_time = datetime.now()
        
        for second in range(self.simulation_duration):
            current_time = datetime.now()
            print(f"⏰ Time: {second:3d}s | Vehicles: {len(self.vehicles)} | Data points: {len(all_data)}", end='\r')
            
            # Generate data for all vehicles
            for vehicle_id in self.vehicles:
                # Generate sensor data
                sensor_data = self.generate_sensor_data(vehicle_id)
                all_data.append(sensor_data)
                
                # Process the data
                if self.data_processor.validate_sensor_data(sensor_data):
                    try:
                        processed = self.data_processor.process_sensor_data(sensor_data)
                        processed_dict = {
                            'vehicle_id': processed.vehicle_id,
                            'timestamp': processed.timestamp,
                            'calculated_speed': processed.calculated_speed,
                            'risk_level': processed.risk_level,
                            'anomalies': processed.anomalies,
                            'processing_timestamp': processed.processing_timestamp
                        }
                        processed_data.append(processed_dict)
                        
                        # Store in HDFS simulation
                        self.storage_manager.upload_data(
                            sensor_data, 
                            f"sensor_data/{vehicle_id}/{current_time.strftime('%Y%m%d_%H%M%S')}.json"
                        )
                        
                    except Exception as e:
                        print(f"\n❌ Processing error for {vehicle_id}: {e}")
            
            # Real-time mode: wait 1 second
            if real_time:
                time.sleep(1)
        
        print(f"\n✅ Simulation Complete!")
        print(f"📊 Generated {len(all_data):,} sensor data points")
        print(f"🔄 Processed {len(processed_data):,} data records")
        
        # Save data to files
        if not os.path.exists('data/simulation'):
            os.makedirs('data/simulation')
        
        # Save raw sensor data
        raw_file = output_file or f'data/simulation/sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(raw_file, 'w') as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"💾 Raw data saved: {raw_file}")
        
        # Save processed data
        processed_file = raw_file.replace('.json', '_processed.json')
        with open(processed_file, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        print(f"🔄 Processed data saved: {processed_file}")
        
        # Save as DataFrame for dashboard
        if all_data:
            df_raw = pd.DataFrame(all_data)
            df_processed = pd.DataFrame(processed_data)
            
            df_raw.to_parquet('data/simulation/sensor_data_latest.parquet', index=False)
            df_processed.to_parquet('data/simulation/processed_data_latest.parquet', index=False)
            
            print(f"📈 DataFrames saved for dashboard")
        
        return all_data, processed_data

def main():
    """Main simulation function"""
    print("Autonomous Vehicle Simulation Data Generator")
    print("=" * 50)
    
    # Configuration
    num_vehicles = 5
    duration = 60  # seconds
    real_time = False
    
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print("Usage: python vehicle_simulation.py [duration_seconds]")
            return
    
    # Create and run simulation
    sim = VehicleSimulation(num_vehicles=num_vehicles, simulation_duration=duration)
    raw_data, processed_data = sim.run_simulation(real_time=real_time)
    
    print(f"\nSimulation Summary:")
    print(f"   - Vehicles: {num_vehicles}")
    print(f"   - Duration: {duration} seconds")
    print(f"   - Raw records: {len(raw_data):,}")
    print(f"   - Processed records: {len(processed_data):,}")
    
    if processed_data:
        risk_counts = pd.Series([d['risk_level'] for d in processed_data]).value_counts()
        print(f"   - Risk levels: {dict(risk_counts)}")
    
    print(f"\nNext steps:")
    print(f"   - Run dashboard: python dashboard/simple_dashboard.py")
    print(f"   - View data: data/simulation/")

if __name__ == "__main__":
    main()
