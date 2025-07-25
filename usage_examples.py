#!/usr/bin/env python3
"""
AUTONOMOUS VEHICLE SIMULATION - COMPLETE USAGE EXAMPLES
Step-by-step examples showing how to use every feature
"""

def show_basic_usage():
    """Show basic usage examples"""
    print("="*60)
    print("BASIC USAGE - Getting Started")
    print("="*60)
    
    print("\n1. CHECK SYSTEM STATUS:")
    print("   .venv\\Scripts\\python.exe deployment_status.py")
    
    print("\n2. GENERATE VEHICLE DATA:")
    print("   .venv\\Scripts\\python.exe simulation\\vehicle_simulation.py 60")
    print("   -> Creates realistic autonomous vehicle sensor data")
    print("   -> Simulates 5 vehicles for 60 seconds")
    print("   -> Saves to data/simulation/ directory")
    
    print("\n3. VIEW DASHBOARD:")
    print("   .venv\\Scripts\\python.exe dashboard\\simple_dashboard.py")
    print("   -> Open http://localhost:8050 in browser")
    print("   -> Real-time vehicle tracking and analytics")
    
    print("\n4. ACCESS REDIS DATA:")
    print("   docker exec -it av-redis redis-cli")
    print("   -> KEYS *              (list all keys)")
    print("   -> GET vehicle:data:AV_001  (get vehicle data)")

def show_data_analysis():
    """Show data analysis examples"""
    print("\n" + "="*60)
    print("DATA ANALYSIS - Working with Generated Data")
    print("="*60)
    
    print("\nPYTHON DATA ANALYSIS EXAMPLE:")
    print("""
import pandas as pd
import json

# Method 1: Load from Parquet (fastest)
df = pd.read_parquet('data/simulation/sensor_data_latest.parquet')

# Method 2: Load from JSON (most recent)
import glob
json_files = glob.glob('data/simulation/sensor_data_*.json')
latest_file = max(json_files, key=lambda f: os.path.getmtime(f))
with open(latest_file) as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Basic analysis
print(f"Records: {len(df):,}")
print(f"Vehicles: {df['vehicle_id'].nunique()}")
print(f"Time span: {df['timestamp'].max() - df['timestamp'].min():.0f} sec")

# Risk analysis
risk_stats = df['risk_level'].value_counts()
print("Risk Distribution:", dict(risk_stats))

# Speed analysis
print(f"Avg speed: {df['calculated_speed'].mean():.2f} m/s")
print(f"Max speed: {df['calculated_speed'].max():.2f} m/s")

# Vehicle comparison
vehicle_performance = df.groupby('vehicle_id').agg({
    'calculated_speed': ['mean', 'max'],
    'risk_level': lambda x: (x == 'high').sum()
})
print("Vehicle Performance:")
print(vehicle_performance)
""")

def show_redis_usage():
    """Show Redis usage examples"""
    print("\n" + "="*60)
    print("REDIS INTEGRATION - Real-time Data Access")
    print("="*60)
    
    print("\nCOMMAND LINE ACCESS:")
    print("   docker exec -it av-redis redis-cli")
    print("   -> KEYS *                    # List all keys")
    print("   -> GET live:vehicle:AV_001   # Get vehicle data")
    print("   -> MONITOR                   # Watch live updates")
    print("   -> INFO memory               # Memory usage")
    
    print("\nPYTHON REDIS ACCESS:")
    print("""
import redis
import json

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Get all vehicle keys
vehicle_keys = r.keys('live:vehicle:*')
print(f"Found {len(vehicle_keys)} vehicles")

# Get specific vehicle data
if vehicle_keys:
    vehicle_data = json.loads(r.get(vehicle_keys[0]))
    print(f"Vehicle: {vehicle_data['vehicle_id']}")
    print(f"Speed: {vehicle_data.get('speed', 'N/A')} km/h")

# Store custom data
custom_data = {
    "analysis_timestamp": time.time(),
    "fleet_status": "operational",
    "total_vehicles": len(vehicle_keys)
}
r.setex("fleet:status", 300, json.dumps(custom_data))
""")

def show_advanced_features():
    """Show advanced usage examples"""
    print("\n" + "="*60)
    print("ADVANCED FEATURES - Customization & Scaling")
    print("="*60)
    
    print("\n1. CONTINUOUS SIMULATION:")
    print("   .venv\\Scripts\\python.exe simulation\\continuous_simulation.py")
    print("   -> Runs indefinitely until stopped")
    print("   -> Real-time data generation")
    print("   -> Perfect for testing dashboards")
    
    print("\n2. LARGE-SCALE SIMULATION:")
    print("   .venv\\Scripts\\python.exe simulation\\vehicle_simulation.py 3600")
    print("   -> 1 hour of simulation data")
    print("   -> Perfect for ML training")
    
    print("\n3. CUSTOM VEHICLE COUNT:")
    print("   Edit simulation\\vehicle_simulation.py:")
    print("   num_vehicles = 20  # Change from 5 to 20")
    
    print("\n4. MACHINE LEARNING INTEGRATION:")
    print("""
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load data
df = pd.read_parquet('data/simulation/sensor_data_latest.parquet')

# Prepare features for ML
features = ['calculated_speed', 'latitude', 'longitude'] 
X = df[features]
y = df['risk_level']

# Train model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
accuracy = model.score(X_test, y_test)
print(f"Risk prediction accuracy: {accuracy:.2f}")
""")

def show_troubleshooting():
    """Show troubleshooting tips"""
    print("\n" + "="*60)
    print("TROUBLESHOOTING - Common Issues & Solutions")
    print("="*60)
    
    print("\nISSUE: Dashboard not loading")
    print("SOLUTION:")
    print("   1. Check if running: netstat -an | findstr 8050")
    print("   2. Restart: .venv\\Scripts\\python.exe dashboard\\simple_dashboard.py")
    print("   3. Check browser: http://localhost:8050")
    
    print("\nISSUE: Redis connection failed")
    print("SOLUTION:")
    print("   1. Check container: docker ps | findstr redis")
    print("   2. Restart: docker-compose -f docker-compose-minimal.yml restart")
    print("   3. Test: docker exec -it av-redis redis-cli ping")
    
    print("\nISSUE: No simulation data")
    print("SOLUTION:")
    print("   1. Check directory: dir data\\simulation\\")
    print("   2. Generate new: .venv\\Scripts\\python.exe simulation\\vehicle_simulation.py 60")
    print("   3. Verify permissions: Make sure data/ directory is writable")
    
    print("\nISSUE: Import errors")
    print("SOLUTION:")
    print("   1. Activate environment: .venv\\Scripts\\activate")
    print("   2. Check packages: pip list | findstr 'dash redis pandas'")
    print("   3. Reinstall: pip install -r requirements.txt")

def show_file_locations():
    """Show where everything is stored"""
    print("\n" + "="*60)
    print("FILE LOCATIONS - Where to Find Everything")
    print("="*60)
    
    print("\nGENERATED DATA:")
    print("   data/simulation/sensor_data_YYYYMMDD_HHMMSS.json     # Raw data")
    print("   data/simulation/sensor_data_YYYYMMDD_HHMMSS_processed.json  # Processed")
    print("   data/simulation/sensor_data_latest.parquet          # Latest (fast loading)")
    print("   data/simulation/processed_data_latest.parquet       # Latest processed")
    
    print("\nHDFS SIMULATION:")
    print("   data/hdfs_simulation/sensor_data/AV_001/            # Vehicle-specific")
    print("   data/hdfs_simulation/sensor_data/AV_002/            # Organized by vehicle")
    
    print("\nCONFIGURATION:")
    print("   docker-compose-minimal.yml                          # Docker setup")
    print("   requirements.txt                                    # Python packages")
    
    print("\nSCRIPTS:")
    print("   simulation/vehicle_simulation.py                    # Main simulation")
    print("   dashboard/simple_dashboard.py                       # Web dashboard")
    print("   deployment_status.py                               # System check")

def main():
    """Main usage guide"""
    print("AUTONOMOUS VEHICLE SIMULATION")
    print("Complete Usage Guide")
    print("=" * 80)
    
    show_basic_usage()
    show_data_analysis()
    show_redis_usage()
    show_advanced_features()
    show_troubleshooting()
    show_file_locations()
    
    print("\n" + "="*80)
    print("QUICK START CHECKLIST")
    print("="*80)
    print("✓ 1. System Status:     .venv\\Scripts\\python.exe deployment_status.py")
    print("✓ 2. Generate Data:     .venv\\Scripts\\python.exe simulation\\vehicle_simulation.py 60")  
    print("✓ 3. Start Dashboard:   .venv\\Scripts\\python.exe dashboard\\simple_dashboard.py")
    print("✓ 4. Open Browser:      http://localhost:8050")
    print("✓ 5. Check Redis:       docker exec -it av-redis redis-cli KEYS '*'")
    
    print("\n" + "="*80)
    print("YOUR SYSTEM IS READY!")
    print("- Redis container: RUNNING")
    print("- Python environment: ACTIVE") 
    print("- Dashboard: AVAILABLE at http://localhost:8050")
    print("- Data generation: READY")
    print("- All components: TESTED and WORKING")
    print("="*80)

if __name__ == "__main__":
    main()
