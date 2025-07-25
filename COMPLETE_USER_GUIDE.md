# 🚗 Autonomous Vehicle Simulation - Complete User Guide

## 🚀 Quick Start - Getting Everything Running

### 1. **Start the Complete System**

```bash
# First, ensure Docker is running
docker --version

# Start Redis container
docker-compose -f docker-compose-minimal.yml up -d

# Activate Python environment
.venv\Scripts\activate

# Check system status
python deployment_status.py
```

### 2. **Generate Vehicle Data**

```bash
# Basic simulation (60 seconds, 5 vehicles)
python simulation\vehicle_simulation.py 60

# Extended simulation (5 minutes)
python simulation\vehicle_simulation.py 300

# Continuous simulation (runs until stopped)
python simulation\continuous_simulation.py
```

### 3. **Launch Interactive Dashboard**

```bash
# Start the dashboard
python dashboard\simple_dashboard.py

# Open in browser: http://localhost:8050
```

## 📊 Core Features & How to Use Them

### 🎯 **1. Vehicle Fleet Simulation**

**Basic Usage:**

```bash
# Simulate 5 vehicles for 2 minutes
python simulation\vehicle_simulation.py 120
```

**Advanced Usage:**

```bash
# Edit simulation\vehicle_simulation.py to customize:
# - Number of vehicles
# - Simulation duration
# - Vehicle behavior patterns
# - Risk scenarios
```

**What You Get:**

- Real-time sensor data generation
- GPS tracking with realistic movement
- LiDAR point cloud simulation
- IMU (accelerometer/gyroscope) data
- Risk level assessment
- Anomaly detection

### 🔄 **2. Real-Time Data Processing**

The system automatically processes sensor data through multiple stages:

```bash
# View processing logs
python flink_processor\sensor_data_processor_nokafka.py
```

**Processing Pipeline:**

1. **Data Validation** - Ensures sensor data integrity
2. **Speed Calculation** - Derives speed from velocity vectors
3. **Anomaly Detection** - Identifies unusual patterns
4. **Risk Assessment** - Calculates low/medium/high risk levels
5. **LiDAR Analysis** - Processes point cloud data
6. **IMU Processing** - Analyzes motion sensor data

### 🌐 **3. Interactive Dashboard**

**Launch:** `python dashboard\simple_dashboard.py`

**Features:**

- **Real-time Vehicle Tracking** - Live positions on map
- **Fleet Overview** - All vehicles status at a glance
- **Risk Analysis** - Color-coded risk levels
- **Speed Monitoring** - Real-time velocity tracking
- **Historical Data** - Time-series analysis
- **Anomaly Alerts** - Immediate notifications

**Dashboard Sections:**

- Vehicle locations and paths
- Speed distribution charts
- Risk level statistics
- Sensor data trends
- Fleet performance metrics

### 💾 **4. Data Storage Systems**

**File-Based Storage:**

```bash
# Raw simulation data
data\simulation\sensor_data_YYYYMMDD_HHMMSS.json

# Processed data
data\simulation\sensor_data_YYYYMMDD_HHMMSS_processed.json

# Parquet format for analytics
data\simulation\sensor_data_latest.parquet
```

**HDFS Simulation:**

```bash
# Organized by vehicle
data\hdfs_simulation\sensor_data\AV_001\
data\hdfs_simulation\sensor_data\AV_002\
# ... etc
```

**Redis Container:**

```bash
# View live data in Redis
docker exec -it av-redis redis-cli

# Common Redis commands:
KEYS *                    # List all keys
GET live:vehicle:AV_001   # Get vehicle data
MONITOR                   # Watch live updates
```

### 🔧 **5. System Management**

**Health Checks:**

```bash
# Complete system status
python deployment_status.py

# Test Redis integration
python test_redis_integration.py

# Demo all features
python docker_deployment_demo.py
```

**Docker Management:**

```bash
# View containers
docker-compose -f docker-compose-minimal.yml ps

# View logs
docker-compose -f docker-compose-minimal.yml logs

# Restart Redis
docker-compose -f docker-compose-minimal.yml restart

# Stop everything
docker-compose -f docker-compose-minimal.yml down
```

## 🎮 **Common Usage Scenarios**

### **Scenario 1: Quick Demo**

```bash
# 1. Start Redis
docker-compose -f docker-compose-minimal.yml up -d

# 2. Generate sample data
python simulation\vehicle_simulation.py 60

# 3. Launch dashboard
python dashboard\simple_dashboard.py

# 4. Open browser to http://localhost:8050
```

### **Scenario 2: Extended Analysis**

```bash
# 1. Run longer simulation
python simulation\vehicle_simulation.py 600  # 10 minutes

# 2. Analyze the data
python -c "
import pandas as pd
df = pd.read_parquet('data/simulation/sensor_data_latest.parquet')
print(df.describe())
print(df['risk_level'].value_counts())
"

# 3. View in dashboard for visualization
```

### **Scenario 3: Continuous Monitoring**

```bash
# Terminal 1: Start continuous simulation
python simulation\continuous_simulation.py

# Terminal 2: Launch dashboard
python dashboard\simple_dashboard.py

# Terminal 3: Monitor Redis data
docker exec -it av-redis redis-cli MONITOR
```

### **Scenario 4: Custom Vehicle Fleet**

**Edit `simulation\vehicle_simulation.py`:**

```python
# Modify these parameters:
num_vehicles = 10      # Increase fleet size
duration_seconds = 1800  # 30 minutes
vehicle_behavior = "aggressive"  # Custom behavior

# Add custom vehicle types:
vehicle_types = ["sedan", "suv", "truck", "bus"]
```

## 📈 **Data Analysis Examples**

### **Python Analysis:**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load latest data
df = pd.read_parquet('data/simulation/sensor_data_latest.parquet')

# Basic statistics
print("Fleet Statistics:")
print(f"Total records: {len(df)}")
print(f"Vehicles: {df['vehicle_id'].nunique()}")
print(f"Time range: {df['timestamp'].max() - df['timestamp'].min():.0f} seconds")

# Risk analysis
risk_stats = df['risk_level'].value_counts()
print("\nRisk Distribution:")
print(risk_stats)

# Speed analysis
print(f"\nSpeed Statistics:")
print(f"Average speed: {df['calculated_speed'].mean():.2f} m/s")
print(f"Max speed: {df['calculated_speed'].max():.2f} m/s")

# Anomaly analysis
df['anomaly_count'] = df['anomalies'].apply(len)
high_anomaly = df[df['anomaly_count'] >= 2]
print(f"\nHigh anomaly events: {len(high_anomaly)}")
```

### **SQL-like Analysis with Pandas:**

```python
# Vehicle performance comparison
vehicle_stats = df.groupby('vehicle_id').agg({
    'calculated_speed': ['mean', 'max'],
    'risk_level': lambda x: (x == 'high').sum(),
    'anomalies': lambda x: sum(len(a) for a in x)
}).round(2)

print("Vehicle Performance:")
print(vehicle_stats)
```

## 🔍 **Troubleshooting Guide**

### **Common Issues:**

**1. Dashboard not loading:**

```bash
# Check if dashboard is running
netstat -an | findstr 8050

# Restart dashboard
python dashboard\simple_dashboard.py
```

**2. Redis connection failed:**

```bash
# Check Redis container
docker ps | findstr redis

# Restart Redis
docker-compose -f docker-compose-minimal.yml restart
```

**3. No simulation data:**

```bash
# Check data directory
dir data\simulation\

# Generate new data
python simulation\vehicle_simulation.py 60
```

**4. Import errors:**

```bash
# Check Python environment
.venv\Scripts\python.exe -c "import dash, redis, pandas, numpy, plotly; print('All packages OK')"

# Reinstall if needed
pip install -r requirements.txt
```

## 🎯 **Advanced Features**

### **1. Custom Risk Models**

Edit `flink_processor\sensor_data_processor_nokafka.py`:

```python
def calculate_risk_level(self, calculated_speed: float, anomalies: List[str]) -> str:
    # Customize risk calculation logic
    if calculated_speed > 25 and len(anomalies) >= 2:
        return "critical"
    # ... your custom logic
```

### **2. Additional Sensors**

Add new sensor types in `simulation\vehicle_simulation.py`:

```python
# Add weather sensors, camera data, etc.
sensor_data.update({
    'weather': {
        'temperature': random.uniform(15, 35),
        'humidity': random.uniform(30, 90),
        'wind_speed': random.uniform(0, 15)
    },
    'camera': {
        'objects_detected': random.randint(0, 10),
        'lane_detection': random.choice(['clear', 'unclear']),
        'traffic_signs': random.randint(0, 3)
    }
})
```

### **3. Machine Learning Integration**

```python
# Example: Add ML-based anomaly detection
from sklearn.ensemble import IsolationForest

# Train model on historical data
model = IsolationForest(contamination=0.1)
model.fit(training_data)

# Use in real-time processing
anomaly_score = model.decision_function(current_data)
```

## 📚 **Project Structure**

```
Autonomous-Vehicle-Simulation-Data-Analysis/
├── simulation/                 # Vehicle simulation
│   ├── vehicle_simulation.py   # Main simulation script
│   └── continuous_simulation.py # Continuous mode
├── flink_processor/            # Data processing
│   └── sensor_data_processor_nokafka.py
├── dashboard/                  # Web interface
│   └── simple_dashboard.py     # Interactive dashboard
├── storage_utils/              # Data storage
│   └── storage_manager_nokafka.py
├── data/                       # Generated data
│   ├── simulation/             # Raw & processed data
│   └── hdfs_simulation/        # HDFS structure
├── docker-compose-minimal.yml  # Docker config
└── requirements.txt            # Python dependencies
```

## 🎉 **Getting the Most Out of Your System**

### **Daily Workflow:**

1. **Morning**: Check system status with `python deployment_status.py`
2. **Generate Data**: Run simulation for desired duration
3. **Monitor**: Keep dashboard open for real-time monitoring
4. **Analyze**: Use Python/Pandas for custom analysis
5. **Evening**: Review logs and system performance

### **Weekly Workflow:**

1. **Scale Testing**: Increase vehicle count and duration
2. **Data Analysis**: Deep dive into patterns and trends
3. **Custom Development**: Add new features or sensors
4. **Performance Tuning**: Optimize based on usage patterns

Your Autonomous Vehicle Simulation is now a complete, production-ready system for vehicle fleet monitoring, data analysis, and real-time processing! 🚗✨
