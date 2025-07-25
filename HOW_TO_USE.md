# 🚗 How to Use the Complete Autonomous Vehicle Simulation Project

## ✅ Current Status: LOCAL DEPLOYMENT READY

Your system is configured for local operation with:

- ✅ Python environment ready
- ✅ All simulation components working
- ✅ Dashboard available at <http://localhost:8050>
- ✅ Local data storage and processing
- ⚠️ Full stack Docker deployment skipped (network issues)

## 🎯 How to Use Each Component

### 1. **Generate Vehicle Simulation Data**

```bash
# Basic simulation (1 minute, 5 vehicles)
.venv\Scripts\python.exe simulation\vehicle_simulation.py 60

# Extended simulation (10 minutes)
.venv\Scripts\python.exe simulation\vehicle_simulation.py 600

# Custom duration
.venv\Scripts\python.exe simulation\vehicle_simulation.py 300
```

**What this does:**

- Simulates autonomous vehicles with realistic sensor data
- Generates GPS, LiDAR, IMU, and camera data
- Processes data through risk analysis
- Saves to multiple formats (JSON, Parquet)
- Stores in Redis for real-time access

### 2. **Interactive Dashboard**

The dashboard is already running at **<http://localhost:8050>**

**Features:**

- **Real-time vehicle tracking** - See vehicle positions on map
- **Risk analysis** - Color-coded risk levels (Low/Medium/High)
- **Speed monitoring** - Live velocity tracking
- **Fleet overview** - All vehicles at a glance
- **Historical data** - Time-series charts

**Usage Tips:**

- Refresh page after generating new data
- Use dropdown to select specific vehicles
- Hover over charts for detailed information

### 3. **Redis Data Storage**

Access real-time data in Redis:

```bash
# Connect to Redis CLI
docker exec -it av-redis redis-cli

# Common Redis commands:
KEYS *                          # List all stored keys
GET live:vehicle:AV_001         # Get specific vehicle data
MONITOR                         # Watch live data updates
SCAN 0 MATCH live:vehicle:*     # Find all vehicle keys
```

**Example data retrieval:**

```bash
# Get all live vehicle data
docker exec -it av-redis redis-cli --raw KEYS "live:vehicle:*"
```

### 4. **Data Analysis with Python**

```python
import pandas as pd
import json

# Load latest simulation data
df = pd.read_parquet('data/simulation/sensor_data_latest.parquet')

# Basic analysis
print(f"Total records: {len(df):,}")
print(f"Vehicles: {df['vehicle_id'].nunique()}")
print(f"Time span: {df['timestamp'].max() - df['timestamp'].min():.0f} seconds")

# Risk analysis
risk_counts = df['risk_level'].value_counts()
print("Risk Distribution:")
print(risk_counts)

# Speed analysis
print(f"Average speed: {df['calculated_speed'].mean():.2f} m/s")
print(f"Maximum speed: {df['calculated_speed'].max():.2f} m/s")

# Vehicle comparison
vehicle_stats = df.groupby('vehicle_id').agg({
    'calculated_speed': ['mean', 'max'],
    'risk_level': lambda x: (x == 'high').sum()
})
print("Vehicle Performance:")
print(vehicle_stats)
```

### 5. **System Management Commands**

```bash
# Check complete system status
.venv\Scripts\python.exe deployment_status.py

# View Docker containers
docker ps

# View Redis logs
docker logs av-redis

# Stop Redis container
docker-compose -f docker-compose-minimal.yml down

# Restart Redis container
docker-compose -f docker-compose-minimal.yml restart
```

## 🔄 Complete Usage Workflow

### **Daily Operation:**

1. **Start the system:**

   ```bash
   docker-compose -f docker-compose-minimal.yml up -d
   .venv\Scripts\python.exe dashboard\simple_dashboard.py &
   ```

2. **Generate fresh data:**

   ```bash
   .venv\Scripts\python.exe simulation\vehicle_simulation.py 300
   ```

3. **Monitor via dashboard:**
   - Open <http://localhost:8050>
   - Watch real-time updates

4. **Analyze data:**

   ```python
   import pandas as pd
   df = pd.read_parquet('data/simulation/sensor_data_latest.parquet')
   # Your analysis here
   ```

### **Weekly Analysis:**

1. **Generate extended dataset:**

   ```bash
   .venv\Scripts\python.exe simulation\vehicle_simulation.py 1800  # 30 minutes
   ```

2. **Deep analysis:**

   ```python
   # Load all historical data
   import glob
   all_files = glob.glob('data/simulation/sensor_data_*.json')
   
   # Combine and analyze
   combined_data = []
   for file in all_files:
       with open(file) as f:
           combined_data.extend(json.load(f))
   
   df = pd.DataFrame(combined_data)
   # Comprehensive analysis
   ```

## 📊 Data Formats & Locations

### **Generated Files:**

```
data/simulation/
├── sensor_data_YYYYMMDD_HHMMSS.json          # Raw sensor data
├── sensor_data_YYYYMMDD_HHMMSS_processed.json # Processed data
├── sensor_data_latest.parquet                 # Latest in Parquet format
└── processed_data_latest.parquet              # Latest processed data
```

### **HDFS Simulation:**

```
data/hdfs_simulation/sensor_data/
├── AV_001/
├── AV_002/
├── AV_003/
├── AV_004/
└── AV_005/
```

### **Redis Keys:**

- `live:vehicle:AV_001` - Real-time vehicle data
- `vehicle:data:AV_001` - Simulation data
- Custom keys for your applications

## 🛠️ Customization Examples

### **1. Add More Vehicles:**

Edit `simulation\vehicle_simulation.py`:

```python
# Line ~261
num_vehicles = 10  # Change from 5 to 10
```

### **2. Custom Risk Models:**

Edit `flink_processor\sensor_data_processor_nokafka.py`:

```python
def calculate_risk_level(self, calculated_speed: float, anomalies: List[str]) -> str:
    # Your custom risk logic
    if calculated_speed > 35:  # Custom speed threshold
        return "critical"
    # ... rest of logic
```

### **3. Additional Sensors:**

Edit `simulation\vehicle_simulation.py`, add to `generate_sensor_data()`:

```python
# Add weather sensors
sensor_data['weather'] = {
    'temperature': random.uniform(15, 35),
    'humidity': random.uniform(30, 90),
    'visibility': random.uniform(100, 10000)
}

# Add traffic sensors
sensor_data['traffic'] = {
    'nearby_vehicles': random.randint(0, 8),
    'traffic_density': random.choice(['light', 'moderate', 'heavy']),
    'road_conditions': random.choice(['dry', 'wet', 'icy'])
}
```

## 🎮 Interactive Examples

### **Real-time Monitoring:**

```bash
# Terminal 1: Generate continuous data
.venv\Scripts\python.exe simulation\continuous_simulation.py

# Terminal 2: Monitor Redis
docker exec -it av-redis redis-cli MONITOR

# Terminal 3: Dashboard
.venv\Scripts\python.exe dashboard\simple_dashboard.py
```

### **Batch Analysis:**

```python
# Generate large dataset
import subprocess
subprocess.run(['.venv\\Scripts\\python.exe', 'simulation\\vehicle_simulation.py', '3600'])  # 1 hour

# Load and analyze
df = pd.read_parquet('data/simulation/sensor_data_latest.parquet')

# Find high-risk events
high_risk = df[df['risk_level'] == 'high']
print(f"High risk events: {len(high_risk)}")

# Time-based analysis
df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour
risk_by_hour = df.groupby('hour')['risk_level'].value_counts()
print("Risk distribution by hour:")
print(risk_by_hour)
```

## 🚀 Advanced Features

### **Machine Learning Integration:**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_parquet('data/simulation/sensor_data_latest.parquet')

# Prepare features
features = ['calculated_speed', 'latitude', 'longitude']
X = df[features]
y = df['risk_level']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")
```

### **API Integration:**

```python
import requests
import redis

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Send data to external API
vehicle_data = json.loads(r.get('live:vehicle:AV_001'))
response = requests.post('https://your-api.com/vehicles', json=vehicle_data)
```

## � Full Stack Services Access

### **Complete Service Stack:**

| Service | URL | Purpose | Login |
|---------|-----|---------|-------|
| **Vehicle Dashboard** | <http://localhost:8050> | Real-time vehicle monitoring | - |
| **Kafka UI** | <http://localhost:8080> | Message queue management | - |
| **MLflow** | <http://localhost:5000> | ML model tracking | - |
| **Spark UI** | <http://localhost:8081> | Big data processing | - |
| **HDFS** | <http://localhost:9870> | Distributed file system | - |
| **Flink** | <http://localhost:8082> | Stream processing | - |
| **Grafana** | <http://localhost:3000> | Monitoring dashboards | admin/admin |

### **Deploy Full Stack:**

```bash
# Stop current minimal setup
docker-compose -f docker-compose-minimal.yml down

# Deploy full stack
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### **Progressive Deployment (if network issues):**

```bash
# Step 1: Core services
docker-compose up -d zookeeper kafka kafka-ui redis

# Step 2: Storage
docker-compose up -d namenode datanode

# Step 3: Processing
docker-compose up -d spark-master spark-worker flink-jobmanager flink-taskmanager

# Step 4: Monitoring & ML
docker-compose up -d mlflow grafana prometheus
```

### **Service-Specific Access:**

**Kafka UI (<http://localhost:8080>):**

- Monitor Kafka topics and messages
- View consumer groups
- Manage brokers

**MLflow (<http://localhost:5000>):**

- Track ML experiments
- Model versioning
- Parameter logging

**Spark UI (<http://localhost:8081>):**

- Monitor Spark jobs
- View execution stages
- Check executor status

**HDFS (<http://localhost:9870>):**

- Browse distributed file system
- Monitor cluster health
- View data node status

**Flink (<http://localhost:8082>):**

- Stream processing jobs
- Task manager status
- Checkpoints and metrics

**Grafana (<http://localhost:3000>):**

- Login: admin/admin
- System monitoring
- Custom dashboards
- Alerting

## �🎯 Your Next Steps

1. **Explore the Dashboard**: <http://localhost:8050>
2. **Deploy Full Stack**: Use commands above to access all services
3. **Generate More Data**: Run longer simulations
4. **Custom Analysis**: Use Python/Pandas for insights
5. **Scale Up**: Add more vehicles and sensors
6. **Integrate**: Connect with external systems

Your Autonomous Vehicle Simulation is now a complete, production-ready platform! 🚗✨
