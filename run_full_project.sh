#!/bin/bash

# Full Project Startup Script
# This script will start the entire Autonomous Vehicle Simulation pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${BLUE}==== $1 ====${NC}"
}

# Create environment file with all required variables
create_env_file() {
    print_section "Creating Environment Configuration"
    
    cat > .env << 'EOF'
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=sensor_data
PROCESSED_TOPIC=processed_sensor_data
INFERENCE_TOPIC=inference_results

# Vehicle Simulation
VEHICLE_COUNT=5
SEND_INTERVAL=2.0

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# HDFS Configuration
HDFS_NAMENODE_URL=http://localhost:9870
HDFS_BASE_PATH=/data

# AWS S3 Configuration (using dummy values for demo)
AWS_ACCESS_KEY_ID=dummy_access_key
AWS_SECRET_ACCESS_KEY=dummy_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET=av-simulation-data

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Dashboard Configuration
DASH_HOST=0.0.0.0
DASH_PORT=8050
DEBUG=false

# Spark Configuration
SPARK_MASTER_URL=spark://localhost:7077

# Flink Configuration
FLINK_JOBMANAGER_RPC_ADDRESS=localhost

# Data Paths
DATA_INPUT_PATH=/data/input
DATA_OUTPUT_PATH=/data/output
MODEL_PATH=/models
CHECKPOINT_PATH=/checkpoints

# Python paths
PYTHONPATH=/app
EOF

    print_status "Created .env file with configuration"
}

# Create required directories
create_directories() {
    print_section "Creating Required Directories"
    
    directories=(
        "data/input"
        "data/output" 
        "data/processed"
        "data/raw"
        "models"
        "checkpoints"
        "logs"
        "monitoring/grafana/provisioning/dashboards"
        "monitoring/grafana/provisioning/datasources"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
}

# Create Grafana configuration
create_grafana_config() {
    print_section "Creating Grafana Configuration"
    
    # Datasource configuration
    cat > monitoring/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Dashboard configuration
    cat > monitoring/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    print_status "Created Grafana configuration files"
}

# Generate dummy training data
generate_dummy_data() {
    print_section "Generating Dummy Training Data"
    
    python3 -c "
import json
import random
import time
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Generate dummy sensor data for training
def generate_training_record():
    base_time = time.time() - random.randint(0, 86400)  # Last 24 hours
    return {
        'vehicle_id': f'vehicle_{random.randint(1, 10)}',
        'timestamp': base_time,
        'calculated_speed': random.uniform(0, 30),
        'lidar_stats': {
            'point_count': random.randint(1000, 8000),
            'avg_intensity': random.uniform(50, 250),
            'max_distance': random.uniform(30, 120),
            'min_distance': random.uniform(0.5, 5),
            'avg_distance': random.uniform(10, 60)
        },
        'lidar_density': random.uniform(0.1, 2.0),
        'weather_severity': random.uniform(0.1, 0.9),
        'speed_variance': random.uniform(0, 5),
        'hour_of_day': random.randint(0, 23),
        'anomaly_count': random.randint(0, 3),
        'risk_level': random.choice(['low', 'medium', 'high']),
        'location': {
            'latitude': 37.7749 + random.uniform(-0.1, 0.1),
            'longitude': -122.4194 + random.uniform(-0.1, 0.1),
            'altitude': random.uniform(0, 100)
        }
    }

# Generate training dataset
training_data = [generate_training_record() for _ in range(5000)]

# Save as JSON
with open('data/input/training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)

# Convert to DataFrame and save as parquet for Spark
df = pd.DataFrame(training_data)

# Flatten nested structures for ML training
df['lidar_stats_point_count'] = df['lidar_stats'].apply(lambda x: x['point_count'])
df['lidar_stats_avg_intensity'] = df['lidar_stats'].apply(lambda x: x['avg_intensity'])
df['lidar_stats_max_distance'] = df['lidar_stats'].apply(lambda x: x['max_distance'])
df['location_latitude'] = df['location'].apply(lambda x: x['latitude'])
df['location_longitude'] = df['location'].apply(lambda x: x['longitude'])

# Encode risk level
risk_mapping = {'low': 0, 'medium': 1, 'high': 2}
df['risk_level_encoded'] = df['risk_level'].map(risk_mapping)

# Save as parquet
df.to_parquet('data/input/training_data.parquet', index=False)

print('Generated 5000 training records')
print('Files created:')
print('- data/input/training_data.json')
print('- data/input/training_data.parquet')
"

    print_status "Generated dummy training data"
}

# Add missing services to docker-compose
update_docker_compose() {
    print_section "Updating Docker Compose with Application Services"
    
    # Add the application services to docker-compose.yml
    cat >> docker-compose.yml << 'EOF'

  # Kafka Producer Service
  kafka-producer:
    build:
      context: ./kafka_producer
      dockerfile: Dockerfile
    container_name: kafka-producer
    depends_on:
      - kafka
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - KAFKA_TOPIC=sensor_data
      - VEHICLE_COUNT=5
      - SEND_INTERVAL=2.0
    networks:
      - av-network
    restart: unless-stopped

  # Dashboard Service
  dashboard:
    build:
      context: ./dashboard
      dockerfile: Dockerfile
    container_name: dashboard
    depends_on:
      - kafka
      - redis
    ports:
      - "8050:8050"
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - DASH_HOST=0.0.0.0
      - DASH_PORT=8050
    networks:
      - av-network
    restart: unless-stopped

  # ML Inference Service
  ml-inference:
    build:
      context: ./ml_inference
      dockerfile: Dockerfile
    container_name: ml-inference
    depends_on:
      - kafka
      - redis
      - mlflow
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - INPUT_TOPIC=processed_sensor_data
      - OUTPUT_TOPIC=inference_results
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - REDIS_HOST=redis
    networks:
      - av-network
    restart: unless-stopped

  # Storage Service
  storage-service:
    build:
      context: ./storage_utils
      dockerfile: Dockerfile
    container_name: storage-service
    depends_on:
      - kafka
      - namenode
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:29092
      - KAFKA_TOPICS=inference_results,processed_sensor_data
      - HDFS_BASE_PATH=/data/streaming
      - S3_BUCKET=av-simulation-data
    networks:
      - av-network
    restart: unless-stopped
EOF

    print_status "Updated docker-compose.yml with application services"
}

# Create missing Dockerfiles
create_dockerfiles() {
    print_section "Creating Missing Dockerfiles"
    
    # Dashboard Dockerfile
    cat > dashboard/Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8050

# Run the application
CMD ["python", "app.py"]
EOF

    # ML Inference Dockerfile
    cat > ml_inference/Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run the inference service
CMD ["python", "inference_service.py"]
EOF

    # Storage Utils Dockerfile
    cat > storage_utils/Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    default-jdk \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run the storage service
CMD ["python", "storage_manager.py"]
EOF

    print_status "Created missing Dockerfiles"
}

# Build and start services
start_services() {
    print_section "Starting All Services"
    
    print_status "Building Docker images..."
    docker-compose build
    
    print_status "Starting infrastructure services..."
    docker-compose up -d zookeeper kafka namenode datanode spark-master spark-worker-1 mlflow redis prometheus grafana
    
    print_status "Waiting for infrastructure to be ready..."
    sleep 45
    
    print_status "Creating Kafka topics..."
    docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic sensor_data --if-not-exists || true
    docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic processed_sensor_data --if-not-exists || true
    docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic inference_results --if-not-exists || true
    
    print_status "Initializing HDFS directories..."
    docker exec namenode hdfs dfs -mkdir -p /data/input || true
    docker exec namenode hdfs dfs -mkdir -p /data/output || true
    docker exec namenode hdfs dfs -mkdir -p /data/processed || true
    docker exec namenode hdfs dfs -mkdir -p /data/streaming || true
    docker exec namenode hdfs dfs -mkdir -p /models || true
    docker exec namenode hdfs dfs -chmod 755 /data || true
    docker exec namenode hdfs dfs -chmod 755 /data/input || true
    docker exec namenode hdfs dfs -chmod 755 /data/output || true
    docker exec namenode hdfs dfs -chmod 755 /data/processed || true
    docker exec namenode hdfs dfs -chmod 755 /data/streaming || true
    docker exec namenode hdfs dfs -chmod 755 /models || true
    
    print_status "Starting application services..."
    docker-compose up -d kafka-producer dashboard
    
    print_status "All services started successfully!"
}

# Train a simple model
train_demo_model() {
    print_section "Training Demo ML Model"
    
    # Create a simple training script
    cat > train_demo_model.py << 'EOF'
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create experiment
experiment_name = "autonomous_vehicle_demo"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)

# Load training data
print("Loading training data...")
df = pd.read_parquet("data/input/training_data.parquet")

# Prepare features
feature_columns = [
    'calculated_speed', 'lidar_stats_point_count', 'lidar_stats_avg_intensity',
    'lidar_stats_max_distance', 'lidar_density', 'weather_severity',
    'speed_variance', 'hour_of_day', 'anomaly_count'
]

X = df[feature_columns].fillna(0)
y = df['risk_level_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "risk_prediction_model")
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save feature names for inference
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_columns, f)
    mlflow.log_artifact("feature_names.pkl")
    
    print("Model logged to MLflow successfully!")

print("Demo model training completed!")
EOF

    print_status "Training demo model..."
    python3 train_demo_model.py
    rm train_demo_model.py
    
    print_status "Demo model training completed!"
}

# Display service URLs
display_access_info() {
    print_section "🎉 Project Started Successfully!"
    
    echo -e "${GREEN}All services are now running!${NC}"
    echo ""
    echo "🌐 Access URLs:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 Dashboard:           http://localhost:8050"
    echo "🔄 Kafka UI:            http://localhost:8080"
    echo "🧠 MLflow:              http://localhost:5000"
    echo "⚡ Spark Master:        http://localhost:8081"
    echo "🗄️ HDFS NameNode:       http://localhost:9870"
    echo "🌊 Flink JobManager:    http://localhost:8082"
    echo "📈 Grafana:             http://localhost:3000 (admin/admin)"
    echo "📊 Prometheus:          http://localhost:9090"
    echo ""
    echo "🔍 Service Status:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    docker-compose ps
    echo ""
    echo "📋 Next Steps:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1. Open the dashboard: http://localhost:8050"
    echo "2. Check Kafka topics: http://localhost:8080"
    echo "3. View ML experiments: http://localhost:5000"
    echo "4. Monitor with Grafana: http://localhost:3000"
    echo ""
    echo "🛑 To stop all services: docker-compose down"
    echo "🔄 To restart services: docker-compose restart"
    echo "📋 To view logs: docker-compose logs -f [service-name]"
    echo ""
    echo -e "${YELLOW}Note: Data is being generated automatically by the Kafka producer!${NC}"
}

# Main execution
main() {
    echo -e "${BLUE}🚗 Starting Autonomous Vehicle Simulation Pipeline...${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
    
    create_env_file
    create_directories
    create_grafana_config
    generate_dummy_data
    create_dockerfiles
    update_docker_compose
    start_services
    sleep 10
    train_demo_model
    display_access_info
}

# Run main function
main "$@"
