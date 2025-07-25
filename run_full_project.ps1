# Full Project Startup Script for Windows PowerShell
# This script will start the entire Autonomous Vehicle Simulation pipeline

param(
    [switch]$SkipBuild,
    [switch]$SkipModel
)

# Colors for output
function Write-Status {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Section {
    param([string]$Title)
    Write-Host "`n==== $Title ====" -ForegroundColor Blue
}

# Create environment file
function New-EnvironmentFile {
    Write-Section "Creating Environment Configuration"
    
    $envContent = @"
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
"@

    Set-Content -Path ".env" -Value $envContent
    Write-Status "Created .env file with configuration"
}

# Create required directories
function New-RequiredDirectories {
    Write-Section "Creating Required Directories"
    
    $directories = @(
        "data\input",
        "data\output", 
        "data\processed",
        "data\raw",
        "models",
        "checkpoints",
        "logs",
        "monitoring\grafana\provisioning\dashboards",
        "monitoring\grafana\provisioning\datasources"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Status "Created directory: $dir"
        }
    }
}

# Create Grafana configuration
function New-GrafanaConfig {
    Write-Section "Creating Grafana Configuration"
    
    # Datasource configuration
    $datasourceConfig = @"
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
"@

    # Dashboard configuration
    $dashboardConfig = @"
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
"@

    Set-Content -Path "monitoring\grafana\provisioning\datasources\prometheus.yml" -Value $datasourceConfig
    Set-Content -Path "monitoring\grafana\provisioning\dashboards\dashboard.yml" -Value $dashboardConfig
    
    Write-Status "Created Grafana configuration files"
}

# Generate dummy training data
function New-DummyData {
    Write-Section "Generating Dummy Training Data"
    
    $pythonScript = @"
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
"@

    Set-Content -Path "generate_data.py" -Value $pythonScript
    python generate_data.py
    Remove-Item "generate_data.py"
    
    Write-Status "Generated dummy training data"
}

# Create missing Dockerfiles
function New-Dockerfiles {
    Write-Section "Creating Missing Dockerfiles"
    
    # Dashboard Dockerfile
    $dashboardDockerfile = @"
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
"@

    # ML Inference Dockerfile
    $inferenceDockerfile = @"
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
"@

    # Storage Utils Dockerfile
    $storageDockerfile = @"
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
"@

    Set-Content -Path "dashboard\Dockerfile" -Value $dashboardDockerfile
    Set-Content -Path "ml_inference\Dockerfile" -Value $inferenceDockerfile
    Set-Content -Path "storage_utils\Dockerfile" -Value $storageDockerfile
    
    Write-Status "Created missing Dockerfiles"
}

# Update docker-compose with application services
function Update-DockerCompose {
    Write-Section "Updating Docker Compose with Application Services"
    
    $additionalServices = @"

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
"@

    Add-Content -Path "docker-compose.yml" -Value $additionalServices
    Write-Status "Updated docker-compose.yml with application services"
}

# Start all services
function Start-AllServices {
    Write-Section "Starting All Services"
    
    if (-not $SkipBuild) {
        Write-Status "Building Docker images..."
        docker-compose build
    }
    
    Write-Status "Starting infrastructure services..."
    docker-compose up -d zookeeper kafka namenode datanode spark-master spark-worker-1 mlflow redis prometheus grafana
    
    Write-Status "Waiting for infrastructure to be ready..."
    Start-Sleep -Seconds 45
    
    Write-Status "Creating Kafka topics..."
    docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic sensor_data --if-not-exists
    docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic processed_sensor_data --if-not-exists
    docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic inference_results --if-not-exists
    
    Write-Status "Initializing HDFS directories..."
    docker exec namenode hdfs dfs -mkdir -p /data/input
    docker exec namenode hdfs dfs -mkdir -p /data/output
    docker exec namenode hdfs dfs -mkdir -p /data/processed
    docker exec namenode hdfs dfs -mkdir -p /data/streaming
    docker exec namenode hdfs dfs -mkdir -p /models
    docker exec namenode hdfs dfs -chmod 755 /data
    
    Write-Status "Starting application services..."
    docker-compose up -d kafka-producer dashboard
    
    Write-Status "All services started successfully!"
}

# Train demo model
function Start-DemoModelTraining {
    if ($SkipModel) {
        Write-Warning "Skipping model training..."
        return
    }
    
    Write-Section "Training Demo ML Model"
    
    $trainingScript = @"
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
"@

    Set-Content -Path "train_demo_model.py" -Value $trainingScript
    Write-Status "Training demo model..."
    python train_demo_model.py
    Remove-Item "train_demo_model.py"
    
    Write-Status "Demo model training completed!"
}

# Display service information
function Show-ServiceInfo {
    Write-Section "Project Started Successfully!"
    
    Write-Host "All services are now running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access URLs:" -ForegroundColor Cyan
    Write-Host "================================================================"
    Write-Host "Dashboard:           http://localhost:8050"
    Write-Host "Kafka UI:            http://localhost:8080"
    Write-Host "MLflow:              http://localhost:5000"
    Write-Host "Spark Master:        http://localhost:8081"
    Write-Host "HDFS NameNode:       http://localhost:9870"
    Write-Host "Flink JobManager:    http://localhost:8082"
    Write-Host "Grafana:             http://localhost:3000 (admin/admin)"
    Write-Host "Prometheus:          http://localhost:9090"
    Write-Host ""
    Write-Host "Service Status:" -ForegroundColor Cyan
    Write-Host "================================================================"
    docker-compose ps
    Write-Host ""
    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "================================================================"
    Write-Host "1. Open the dashboard: http://localhost:8050"
    Write-Host "2. Check Kafka topics: http://localhost:8080"
    Write-Host "3. View ML experiments: http://localhost:5000"
    Write-Host "4. Monitor with Grafana: http://localhost:3000"
    Write-Host ""
    Write-Host "To stop all services: docker-compose down"
    Write-Host "To restart services: docker-compose restart"
    Write-Host "To view logs: docker-compose logs -f [service-name]"
    Write-Host ""
    Write-Host "Note: Data is being generated automatically by the Kafka producer!" -ForegroundColor Yellow
}

# Main execution
function Main {
    Write-Host "Starting Autonomous Vehicle Simulation Pipeline..." -ForegroundColor Blue
    Write-Host "====================================================" -ForegroundColor Blue
    
    New-EnvironmentFile
    New-RequiredDirectories
    New-GrafanaConfig
    New-DummyData
    New-Dockerfiles
    Update-DockerCompose
    Start-AllServices
    Start-Sleep -Seconds 10
    Start-DemoModelTraining
    Show-ServiceInfo
}

# Run main function
try {
    Main
}
catch {
    Write-Error "Script failed: $($_.Exception.Message)"
    exit 1
}
