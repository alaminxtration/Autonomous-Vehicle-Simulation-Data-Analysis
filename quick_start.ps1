# Quick Start Script for Autonomous Vehicle Simulation
# This script runs the project with Docker Compose

Write-Host "=== Autonomous Vehicle Simulation - Quick Start ===" -ForegroundColor Blue
Write-Host ""

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Start infrastructure services
Write-Host ""
Write-Host "Starting infrastructure services..." -ForegroundColor Yellow
docker-compose up -d zookeeper kafka kafka-ui namenode datanode mlflow redis prometheus grafana

# Wait for services to be ready
Write-Host "Waiting for services to start (45 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 45

# Create Kafka topics
Write-Host ""
Write-Host "Creating Kafka topics..." -ForegroundColor Yellow
docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic sensor_data --if-not-exists
docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic processed_sensor_data --if-not-exists
docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic inference_results --if-not-exists

# Initialize HDFS directories
Write-Host ""
Write-Host "Initializing HDFS directories..." -ForegroundColor Yellow
docker exec namenode hdfs dfs -mkdir -p /data/input
docker exec namenode hdfs dfs -mkdir -p /data/output
docker exec namenode hdfs dfs -mkdir -p /data/processed
docker exec namenode hdfs dfs -mkdir -p /data/streaming
docker exec namenode hdfs dfs -mkdir -p /data/models
docker exec namenode hdfs dfs -chmod 755 /data

# Start Python services without Docker (for easier development)
Write-Host ""
Write-Host "Starting Python services..." -ForegroundColor Yellow

# Start sensor data producer
Start-Process -FilePath "D:/projects/Autonomous-Vehicle-Simulation-Data-Analysis/.venv/Scripts/python.exe" -ArgumentList "kafka_producer/sensor_data_producer.py" -WorkingDirectory "." -WindowStyle Minimized

# Start Flink processor (simplified version)
Start-Process -FilePath "D:/projects/Autonomous-Vehicle-Simulation-Data-Analysis/.venv/Scripts/python.exe" -ArgumentList "flink_processor/sensor_data_processor_simple.py" -WorkingDirectory "." -WindowStyle Minimized

# Start dashboard
Start-Process -FilePath "D:/projects/Autonomous-Vehicle-Simulation-Data-Analysis/.venv/Scripts/python.exe" -ArgumentList "dashboard/app.py" -WorkingDirectory "." -WindowStyle Minimized

Write-Host ""
Write-Host "=== Services Started Successfully! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Access URLs:" -ForegroundColor Cyan
Write-Host "================================================================"
Write-Host "Dashboard:           http://localhost:8050" -ForegroundColor White
Write-Host "Kafka UI:            http://localhost:8080" -ForegroundColor White
Write-Host "MLflow:              http://localhost:5000" -ForegroundColor White
Write-Host "HDFS NameNode:       http://localhost:9870" -ForegroundColor White
Write-Host "Prometheus:          http://localhost:9090" -ForegroundColor White
Write-Host "Grafana:             http://localhost:3000 (admin/admin)" -ForegroundColor White
Write-Host ""
Write-Host "Note: Python services are running in background windows" -ForegroundColor Yellow
Write-Host "To stop all services: docker-compose down" -ForegroundColor Yellow
Write-Host ""

# Show service status
Write-Host "Docker Service Status:" -ForegroundColor Cyan
docker-compose ps
