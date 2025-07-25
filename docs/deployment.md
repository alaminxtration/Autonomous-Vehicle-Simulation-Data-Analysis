# Autonomous Vehicle Simulation Data Analysis - Deployment Guide

## 🚀 Quick Start (Recommended for Development)

### Prerequisites

- **Windows 10/11** with WSL2 enabled
- **Docker Desktop** with WSL2 integration
- **Python 3.8+** installed
- **Git** (optional but recommended)
- **8GB+ RAM** recommended

### 1. Setup WSL2 (If not already done)

```powershell
# Run in PowerShell as Administrator
wsl --install
# Restart your computer when prompted
```

### 2. Install Docker Desktop

1. Download from [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. Install and enable WSL2 integration
3. Go to Settings > Resources > WSL Integration > Enable integration

### 3. Clone and Setup Project

```bash
# In WSL2 terminal
git clone <your-repo-url>
cd Autonomous-Vehicle-Simulation-Data-Analysis

# Make scripts executable
chmod +x scripts/setup.sh
chmod +x start_pipeline.sh stop_pipeline.sh status_pipeline.sh

# Run setup script
./scripts/setup.sh
```

**Or in Windows PowerShell:**

```powershell
# Navigate to project directory
cd Autonomous-Vehicle-Simulation-Data-Analysis

# Run PowerShell setup
.\scripts\setup.ps1
```

### 4. Update Configuration

Edit the `.env` file and update your AWS credentials:

```bash
AWS_ACCESS_KEY_ID=your_actual_access_key
AWS_SECRET_ACCESS_KEY=your_actual_secret_key
S3_BUCKET=your-bucket-name
```

### 5. Start the Pipeline

```bash
# Linux/WSL
./start_pipeline.sh

# Windows PowerShell
.\start_pipeline.ps1
```

### 6. Access Services

- **Dashboard**: <http://localhost:8050>
- **Kafka UI**: <http://localhost:8080>
- **MLflow**: <http://localhost:5000>
- **Spark UI**: <http://localhost:8081>
- **HDFS**: <http://localhost:9870>
- **Flink**: <http://localhost:8082>
- **Grafana**: <http://localhost:3000> (admin/admin)

---

## 🔧 Manual Step-by-Step Setup

### Phase 1: Infrastructure Setup

#### Start Core Services

```bash
# Start basic infrastructure
docker-compose up -d zookeeper kafka namenode datanode

# Wait for services to start
sleep 30

# Verify services
docker-compose ps
```

#### Create Kafka Topics

```bash
# Create required Kafka topics
docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic sensor_data
docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic processed_sensor_data
docker exec kafka kafka-topics --create --bootstrap-server kafka:29092 --replication-factor 1 --partitions 3 --topic inference_results
```

#### Initialize HDFS

```bash
# Create HDFS directories
docker exec namenode hdfs dfs -mkdir -p /data/raw
docker exec namenode hdfs dfs -mkdir -p /data/processed
docker exec namenode hdfs dfs -mkdir -p /data/streaming
docker exec namenode hdfs dfs -mkdir -p /models
docker exec namenode hdfs dfs -mkdir -p /checkpoints
```

### Phase 2: Data Ingestion

#### Start Kafka Producer

```bash
# Build producer image
docker build -t av-kafka-producer kafka_producer/

# Start producer
docker run -d --name kafka-producer \
  --network autonomous-vehicle-simulation-data-analysis_av-network \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:29092 \
  -e VEHICLE_COUNT=5 \
  av-kafka-producer
```

### Phase 3: Real-time Processing

#### Start Flink Services

```bash
# Start Flink cluster
docker-compose up -d jobmanager taskmanager

# Submit Flink job
docker exec jobmanager flink run -py /opt/flink/jobs/sensor_data_processor.py
```

### Phase 4: Batch Processing

#### Run Spark Jobs

```bash
# Start Spark cluster
docker-compose up -d spark-master spark-worker-1

# Submit Spark job
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  /opt/bitnami/spark/jobs/sensor_data_preprocessing.py
```

### Phase 5: ML Training

#### Start MLflow and Train Models

```bash
# Start MLflow
docker-compose up -d mlflow

# Activate Python environment
source venv/bin/activate

# Train models
cd ml_training
python train_models.py
```

### Phase 6: Real-time Inference

#### Start Inference Service

```bash
# Get the latest model run ID from MLflow UI
# Update MODEL_RUN_ID in your environment

cd ml_inference
python inference_service.py
```

### Phase 7: Visualization

#### Start Dashboard

```bash
cd dashboard
python app.py
```

### Phase 8: Storage

#### Start Storage Services

```bash
cd storage_utils
python storage_manager.py
```

---

## ☸️ Kubernetes Deployment (Production)

### Prerequisites

- Kubernetes cluster (local or cloud)
- kubectl configured
- Helm 3.x (optional)

### Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/kafka-mlflow-deployment.yaml

# Check deployment status
kubectl get pods -n autonomous-vehicle

# Get service URLs
kubectl get services -n autonomous-vehicle
```

### Access Services in K8s

```bash
# Port forward to access services
kubectl port-forward -n autonomous-vehicle svc/kafka-ui 8080:8080 &
kubectl port-forward -n autonomous-vehicle svc/mlflow 5000:5000 &
kubectl port-forward -n autonomous-vehicle svc/dashboard 8050:8050 &
```

---

## 🐛 Troubleshooting

### Common Issues

#### Docker Issues

```bash
# If Docker containers fail to start
docker-compose down
docker system prune -f
docker-compose up -d

# Check container logs
docker-compose logs kafka
docker-compose logs namenode
```

#### WSL2 Issues

```bash
# Restart WSL if needed
wsl --shutdown
wsl

# Check WSL memory usage
wsl --list --verbose
```

#### Port Conflicts

```bash
# Check what's using ports
netstat -tulpn | grep :9092
netstat -tulpn | grep :8080

# Kill processes if needed
sudo kill -9 <PID>
```

#### Python Dependencies

```bash
# If virtual environment issues
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Performance Tuning

#### For WSL2

Create/edit `~/.wslconfig`:

```ini
[wsl2]
memory=8GB
processors=4
swap=2GB
localhostForwarding=true
```

#### For Docker

Increase Docker resources in Docker Desktop:

- Memory: 6GB+
- CPUs: 4+
- Swap: 2GB+

---

## 📊 Monitoring and Health Checks

### Service Health Endpoints

- Kafka: <http://localhost:8080> (Kafka UI)
- HDFS: <http://localhost:9870/dfshealth.html>
- Spark: <http://localhost:8081>
- Flink: <http://localhost:8082>
- MLflow: <http://localhost:5000>
- Grafana: <http://localhost:3000>

### Log Monitoring

```bash
# View service logs
docker-compose logs -f kafka
docker-compose logs -f spark-master
docker-compose logs -f flink-jobmanager

# Check resource usage
docker stats
```

### Data Pipeline Monitoring

```bash
# Check Kafka topics
docker exec kafka kafka-topics --bootstrap-server kafka:29092 --list

# Check HDFS data
docker exec namenode hdfs dfs -ls /data/

# Check MLflow experiments
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

---

## 🔒 Security Considerations

### Development Environment

- Use environment variables for secrets
- Don't commit `.env` files to version control
- Use Docker secrets for sensitive data

### Production Environment

- Enable authentication for all services
- Use TLS/SSL certificates
- Implement network segmentation
- Regular security updates

---

## 📚 Additional Resources

### Documentation

- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Spark Documentation](https://spark.apache.org/docs/latest/)
- [Flink Documentation](https://nightlies.apache.org/flink/flink-docs-stable/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Learning Resources

- [Kafka Tutorials](https://kafka-tutorials.confluent.io/)
- [Spark Examples](https://spark.apache.org/examples.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Community

- [Stack Overflow](https://stackoverflow.com/questions/tagged/apache-kafka+apache-spark)
- [Apache Kafka Users](https://kafka.apache.org/contact)
- [MLflow Community](https://github.com/mlflow/mlflow/discussions)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
