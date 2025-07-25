# Autonomous Vehicle Simulation Data Analysis Pipeline

A comprehensive end-to-end pipeline for autonomous vehicle simulation data analysis using modern data engineering and machine learning tools.

## 🏗️ Architecture Overview

```
Kafka Producer → Kafka Cluster → Flink (Real-time Processing) → HDFS/S3 Storage
                                      ↓
Spark (Batch Processing) → ML Training (PyTorch/TensorFlow) → MLflow Tracking
                                      ↓
Real-time Inference → Dash Dashboard → Monitoring (Prometheus/Grafana)
```

## 🛠️ Tech Stack

- **Streaming**: Apache Kafka
- **Real-time Processing**: Apache Flink (PyFlink)
- **Batch Processing**: Apache Spark (PySpark)
- **ML Frameworks**: PyTorch, TensorFlow
- **ML Tracking**: MLflow
- **Storage**: AWS S3, HDFS
- **Visualization**: Dash/Plotly
- **Orchestration**: Docker, Kubernetes
- **Monitoring**: Prometheus, Grafana

## 📁 Project Structure

```
├── kafka_producer/          # Kafka data producers
├── flink_processor/         # Real-time stream processing
├── spark_jobs/             # Batch processing jobs
├── ml_training/            # ML model training scripts
├── ml_inference/           # Real-time inference services
├── mlflow_tracking/        # MLflow configurations
├── storage_utils/          # S3/HDFS utilities
├── dashboard/              # Dash visualization
├── docker/                 # Docker configurations
├── k8s/                    # Kubernetes manifests
├── monitoring/             # Prometheus/Grafana configs
├── config/                 # Configuration files
├── scripts/                # Utility scripts
└── docs/                   # Documentation
```

## 🚀 Quick Start

1. **Prerequisites**: WSL2, Docker, Docker Compose
2. **Setup Environment**: `./scripts/setup.sh`
3. **Start Services**: `docker-compose up -d`
4. **Run Pipeline**: Follow phase-by-phase instructions below

## 📋 Phase-by-Phase Implementation

### Phase 1: Infrastructure Setup

- Kafka + Zookeeper cluster
- HDFS setup
- MLflow tracking server

### Phase 2: Data Ingestion

- Kafka producer for sensor data simulation
- Data schema definition

### Phase 3: Real-time Processing

- Flink jobs for stream processing
- Data validation and enrichment

### Phase 4: Batch Processing

- Spark jobs for data preprocessing
- Feature engineering pipelines

### Phase 5: ML Training

- Object detection model training
- MLflow experiment tracking

### Phase 6: Real-time Inference

- Model serving infrastructure
- Real-time prediction pipeline

### Phase 7: Visualization

- Dash dashboard for monitoring
- Real-time metrics display

### Phase 8: Deployment

- Kubernetes deployment
- Monitoring and alerting

## 🔧 Development Environment

This project is optimized for:

- **WSL2** on Windows
- **Docker** containerization
- **VS Code** development environment
- **Cross-platform** compatibility

## 📚 Documentation

- [Setup Guide](docs/setup.md)
- [API Reference](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and development process.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
