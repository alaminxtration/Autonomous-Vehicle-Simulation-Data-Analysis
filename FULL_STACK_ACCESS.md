# 🌐 Full Stack Access Guide - Autonomous Vehicle Simulation

## 🔧 Current Status vs Full Stack

### ✅ **Currently Running (Minimal Setup):**

- **Redis**: `localhost:6379` ✅ ACTIVE
- **Dashboard**: `http://localhost:8050` ✅ ACTIVE  
- **Python Services**: Local environment ✅ ACTIVE

### 🎯 **Full Stack Services (Available):**

- **Dashboard**: `http://localhost:8050` - Vehicle monitoring
- **Kafka UI**: `http://localhost:8080` - Message queue management
- **MLflow**: `http://localhost:5000` - ML model tracking
- **Spark UI**: `http://localhost:8081` - Big data processing
- **HDFS**: `http://localhost:9870` - Distributed file system
- **Flink**: `http://localhost:8082` - Stream processing
- **Grafana**: `http://localhost:3000` - Monitoring dashboards (admin/admin)

## 🚀 How to Deploy Full Stack

### **Option 1: Try Full Stack Deployment**

```bash
# Stop current minimal setup
docker-compose -f docker-compose-minimal.yml down

# Try full stack (requires good internet connection)
docker-compose up -d

# Check status
docker-compose ps
```

### **Option 2: Gradual Service Deployment**

Let's add services one by one to avoid network issues:

#### **Step 1: Core Services (Kafka + Zookeeper)**

```bash
# Create intermediate compose file
docker-compose -f docker-compose-core.yml up -d zookeeper kafka kafka-ui
```

#### **Step 2: Add HDFS**

```bash
docker-compose -f docker-compose-core.yml up -d namenode datanode
```

#### **Step 3: Add Processing (Spark + Flink)**

```bash
docker-compose -f docker-compose-core.yml up -d spark-master spark-worker flink-jobmanager flink-taskmanager
```

#### **Step 4: Add Monitoring (MLflow + Grafana)**

```bash
docker-compose -f docker-compose-core.yml up -d mlflow grafana prometheus
```

## 📋 Service Access Guide

### **1. Kafka UI - Message Queue Management**

```bash
# URL: http://localhost:8080
# Purpose: Monitor Kafka topics, messages, consumers
# Dependencies: Requires Kafka + Zookeeper running

# Start Kafka stack:
docker-compose up -d zookeeper kafka kafka-ui

# Access: Browser -> http://localhost:8080
# Features:
# - View Kafka topics
# - Monitor message flow
# - Consumer group management
# - Broker health status
```

### **2. MLflow - ML Model Tracking**

```bash
# URL: http://localhost:5000  
# Purpose: Track ML experiments, models, parameters
# Dependencies: Standalone service

# Start MLflow:
docker-compose up -d mlflow

# Access: Browser -> http://localhost:5000
# Features:
# - Track ML experiments
# - Model versioning
# - Parameter logging
# - Model deployment
```

### **3. Spark UI - Big Data Processing**

```bash
# URL: http://localhost:8081
# Purpose: Monitor Spark jobs and performance
# Dependencies: Spark master + worker

# Start Spark:
docker-compose up -d spark-master spark-worker

# Access: Browser -> http://localhost:8081
# Features:
# - Job monitoring
# - Stage details
# - Executor status
# - SQL queries
```

### **4. HDFS Web UI - Distributed Storage**

```bash
# URL: http://localhost:9870
# Purpose: Monitor HDFS file system
# Dependencies: NameNode + DataNode

# Start HDFS:
docker-compose up -d namenode datanode

# Access: Browser -> http://localhost:9870
# Features:
# - File system browser
# - Cluster health
# - Data node status
# - Storage utilization
```

### **5. Flink Dashboard - Stream Processing**

```bash
# URL: http://localhost:8082
# Purpose: Monitor real-time stream processing
# Dependencies: JobManager + TaskManager

# Start Flink:
docker-compose up -d flink-jobmanager flink-taskmanager

# Access: Browser -> http://localhost:8082
# Features:
# - Job overview
# - Task manager status
# - Checkpoints
# - Metrics
```

### **6. Grafana - Monitoring Dashboards**

```bash
# URL: http://localhost:3000
# Default Login: admin/admin
# Purpose: System monitoring and visualization
# Dependencies: Prometheus (data source)

# Start monitoring stack:
docker-compose up -d prometheus grafana

# Access: Browser -> http://localhost:3000
# Login: admin / admin
# Features:
# - System metrics
# - Custom dashboards
# - Alerting
# - Data visualization
```

## 🛠️ Troubleshooting Network Issues

Since we had Docker Hub connectivity issues, here are alternatives:

### **Option A: Use Local Images (if available)**

```bash
# Check available local images
docker images

# Look for:
# - redis:7-alpine ✅ (already working)
# - confluentinc/cp-kafka
# - bde2020/hadoop-namenode
# - apache/spark
# - etc.
```

### **Option B: Alternative Image Sources**

```bash
# Use different registries or tags
# Edit docker-compose.yml to use:
# - Lighter images
# - Alternative registries
# - Specific versions available locally
```

### **Option C: Progressive Deployment**

Let me create a step-by-step deployment script:
