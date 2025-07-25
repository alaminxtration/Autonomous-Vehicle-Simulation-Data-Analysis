#!/bin/bash

# Autonomous Vehicle Simulation Data Analysis Setup Script
# Optimized for WSL2 and Windows environments

set -e

echo "🚗 Setting up Autonomous Vehicle Simulation Data Analysis Pipeline..."
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if running on WSL
check_wsl() {
    if grep -qi microsoft /proc/version; then
        print_status "Detected WSL environment"
        export WSL_ENV=true
    else
        print_status "Detected native Linux environment"
        export WSL_ENV=false
    fi
}

# Check prerequisites
check_prerequisites() {
    print_section "Checking Prerequisites"
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_status "Docker is installed"
        if docker --version | grep -q "Docker version"; then
            print_status "Docker version: $(docker --version)"
        fi
    else
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_status "Docker Compose is installed"
        print_status "Docker Compose version: $(docker-compose --version)"
    else
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_status "Python3 is installed"
        print_status "Python version: $(python3 --version)"
    else
        print_error "Python3 is not installed. Please install Python3 first."
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        print_status "pip3 is installed"
    else
        print_error "pip3 is not installed. Please install pip3 first."
        exit 1
    fi
}

# Create environment file
create_env_file() {
    print_section "Creating Environment Configuration"
    
    if [ ! -f .env ]; then
        cat > .env << EOF
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=sensor_data
PROCESSED_TOPIC=processed_sensor_data
INFERENCE_TOPIC=inference_results

# Vehicle Simulation
VEHICLE_COUNT=5
SEND_INTERVAL=1.0

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# HDFS Configuration
HDFS_NAMENODE_URL=http://localhost:9870
HDFS_BASE_PATH=/data

# AWS S3 Configuration (update with your credentials)
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
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
EOF
        print_status "Created .env file with default configuration"
        print_warning "Please update AWS credentials in .env file before running"
    else
        print_status ".env file already exists"
    fi
}

# Create directory structure
create_directories() {
    print_section "Creating Directory Structure"
    
    directories=(
        "data/input"
        "data/output" 
        "data/processed"
        "data/raw"
        "models"
        "checkpoints"
        "logs"
        "notebooks"
        "tests"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory already exists: $dir"
        fi
    done
}

# Install Python dependencies
install_python_deps() {
    print_section "Installing Python Dependencies"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies for each component
    components=("kafka_producer" "flink_processor" "spark_jobs" "ml_training" "ml_inference" "storage_utils" "dashboard")
    
    for component in "${components[@]}"; do
        if [ -f "${component}/requirements.txt" ]; then
            print_status "Installing dependencies for $component..."
            pip install -r "${component}/requirements.txt"
        fi
    done
    
    # Install additional development dependencies
    pip install jupyter notebook ipykernel pytest black flake8
    
    deactivate
    print_status "Python dependencies installed"
}

# Configure WSL specific settings
configure_wsl() {
    if [ "$WSL_ENV" = true ]; then
        print_section "Configuring WSL Specific Settings"
        
        # Set memory limits for WSL2
        wsl_config="$HOME/.wslconfig"
        if [ ! -f "$wsl_config" ]; then
            cat > "$wsl_config" << EOF
[wsl2]
memory=8GB
processors=4
swap=2GB
localhostForwarding=true
EOF
            print_status "Created .wslconfig with optimized settings"
            print_warning "Please restart WSL to apply memory settings: wsl --shutdown"
        fi
        
        # Configure Docker for WSL
        print_status "Docker Desktop integration should be enabled for WSL2"
        print_status "Make sure to enable WSL2 integration in Docker Desktop settings"
    fi
}

# Build Docker images
build_docker_images() {
    print_section "Building Docker Images"
    
    # Build MLflow image
    if [ -f "mlflow_tracking/Dockerfile" ]; then
        print_status "Building MLflow image..."
        docker build -t av-mlflow mlflow_tracking/
    fi
    
    # Build Kafka producer image
    if [ -f "kafka_producer/Dockerfile" ]; then
        print_status "Building Kafka producer image..."
        docker build -t av-kafka-producer kafka_producer/
    fi
    
    print_status "Docker images built successfully"
}

# Start core services
start_core_services() {
    print_section "Starting Core Services"
    
    print_status "Starting infrastructure services..."
    docker-compose up -d zookeeper kafka namenode datanode spark-master spark-worker-1 mlflow redis
    
    print_status "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_service_health
}

# Check service health
check_service_health() {
    print_section "Checking Service Health"
    
    services=(
        "Kafka:localhost:9092"
        "Kafka UI:localhost:8080"
        "HDFS NameNode:localhost:9870"
        "Spark Master:localhost:8081"
        "MLflow:localhost:5000"
        "Flink JobManager:localhost:8082"
    )
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        host=$(echo $service | cut -d: -f2)
        port=$(echo $service | cut -d: -f3)
        
        if timeout 5 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            print_status "$name is healthy"
        else
            print_warning "$name is not responding on $host:$port"
        fi
    done
}

# Create Kafka topics
create_kafka_topics() {
    print_section "Creating Kafka Topics"
    
    topics=("sensor_data" "processed_sensor_data" "inference_results")
    
    for topic in "${topics[@]}"; do
        print_status "Creating Kafka topic: $topic"
        docker exec kafka kafka-topics --create \
            --bootstrap-server kafka:29092 \
            --replication-factor 1 \
            --partitions 3 \
            --topic "$topic" \
            --if-not-exists 2>/dev/null || true
    done
}

# Initialize HDFS directories
initialize_hdfs() {
    print_section "Initializing HDFS Directories"
    
    hdfs_dirs=("/data" "/data/raw" "/data/processed" "/data/streaming" "/models" "/checkpoints")
    
    for dir in "${hdfs_dirs[@]}"; do
        print_status "Creating HDFS directory: $dir"
        docker exec namenode hdfs dfs -mkdir -p "$dir" 2>/dev/null || true
        docker exec namenode hdfs dfs -chmod 755 "$dir" 2>/dev/null || true
    done
}

# Create sample data
create_sample_data() {
    print_section "Creating Sample Data"
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        
        # Create a simple data generator script
        cat > generate_sample_data.py << 'EOF'
import json
import random
import time
from datetime import datetime

def generate_sample_record():
    return {
        "vehicle_id": f"vehicle_{random.randint(1, 5)}",
        "timestamp": time.time(),
        "location": {
            "latitude": 37.7749 + random.uniform(-0.01, 0.01),
            "longitude": -122.4194 + random.uniform(-0.01, 0.01),
            "altitude": random.uniform(0, 100)
        },
        "velocity": {
            "x": random.uniform(-10, 30),
            "y": random.uniform(-5, 5),
            "z": random.uniform(-1, 1)
        },
        "calculated_speed": random.uniform(0, 25),
        "lidar_stats": {
            "point_count": random.randint(1000, 5000),
            "avg_intensity": random.uniform(50, 200),
            "max_distance": random.uniform(50, 120)
        },
        "weather_conditions": {
            "temperature": random.uniform(15, 30),
            "humidity": random.uniform(40, 80),
            "visibility": random.uniform(1000, 10000),
            "precipitation": random.choice(["none", "light_rain", "heavy_rain"])
        }
    }

# Generate sample data file
sample_data = [generate_sample_record() for _ in range(100)]

with open("data/input/sample_sensor_data.json", "w") as f:
    json.dump(sample_data, f, indent=2)

print("Generated sample data in data/input/sample_sensor_data.json")
EOF
        
        python3 generate_sample_data.py
        rm generate_sample_data.py
        deactivate
        
        print_status "Sample data created"
    fi
}

# Create utility scripts
create_utility_scripts() {
    print_section "Creating Utility Scripts"
    
    # Start script
    cat > start_pipeline.sh << 'EOF'
#!/bin/bash
echo "Starting Autonomous Vehicle Pipeline..."

# Source environment
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start all services
docker-compose up -d

echo "Waiting for services to start..."
sleep 30

# Start data producer
echo "Starting Kafka producer..."
docker-compose exec kafka-producer python sensor_data_producer.py &

# Start Flink processor
echo "Starting Flink processor..."
docker-compose exec jobmanager flink run -py /opt/flink/jobs/sensor_data_processor.py &

echo "Pipeline started successfully!"
echo "Access points:"
echo "- Kafka UI: http://localhost:8080"
echo "- HDFS: http://localhost:9870"
echo "- Spark: http://localhost:8081"
echo "- MLflow: http://localhost:5000"
echo "- Flink: http://localhost:8082"
echo "- Dashboard: http://localhost:8050"
echo "- Grafana: http://localhost:3000"
EOF
    
    # Stop script
    cat > stop_pipeline.sh << 'EOF'
#!/bin/bash
echo "Stopping Autonomous Vehicle Pipeline..."
docker-compose down
echo "Pipeline stopped."
EOF
    
    # Status script
    cat > status_pipeline.sh << 'EOF'
#!/bin/bash
echo "Autonomous Vehicle Pipeline Status:"
echo "=================================="
docker-compose ps
EOF
    
    chmod +x start_pipeline.sh stop_pipeline.sh status_pipeline.sh
    print_status "Utility scripts created"
}

# Display final instructions
display_instructions() {
    print_section "Setup Complete!"
    
    echo -e "${GREEN}🎉 Autonomous Vehicle Simulation Data Analysis pipeline setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Update AWS credentials in .env file"
    echo "2. Start the pipeline: ./start_pipeline.sh"
    echo "3. Access the dashboard: http://localhost:8050"
    echo ""
    echo "Service URLs:"
    echo "- Kafka UI: http://localhost:8080"
    echo "- HDFS NameNode: http://localhost:9870"
    echo "- Spark Master: http://localhost:8081"
    echo "- MLflow: http://localhost:5000"
    echo "- Flink JobManager: http://localhost:8082"
    echo "- Grafana: http://localhost:3000 (admin/admin)"
    echo "- Dashboard: http://localhost:8050"
    echo ""
    echo "Utility commands:"
    echo "- Start: ./start_pipeline.sh"
    echo "- Stop: ./stop_pipeline.sh"
    echo "- Status: ./status_pipeline.sh"
    echo ""
    echo -e "${YELLOW}Note: If running on WSL, restart WSL after setup: wsl --shutdown${NC}"
}

# Main execution
main() {
    check_wsl
    check_prerequisites
    create_env_file
    create_directories
    install_python_deps
    configure_wsl
    build_docker_images
    start_core_services
    create_kafka_topics
    initialize_hdfs
    create_sample_data
    create_utility_scripts
    display_instructions
}

# Run main function
main "$@"
