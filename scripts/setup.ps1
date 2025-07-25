# Autonomous Vehicle Simulation Data Analysis Setup Script for Windows PowerShell
# Optimized for Windows with WSL2 and Docker Desktop

param(
    [switch]$SkipPrerequisites,
    [switch]$SkipDocker,
    [string]$PythonPath = "python"
)

# Colors for output
$Red = [System.ConsoleColor]::Red
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Blue = [System.ConsoleColor]::Blue
$White = [System.ConsoleColor]::White

function Write-ColorOutput {
    param(
        [string]$Message,
        [System.ConsoleColor]$Color = $White
    )
    $originalColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Output $Message
    $Host.UI.RawUI.ForegroundColor = $originalColor
}

function Write-Status {
    param([string]$Message)
    Write-ColorOutput "[INFO] $Message" $Green
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "[WARNING] $Message" $Yellow
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "[ERROR] $Message" $Red
}

function Write-Section {
    param([string]$Title)
    Write-ColorOutput "`n==== $Title ====" $Blue
}

# Check if running as Administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check prerequisites
function Test-Prerequisites {
    Write-Section "Checking Prerequisites"
    
    $allGood = $true
    
    # Check WSL
    try {
        $wslVersion = wsl --version 2>$null
        if ($wslVersion) {
            Write-Status "WSL is installed"
            Write-Status "WSL Version: $($wslVersion[0])"
        }
    }
    catch {
        Write-Error "WSL is not installed or not accessible"
        Write-Warning "Please install WSL2 from Microsoft Store or run: wsl --install"
        $allGood = $false
    }
    
    # Check Docker Desktop
    try {
        $dockerVersion = docker --version 2>$null
        if ($dockerVersion) {
            Write-Status "Docker is installed"
            Write-Status "Docker Version: $dockerVersion"
        }
    }
    catch {
        Write-Error "Docker is not installed or not accessible"
        Write-Warning "Please install Docker Desktop for Windows"
        $allGood = $false
    }
    
    # Check Docker Compose
    try {
        $dockerComposeVersion = docker-compose --version 2>$null
        if ($dockerComposeVersion) {
            Write-Status "Docker Compose is installed"
            Write-Status "Docker Compose Version: $dockerComposeVersion"
        }
    }
    catch {
        Write-Error "Docker Compose is not installed"
        $allGood = $false
    }
    
    # Check Python
    try {
        $pythonVersion = & $PythonPath --version 2>$null
        if ($pythonVersion) {
            Write-Status "Python is installed"
            Write-Status "Python Version: $pythonVersion"
        }
    }
    catch {
        Write-Error "Python is not installed or not accessible"
        Write-Warning "Please install Python from python.org or Microsoft Store"
        $allGood = $false
    }
    
    # Check Git
    try {
        $gitVersion = git --version 2>$null
        if ($gitVersion) {
            Write-Status "Git is installed"
        }
    }
    catch {
        Write-Warning "Git is not installed (optional but recommended)"
    }
    
    if (-not $allGood) {
        Write-Error "Some prerequisites are missing. Please install them and run the script again."
        exit 1
    }
}

# Create environment file
function New-EnvironmentFile {
    Write-Section "Creating Environment Configuration"
    
    if (-not (Test-Path ".env")) {
        $envContent = @"
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
"@
        Set-Content -Path ".env" -Value $envContent
        Write-Status "Created .env file with default configuration"
        Write-Warning "Please update AWS credentials in .env file before running"
    }
    else {
        Write-Status ".env file already exists"
    }
}

# Create directory structure
function New-DirectoryStructure {
    Write-Section "Creating Directory Structure"
    
    $directories = @(
        "data\input",
        "data\output", 
        "data\processed",
        "data\raw",
        "models",
        "checkpoints",
        "logs",
        "notebooks",
        "tests"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Status "Created directory: $dir"
        }
        else {
            Write-Status "Directory already exists: $dir"
        }
    }
}

# Install Python dependencies
function Install-PythonDependencies {
    Write-Section "Installing Python Dependencies"
    
    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        Write-Status "Creating Python virtual environment..."
        & $PythonPath -m venv venv
    }
    
    # Activate virtual environment
    $activateScript = "venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        
        # Upgrade pip
        Write-Status "Upgrading pip..."
        python -m pip install --upgrade pip
        
        # Install dependencies for each component
        $components = @("kafka_producer", "flink_processor", "spark_jobs", "ml_training", "ml_inference", "storage_utils", "dashboard")
        
        foreach ($component in $components) {
            $requirementsFile = "$component\requirements.txt"
            if (Test-Path $requirementsFile) {
                Write-Status "Installing dependencies for $component..."
                pip install -r $requirementsFile
            }
        }
        
        # Install additional development dependencies
        Write-Status "Installing development dependencies..."
        pip install jupyter notebook ipykernel pytest black flake8
        
        deactivate
        Write-Status "Python dependencies installed"
    }
    else {
        Write-Error "Failed to create virtual environment"
    }
}

# Configure Docker for WSL
function Set-DockerConfiguration {
    Write-Section "Configuring Docker for WSL"
    
    Write-Status "Checking Docker Desktop WSL integration..."
    
    # Check if Docker is running
    try {
        docker info | Out-Null
        Write-Status "Docker is running"
    }
    catch {
        Write-Warning "Docker is not running. Please start Docker Desktop."
        Write-Warning "Make sure WSL2 integration is enabled in Docker Desktop settings."
    }
    
    Write-Status "Docker configuration check complete"
    Write-Warning "Ensure WSL2 integration is enabled in Docker Desktop > Settings > Resources > WSL Integration"
}

# Build Docker images
function Build-DockerImages {
    Write-Section "Building Docker Images"
    
    # Build MLflow image
    if (Test-Path "mlflow_tracking\Dockerfile") {
        Write-Status "Building MLflow image..."
        docker build -t av-mlflow mlflow_tracking\
    }
    
    # Build Kafka producer image
    if (Test-Path "kafka_producer\Dockerfile") {
        Write-Status "Building Kafka producer image..."
        docker build -t av-kafka-producer kafka_producer\
    }
    
    Write-Status "Docker images built successfully"
}

# Start core services
function Start-CoreServices {
    Write-Section "Starting Core Services"
    
    Write-Status "Starting infrastructure services..."
    docker-compose up -d zookeeper kafka namenode datanode spark-master spark-worker-1 mlflow redis
    
    Write-Status "Waiting for services to be ready..."
    Start-Sleep -Seconds 30
    
    # Check service health
    Test-ServiceHealth
}

# Check service health
function Test-ServiceHealth {
    Write-Section "Checking Service Health"
    
    $services = @(
        @{Name = "Kafka"; Host = "localhost"; Port = 9092 },
        @{Name = "Kafka UI"; Host = "localhost"; Port = 8080 },
        @{Name = "HDFS NameNode"; Host = "localhost"; Port = 9870 },
        @{Name = "Spark Master"; Host = "localhost"; Port = 8081 },
        @{Name = "MLflow"; Host = "localhost"; Port = 5000 },
        @{Name = "Flink JobManager"; Host = "localhost"; Port = 8082 }
    )
    
    foreach ($service in $services) {
        try {
            $connection = Test-NetConnection -ComputerName $service.Host -Port $service.Port -WarningAction SilentlyContinue
            if ($connection.TcpTestSucceeded) {
                Write-Status "$($service.Name) is healthy"
            }
            else {
                Write-Warning "$($service.Name) is not responding on $($service.Host):$($service.Port)"
            }
        }
        catch {
            Write-Warning "$($service.Name) health check failed"
        }
    }
}

# Create Kafka topics
function New-KafkaTopics {
    Write-Section "Creating Kafka Topics"
    
    $topics = @("sensor_data", "processed_sensor_data", "inference_results")
    
    foreach ($topic in $topics) {
        Write-Status "Creating Kafka topic: $topic"
        docker exec kafka kafka-topics --create `
            --bootstrap-server kafka:29092 `
            --replication-factor 1 `
            --partitions 3 `
            --topic $topic `
            --if-not-exists 2>$null
    }
}

# Initialize HDFS directories
function Initialize-HDFS {
    Write-Section "Initializing HDFS Directories"
    
    $hdfsDirectories = @("/data", "/data/raw", "/data/processed", "/data/streaming", "/models", "/checkpoints")
    
    foreach ($dir in $hdfsDirectories) {
        Write-Status "Creating HDFS directory: $dir"
        docker exec namenode hdfs dfs -mkdir -p $dir 2>$null
        docker exec namenode hdfs dfs -chmod 755 $dir 2>$null
    }
}

# Create sample data
function New-SampleData {
    Write-Section "Creating Sample Data"
    
    $activateScript = "venv\Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        & $activateScript
        
        # Create a simple data generator script
        $sampleDataScript = @"
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
"@
        
        Set-Content -Path "generate_sample_data.py" -Value $sampleDataScript
        python generate_sample_data.py
        Remove-Item "generate_sample_data.py"
        
        deactivate
        Write-Status "Sample data created"
    }
}

# Create utility scripts
function New-UtilityScripts {
    Write-Section "Creating Utility Scripts"
    
    # Start script (PowerShell)
    $startScript = @"
# Start Autonomous Vehicle Pipeline
Write-Host "Starting Autonomous Vehicle Pipeline..." -ForegroundColor Green

# Load environment variables
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

# Start all services
docker-compose up -d

Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host "Pipeline started successfully!" -ForegroundColor Green
Write-Host "Access points:" -ForegroundColor Cyan
Write-Host "- Kafka UI: http://localhost:8080"
Write-Host "- HDFS: http://localhost:9870"
Write-Host "- Spark: http://localhost:8081"
Write-Host "- MLflow: http://localhost:5000"
Write-Host "- Flink: http://localhost:8082"
Write-Host "- Dashboard: http://localhost:8050"
Write-Host "- Grafana: http://localhost:3000"
"@
    
    # Stop script (PowerShell)
    $stopScript = @"
# Stop Autonomous Vehicle Pipeline
Write-Host "Stopping Autonomous Vehicle Pipeline..." -ForegroundColor Yellow
docker-compose down
Write-Host "Pipeline stopped." -ForegroundColor Green
"@
    
    # Status script (PowerShell)
    $statusScript = @"
# Autonomous Vehicle Pipeline Status
Write-Host "Autonomous Vehicle Pipeline Status:" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
docker-compose ps
"@
    
    Set-Content -Path "start_pipeline.ps1" -Value $startScript
    Set-Content -Path "stop_pipeline.ps1" -Value $stopScript
    Set-Content -Path "status_pipeline.ps1" -Value $statusScript
    
    Write-Status "PowerShell utility scripts created"
}

# Display final instructions
function Show-FinalInstructions {
    Write-Section "Setup Complete!"
    
    Write-ColorOutput "🎉 Autonomous Vehicle Simulation Data Analysis pipeline setup complete!" $Green
    Write-Output ""
    Write-Output "Next steps:"
    Write-Output "1. Update AWS credentials in .env file"
    Write-Output "2. Start the pipeline: .\start_pipeline.ps1"
    Write-Output "3. Access the dashboard: http://localhost:8050"
    Write-Output ""
    Write-Output "Service URLs:"
    Write-Output "- Kafka UI: http://localhost:8080"
    Write-Output "- HDFS NameNode: http://localhost:9870"
    Write-Output "- Spark Master: http://localhost:8081"
    Write-Output "- MLflow: http://localhost:5000"
    Write-Output "- Flink JobManager: http://localhost:8082"
    Write-Output "- Grafana: http://localhost:3000 (admin/admin)"
    Write-Output "- Dashboard: http://localhost:8050"
    Write-Output ""
    Write-Output "PowerShell utility commands:"
    Write-Output "- Start: .\start_pipeline.ps1"
    Write-Output "- Stop: .\stop_pipeline.ps1"
    Write-Output "- Status: .\status_pipeline.ps1"
    Write-Output ""
    Write-ColorOutput "Note: Make sure Docker Desktop is running and WSL2 integration is enabled!" $Yellow
}

# Main execution
function Main {
    Write-ColorOutput "🚗 Setting up Autonomous Vehicle Simulation Data Analysis Pipeline..." $Blue
    Write-ColorOutput "===============================================================================" $Blue
    
    if (-not $SkipPrerequisites) {
        Test-Prerequisites
    }
    
    New-EnvironmentFile
    New-DirectoryStructure
    Install-PythonDependencies
    
    if (-not $SkipDocker) {
        Set-DockerConfiguration
        Build-DockerImages
        Start-CoreServices
        New-KafkaTopics
        Initialize-HDFS
    }
    
    New-SampleData
    New-UtilityScripts
    Show-FinalInstructions
}

# Run main function
try {
    Main
}
catch {
    Write-Error "Setup failed: $($_.Exception.Message)"
    Write-Error "Stack trace: $($_.Exception.StackTrace)"
    exit 1
}
