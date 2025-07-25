# Enhanced Docker Startup Script with Docker Desktop Management

Write-Host ""
Write-Host "AUTONOMOUS VEHICLE SIMULATION - DOCKER STARTUP" -ForegroundColor Blue
Write-Host "===============================================" -ForegroundColor Blue
Write-Host ""

# Function to check if Docker Desktop is running
function Test-DockerRunning {
    try {
        docker version | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to start Docker Desktop
function Start-DockerDesktop {
    Write-Host "Attempting to start Docker Desktop..." -ForegroundColor Yellow
    
    # Try to find Docker Desktop executable
    $dockerDesktopPaths = @(
        "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe",
        "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe",
        "${env:LOCALAPPDATA}\Programs\Docker\Docker\Docker Desktop.exe"
    )
    
    $dockerDesktopPath = $null
    foreach ($path in $dockerDesktopPaths) {
        if (Test-Path $path) {
            $dockerDesktopPath = $path
            break
        }
    }
    
    if ($dockerDesktopPath) {
        Write-Host "Found Docker Desktop at: $dockerDesktopPath" -ForegroundColor Green
        Start-Process -FilePath $dockerDesktopPath
        
        Write-Host "Waiting for Docker Desktop to start..." -ForegroundColor Yellow
        $timeout = 120 # 2 minutes timeout
        $elapsed = 0
        
        while (-not (Test-DockerRunning) -and $elapsed -lt $timeout) {
            Start-Sleep -Seconds 5
            $elapsed += 5
            Write-Host "." -NoNewline -ForegroundColor Cyan
        }
        
        Write-Host ""
        
        if (Test-DockerRunning) {
            Write-Host "Docker Desktop started successfully!" -ForegroundColor Green
            return $true
        } else {
            Write-Host "Docker Desktop failed to start within timeout" -ForegroundColor Red
            return $false
        }
    } else {
        Write-Host "Docker Desktop executable not found" -ForegroundColor Red
        Write-Host "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
        return $false
    }
}

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
if (-not (Test-DockerRunning)) {
    Write-Host "Docker is not running" -ForegroundColor Red
    
    $startDocker = Read-Host "Would you like to start Docker Desktop automatically? (y/n)"
    if ($startDocker -eq "y" -or $startDocker -eq "Y") {
        if (-not (Start-DockerDesktop)) {
            Write-Host ""
            Write-Host "Please start Docker Desktop manually and run this script again" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Host ""
        Write-Host "Please start Docker Desktop manually and run this script again" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "Docker is running" -ForegroundColor Green
}

# Alternative: Run without Docker if Docker is not available
Write-Host ""
$useDocker = Read-Host "Docker is available. Use Docker deployment? (y/n, 'n' will run locally)"

if ($useDocker -eq "n" -or $useDocker -eq "N") {
    Write-Host ""
    Write-Host "Running project locally without Docker..." -ForegroundColor Cyan
    Write-Host ""
    
    # Run the local version
    .\test_components_nokafka.ps1
    
    Write-Host ""
    Write-Host "Starting local dashboard..." -ForegroundColor Yellow
    .venv\Scripts\python.exe dashboard\simple_dashboard.py
    
    exit 0
}

# Continue with Docker deployment
Write-Host ""
Write-Host "Proceeding with Docker deployment..." -ForegroundColor Green

# Clean up any existing containers
Write-Host ""
Write-Host "Cleaning up existing containers..." -ForegroundColor Yellow
docker-compose -f docker-compose-simple.yml down --remove-orphans 2>$null

# Create data directories
Write-Host ""
Write-Host "Creating data directories..." -ForegroundColor Yellow
if (-not (Test-Path "data")) { New-Item -ItemType Directory -Path "data" -Force | Out-Null }
if (-not (Test-Path "data/simulation")) { New-Item -ItemType Directory -Path "data/simulation" -Force | Out-Null }
if (-not (Test-Path "data/hdfs_simulation")) { New-Item -ItemType Directory -Path "data/hdfs_simulation" -Force | Out-Null }

# Build and start services
Write-Host ""
Write-Host "Building and starting Docker services..." -ForegroundColor Yellow
Write-Host "This may take a few minutes on first run..." -ForegroundColor Cyan

docker-compose -f docker-compose-simple.yml up --build -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Docker services started successfully!" -ForegroundColor Green
    
    # Wait for services to be ready
    Write-Host ""
    Write-Host "Waiting for services to initialize (30 seconds)..." -ForegroundColor Yellow
    Start-Sleep -Seconds 30
    
    # Check service status
    Write-Host ""
    Write-Host "Service Status:" -ForegroundColor Cyan
    docker-compose -f docker-compose-simple.yml ps
    
    Write-Host ""
    Write-Host "Access URLs:" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Dashboard:     http://localhost:8050" -ForegroundColor White
    Write-Host "Redis:         localhost:6379" -ForegroundColor White
    
    Write-Host ""
    Write-Host "DOCKER DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
    Write-Host "Open http://localhost:8050 to view the dashboard" -ForegroundColor Green
    
} else {
    Write-Host ""
    Write-Host "Failed to start Docker services" -ForegroundColor Red
    Write-Host ""
    Write-Host "Would you like to run locally instead? (y/n)" -ForegroundColor Yellow
    $runLocal = Read-Host
    
    if ($runLocal -eq "y" -or $runLocal -eq "Y") {
        Write-Host ""
        Write-Host "Running project locally..." -ForegroundColor Cyan
        .\test_components_nokafka.ps1
        Write-Host ""
        Write-Host "Starting local dashboard..." -ForegroundColor Yellow
        .venv\Scripts\python.exe dashboard\simple_dashboard.py
    }
}

Write-Host ""
