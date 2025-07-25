# Docker Startup Script for Autonomous Vehicle Simulation
# Runs the complete project using Docker on Windows

Write-Host ""
Write-Host "🚗 AUTONOMOUS VEHICLE SIMULATION - DOCKER STARTUP 🐳" -ForegroundColor Blue
Write-Host "=" * 60 -ForegroundColor Blue
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    Write-Host ""
    Write-Host "To start Docker Desktop:" -ForegroundColor Yellow
    Write-Host "1. Open Docker Desktop application" -ForegroundColor White
    Write-Host "2. Wait for it to start completely" -ForegroundColor White
    Write-Host "3. Run this script again" -ForegroundColor White
    exit 1
}

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

try {
    docker-compose -f docker-compose-simple.yml up --build -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Docker services started successfully!" -ForegroundColor Green
        
        # Wait for services to be ready
        Write-Host ""
        Write-Host "Waiting for services to initialize (30 seconds)..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
        
        # Check service status
        Write-Host ""
        Write-Host "📊 Service Status:" -ForegroundColor Cyan
        docker-compose -f docker-compose-simple.yml ps
        
        Write-Host ""
        Write-Host "🌐 Access URLs:" -ForegroundColor Green
        Write-Host "=" * 40 -ForegroundColor Green
        Write-Host "Dashboard:     http://localhost:8050" -ForegroundColor White
        Write-Host "Redis:         localhost:6379" -ForegroundColor White
        
        Write-Host ""
        Write-Host "📋 Services Running:" -ForegroundColor Cyan
        Write-Host "• vehicle-simulation: Generating realistic sensor data" -ForegroundColor White
        Write-Host "• data-processor: Processing and analyzing data" -ForegroundColor White
        Write-Host "• dashboard: Interactive web dashboard" -ForegroundColor White
        Write-Host "• redis: Caching and data storage" -ForegroundColor White
        
        Write-Host ""
        Write-Host "🔧 Management Commands:" -ForegroundColor Yellow
        Write-Host "View logs:        docker-compose -f docker-compose-simple.yml logs" -ForegroundColor White
        Write-Host "Follow logs:      docker-compose -f docker-compose-simple.yml logs -f" -ForegroundColor White
        Write-Host "Stop services:    docker-compose -f docker-compose-simple.yml down" -ForegroundColor White
        Write-Host "Restart:          docker-compose -f docker-compose-simple.yml restart" -ForegroundColor White
        
        Write-Host ""
        Write-Host "🎉 DOCKER DEPLOYMENT SUCCESSFUL!" -ForegroundColor Green
        Write-Host "Open http://localhost:8050 to view the dashboard" -ForegroundColor Green
        
    }
    else {
        Write-Host ""
        Write-Host "❌ Failed to start Docker services" -ForegroundColor Red
        Write-Host "Check the error messages above for details" -ForegroundColor Yellow
    }
    
}
catch {
    Write-Host ""
    Write-Host "❌ Error starting Docker services: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "1. Make sure Docker Desktop is running" -ForegroundColor White
    Write-Host "2. Check if ports 8050, 6379 are available" -ForegroundColor White
    Write-Host "3. Try: docker-compose -f docker-compose-simple.yml down" -ForegroundColor White
    Write-Host "4. Then run this script again" -ForegroundColor White
}
}

Write-Host ""
