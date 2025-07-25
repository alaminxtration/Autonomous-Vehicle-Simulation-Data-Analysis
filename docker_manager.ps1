# Autonomous Vehicle Simulation - Docker Management Script
# Handles Docker deployment with network issue workarounds

Write-Host ""
Write-Host "🚗 AUTONOMOUS VEHICLE SIMULATION - DOCKER MANAGER" -ForegroundColor Blue
Write-Host "=================================================" -ForegroundColor Blue
Write-Host ""

function Test-DockerConnectivity {
    Write-Host "Testing Docker Hub connectivity..." -ForegroundColor Yellow
    try {
        $result = docker pull hello-world 2>&1
        if ($LASTEXITCODE -eq 0) {
            docker rmi hello-world 2>$null
            Write-Host "✅ Docker Hub connectivity: OK" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "❌ Docker Hub connectivity: FAILED" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Docker Hub connectivity: FAILED" -ForegroundColor Red
        return $false
    }
}

function Show-CurrentStatus {
    Write-Host ""
    Write-Host "📊 CURRENT PROJECT STATUS:" -ForegroundColor Cyan
    Write-Host "=" * 40 -ForegroundColor Cyan
    
    # Check if dashboard is running
    $dashboardRunning = $false
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8050" -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $dashboardRunning = $true
            Write-Host "✅ Dashboard: RUNNING (http://localhost:8050)" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "❌ Dashboard: NOT RUNNING" -ForegroundColor Red
    }
    
    # Check data directory
    $dataExists = Test-Path "data/simulation"
    if ($dataExists) {
        $dataFiles = Get-ChildItem "data/simulation" -Filter "*.json" | Measure-Object
        Write-Host "✅ Data: $($dataFiles.Count) simulation files available" -ForegroundColor Green
    }
    else {
        Write-Host "⚠️  Data: No simulation data found" -ForegroundColor Yellow
    }
    
    # Check Python environment
    if (Test-Path ".venv/Scripts/python.exe") {
        Write-Host "✅ Python Environment: READY" -ForegroundColor Green
    }
    else {
        Write-Host "❌ Python Environment: NOT FOUND" -ForegroundColor Red
    }
    
    return $dashboardRunning
}

function Start-LocalServices {
    Write-Host ""
    Write-Host "🚀 STARTING LOCAL SERVICES..." -ForegroundColor Green
    Write-Host ""
    
    # Generate fresh simulation data
    Write-Host "Generating fresh simulation data..." -ForegroundColor Yellow
    & .venv\Scripts\python.exe simulation\vehicle_simulation.py 60
    
    Write-Host ""
    Write-Host "Starting dashboard..." -ForegroundColor Yellow
    Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "dashboard\simple_dashboard.py" -WindowStyle Hidden
    
    Start-Sleep -Seconds 5
    
    Write-Host ""
    Write-Host "✅ LOCAL SERVICES STARTED!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🌐 ACCESS URLS:" -ForegroundColor Cyan
    Write-Host "Dashboard: http://localhost:8050" -ForegroundColor White
    Write-Host ""
}

function Show-DockerAlternatives {
    Write-Host ""
    Write-Host "🐳 DOCKER ALTERNATIVES:" -ForegroundColor Yellow
    Write-Host "=" * 40 -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Fix Network and Retry Docker" -ForegroundColor Cyan
    Write-Host "• Check internet connection" -ForegroundColor White
    Write-Host "• Restart Docker Desktop" -ForegroundColor White
    Write-Host "• Try: docker pull python:3.12-slim" -ForegroundColor White
    Write-Host "• Then run: docker-compose up" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 2: Use Redis Only (Minimal Docker)" -ForegroundColor Cyan
    Write-Host "• docker run -d -p 6379:6379 redis:7-alpine" -ForegroundColor White
    Write-Host "• Provides caching for the Python services" -ForegroundColor White
    Write-Host ""
    Write-Host "Option 3: Offline Docker Images" -ForegroundColor Cyan
    Write-Host "• Use pre-downloaded images when network improves" -ForegroundColor White
    Write-Host "• docker save/load for image transfer" -ForegroundColor White
}

function Main {
    Write-Host "Current working directory: $(Get-Location)" -ForegroundColor Gray
    
    # Show current status
    $dashboardRunning = Show-CurrentStatus
    
    # Test Docker connectivity
    $dockerConnected = Test-DockerConnectivity
    
    Write-Host ""
    Write-Host "🎯 RECOMMENDED ACTION:" -ForegroundColor Green
    Write-Host "=" * 40 -ForegroundColor Green
    
    if ($dashboardRunning) {
        Write-Host "✅ Your project is already running perfectly!" -ForegroundColor Green
        Write-Host "   Dashboard: http://localhost:8050" -ForegroundColor White
        Write-Host ""
        Write-Host "🔄 Generate more data:" -ForegroundColor Cyan
        Write-Host "   .venv\Scripts\python.exe simulation\vehicle_simulation.py 120" -ForegroundColor White
        Write-Host ""
        
        if ($dockerConnected) {
            Write-Host "💡 Docker is available. Start full stack? (y/n)" -ForegroundColor Yellow
            $choice = Read-Host
            if ($choice -eq "y" -or $choice -eq "Y") {
                Write-Host ""
                Write-Host "Starting full Docker stack..." -ForegroundColor Green
                docker-compose up -d
            }
        }
        else {
            Write-Host "⚠️  Docker has network issues, but local version works great!" -ForegroundColor Yellow
            Show-DockerAlternatives
        }
    }
    else {
        if ($dockerConnected) {
            Write-Host "🐳 Docker is available. Choose option:" -ForegroundColor Green
            Write-Host "1. Start full Docker stack" -ForegroundColor White
            Write-Host "2. Start local services only" -ForegroundColor White
            $choice = Read-Host "Enter choice (1/2)"
            
            if ($choice -eq "1") {
                Write-Host ""
                Write-Host "Starting Docker stack..." -ForegroundColor Green
                docker-compose up -d
            }
            else {
                Start-LocalServices
            }
        }
        else {
            Write-Host "🚀 Starting local services (Docker unavailable)..." -ForegroundColor Green
            Start-LocalServices
        }
    }
    
    Write-Host ""
    Write-Host "📋 MANAGEMENT COMMANDS:" -ForegroundColor Cyan
    Write-Host "Stop all: docker-compose down" -ForegroundColor White
    Write-Host "View logs: docker-compose logs -f" -ForegroundColor White
    Write-Host "Test components: .\test_components_nokafka.ps1" -ForegroundColor White
    Write-Host ""
}

# Run the main function
Main
