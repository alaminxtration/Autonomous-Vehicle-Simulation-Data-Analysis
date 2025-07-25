# Setup Guide

## Prerequisites

- Python 3.8 or higher
- Docker Desktop (for containerized deployment)
- Git (for version control)

## Installation Methods

### Method 1: Direct Installation from GitHub

```bash
pip install git+https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis.git
```

### Method 2: Local Development Setup

```bash
# Clone the repository
git clone https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis.git
cd Autonomous-Vehicle-Simulation-Data-Analysis

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Configuration

### Python Environment

1. Ensure Python 3.8+ is installed:
```bash
python --version
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

### Docker Environment

1. Install Docker Desktop
2. Verify Docker is running:
```bash
docker --version
docker-compose --version
```

3. Start required services:
```bash
docker-compose up -d
```

## Verification

### Test Python Installation
```bash
python -c "from autonomous_vehicle_simulation import VehicleSimulation; print('✅ Installation successful!')"
```

### Test Docker Services
```bash
docker-compose ps
```

Expected output should show Redis and Grafana containers running.

### Access Services

- **Dashboard**: http://localhost:8050
- **Grafana**: http://localhost:3000 (admin/admin)
- **Redis**: localhost:6379

## Troubleshooting

### Common Issues

1. **ImportError**: Ensure virtual environment is activated
2. **Docker connection**: Verify Docker Desktop is running
3. **Port conflicts**: Check if ports 8050, 3000, 6379 are available

### Getting Help

- Check [GitHub Issues](https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis/issues)
- Review [Complete User Guide](../COMPLETE_USER_GUIDE.md)
- See [Troubleshooting Guide](#troubleshooting-common-issues) below

## Next Steps

1. Follow [How to Use Guide](../HOW_TO_USE.md) for usage examples
2. Check [Deployment Guide](deployment.md) for production setup
3. Review [Contributing Guidelines](../CONTRIBUTING.md) to contribute

## Troubleshooting Common Issues

### Python Environment Issues

**Problem**: `ModuleNotFoundError`
```bash
# Solution: Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

**Problem**: `Permission denied`
```bash
# Solution: Use virtual environment or --user flag
pip install --user -r requirements.txt
```

### Docker Issues

**Problem**: Container won't start
```bash
# Check Docker status
docker ps -a
docker logs <container_name>

# Restart services
docker-compose down
docker-compose up -d
```

**Problem**: Port already in use
```bash
# Find process using port
netstat -ano | findstr :8050  # Windows
lsof -i :8050  # macOS/Linux

# Stop the process or change port in docker-compose.yml
```

### Dashboard Issues

**Problem**: Dashboard not loading
1. Check Python server is running
2. Verify port 8050 is accessible
3. Check browser console for errors

**Problem**: No data showing
1. Verify Redis is running: `docker ps`
2. Check simulation is generating data
3. Restart services: `docker-compose restart`

### Performance Issues

**Problem**: Slow simulation
1. Reduce number of vehicles
2. Increase update intervals
3. Check system resources

**Problem**: High memory usage
1. Limit simulation duration
2. Optimize data retention
3. Monitor with: `docker stats`
