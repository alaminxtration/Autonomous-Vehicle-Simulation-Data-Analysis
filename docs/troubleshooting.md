# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Problem: `pip install` fails with permission error
**Solution**:
```bash
# Use virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# OR use --user flag
pip install --user -r requirements.txt
```

#### Problem: `ModuleNotFoundError` after installation
**Solution**:
```bash
# Verify virtual environment is activated
# You should see (.venv) in your prompt

# If not activated:
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Reinstall if needed
pip install -r requirements.txt
```

#### Problem: GitHub installation fails
**Solution**:
```bash
# Use HTTPS instead of SSH
pip install git+https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis.git

# If still fails, clone and install locally
git clone https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis.git
cd Autonomous-Vehicle-Simulation-Data-Analysis
pip install -e .
```

### Docker Issues

#### Problem: Docker containers won't start
**Solution**:
```bash
# Check Docker is running
docker --version

# Check container status
docker-compose ps

# View container logs
docker-compose logs redis
docker-compose logs grafana

# Restart services
docker-compose down
docker-compose up -d
```

#### Problem: Port already in use (8050, 3000, 6379)
**Solution**:
```bash
# Find what's using the port
netstat -ano | findstr :8050  # Windows
lsof -i :8050  # macOS/Linux

# Kill the process or change port in docker-compose.yml
# Edit docker-compose.yml and change port mapping:
# ports:
#   - "8051:8050"  # Use 8051 instead
```

#### Problem: Redis connection refused
**Solution**:
```bash
# Check Redis container
docker ps | grep redis

# Restart Redis
docker-compose restart redis

# Check Redis logs
docker-compose logs redis

# Test Redis connection
docker exec -it <redis-container-name> redis-cli ping
```

### Dashboard Issues

#### Problem: Dashboard won't load at localhost:8050
**Solution**:
1. Check if Python process is running
2. Verify no firewall blocking port 8050
3. Try accessing via 127.0.0.1:8050
4. Check browser console for errors

#### Problem: Dashboard shows "No data available"
**Solution**:
```bash
# Check if simulation is running
# Verify Redis container is up
docker-compose ps

# Check if data is in Redis
docker exec -it <redis-container-name> redis-cli keys "*"

# Restart simulation
python -m dashboard.app
```

#### Problem: Dashboard is slow or unresponsive
**Solution**:
1. Reduce number of vehicles in simulation
2. Increase update intervals
3. Clear browser cache
4. Check system resources (RAM/CPU)

### Simulation Issues

#### Problem: Simulation crashes or freezes
**Solution**:
```bash
# Check available memory
# Reduce simulation parameters
sim = VehicleSimulation(
    num_vehicles=5,      # Reduce from higher number
    duration=30,         # Reduce duration
    update_interval=0.5  # Increase interval
)

# Check for infinite loops in code
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Problem: Poor simulation performance
**Solution**:
1. **Reduce complexity**:
   - Fewer vehicles
   - Shorter duration
   - Larger update intervals

2. **Optimize system**:
   - Close unnecessary applications
   - Increase virtual memory
   - Use SSD storage

3. **Code optimization**:
   - Profile bottlenecks
   - Use vectorized operations
   - Implement data sampling

### Data Issues

#### Problem: Data export fails
**Solution**:
```python
# Check file permissions
import os
os.access("output_path", os.W_OK)

# Use absolute paths
export_data(data, r"C:\full\path\to\file.csv")

# Try different formats
export_data(data, "output.json", format="json")
```

#### Problem: Large memory usage
**Solution**:
```python
# Implement data sampling
data_sample = data.sample(frac=0.1)  # Use 10% of data

# Use data streaming
for chunk in pd.read_csv("large_file.csv", chunksize=1000):
    process_chunk(chunk)

# Clear variables when done
del large_data_variable
import gc
gc.collect()
```

### Grafana Issues

#### Problem: Can't access Grafana at localhost:3000
**Solution**:
```bash
# Check Grafana container
docker-compose logs grafana

# Restart Grafana
docker-compose restart grafana

# Check port mapping
docker-compose ps

# Default login: admin/admin
```

#### Problem: Grafana shows no data
**Solution**:
1. Configure data source (Redis/Prometheus)
2. Check dashboard queries
3. Verify time range settings
4. Import provided dashboard configurations

### Performance Optimization

#### System Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB RAM, 4 CPU cores
- **Storage**: 2GB free space

#### Optimization Tips
1. **Python optimizations**:
   ```python
   # Use NumPy for calculations
   import numpy as np
   positions = np.array(vehicle_positions)
   
   # Vectorize operations
   distances = np.sqrt(np.sum((positions[:, None] - positions) ** 2, axis=2))
   ```

2. **Docker optimizations**:
   ```yaml
   # In docker-compose.yml, limit resources
   services:
     redis:
       mem_limit: 512m
       cpus: 0.5
   ```

3. **Data optimizations**:
   ```python
   # Use appropriate data types
   df['vehicle_id'] = df['vehicle_id'].astype('int32')
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   
   # Implement data compression
   df.to_csv('data.csv.gz', compression='gzip')
   ```

### Network Issues

#### Problem: Cannot access from other machines
**Solution**:
```bash
# Change host binding in dashboard
run_dashboard(host="0.0.0.0", port=8050)

# Update Docker compose
services:
  dashboard:
    ports:
      - "0.0.0.0:8050:8050"
```

#### Problem: Slow network responses
**Solution**:
1. Check network bandwidth
2. Implement data compression
3. Use WebSocket connections
4. Cache frequently accessed data

### Development Issues

#### Problem: Code changes not reflected
**Solution**:
```bash
# Install in development mode
pip install -e .

# Restart services after code changes
docker-compose restart

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -delete
```

#### Problem: Import errors in development
**Solution**:
```python
# Add project root to Python path
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# Use relative imports properly
from .dashboard import run_dashboard  # Within package
from autonomous_vehicle_simulation import VehicleSimulation  # External
```

## Getting Help

### Log Collection
When reporting issues, include:

1. **System info**:
   ```bash
   python --version
   docker --version
   pip list | grep autonomous
   ```

2. **Error logs**:
   ```bash
   # Python errors
   python simulation.py 2>&1 | tee error.log
   
   # Docker logs
   docker-compose logs > docker.log
   ```

3. **System resources**:
   ```bash
   # Windows
   wmic OS get TotalVisibleMemorySize,FreePhysicalMemory
   
   # Linux/macOS
   free -h
   top -l 1 | head -10
   ```

### Support Channels

1. **GitHub Issues**: [Report bugs and feature requests](https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis/issues)
2. **Discussions**: [Ask questions and share ideas](https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis/discussions)
3. **Documentation**: [Check user guides](../README.md)

### Before Reporting Issues

1. Check this troubleshooting guide
2. Search existing GitHub issues
3. Verify system requirements
4. Test with minimal example
5. Collect relevant logs and error messages

### Issue Template

When reporting issues, please include:

```
**Environment:**
- OS: [Windows 10/macOS/Linux]
- Python version: [3.8/3.9/3.10/3.11]
- Docker version: [if using Docker]

**Issue Description:**
[Clear description of the problem]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Behavior:**
[What you expected to happen]

**Actual Behavior:**
[What actually happened]

**Error Messages:**
[Include full error traceback]

**Additional Context:**
[Any other relevant information]
```
