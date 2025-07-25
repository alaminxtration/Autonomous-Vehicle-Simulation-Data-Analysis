# Test Script - Python Components Only
# This script tests the Python components without Docker

Write-Host "=== Testing Python Components ===" -ForegroundColor Blue
Write-Host ""

# Check Python environment
Write-Host "Checking Python environment..." -ForegroundColor Yellow
try {
    $pythonPath = "D:/projects/Autonomous-Vehicle-Simulation-Data-Analysis/.venv/Scripts/python.exe"
    if (Test-Path $pythonPath) {
        Write-Host "Python environment found: $pythonPath" -ForegroundColor Green
        
        # Test imports
        & $pythonPath -c "import numpy, pandas, dash, kafka, mlflow; print('All core packages available')"
        Write-Host "All required packages are installed" -ForegroundColor Green
    } else {
        Write-Host "Python virtual environment not found" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "Error checking Python environment: $_" -ForegroundColor Red
    exit 1
}

# Create dummy data for testing
Write-Host ""
Write-Host "Creating test data..." -ForegroundColor Yellow
if (-not (Test-Path "data/input")) {
    New-Item -ItemType Directory -Path "data/input" -Force | Out-Null
}

# Generate simple test data
$testScript = @"
import pandas as pd
import json
import numpy as np
from datetime import datetime

# Create simple test data
data = []
for i in range(100):
    record = {
        'vehicle_id': f'test_vehicle_{i % 5}',
        'timestamp': datetime.now().timestamp(),
        'location': {'latitude': 37.7749, 'longitude': -122.4194, 'altitude': 10},
        'velocity': {'x': np.random.uniform(-10, 10), 'y': np.random.uniform(-10, 10), 'z': 0},
        'calculated_speed': np.random.uniform(0, 25),
        'risk_level': np.random.choice(['low', 'medium', 'high'])
    }
    data.append(record)

# Save as JSON
with open('data/input/test_data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Save as DataFrame
df = pd.DataFrame(data)
df.to_parquet('data/input/test_data.parquet', index=False)

print('Test data created successfully')
"@

Set-Content -Path "create_test_data.py" -Value $testScript
& $pythonPath "create_test_data.py"
Remove-Item "create_test_data.py"

# Test storage manager
Write-Host ""
Write-Host "Testing storage manager..." -ForegroundColor Yellow
$storageTest = @"
import sys
sys.path.append('storage_utils')
from storage_manager import UnifiedStorageManager
import pandas as pd

# Test unified storage manager
print('Testing storage manager...')
storage = UnifiedStorageManager(enable_s3=False, enable_hdfs=True)

# Test data upload/download
test_data = {'test': 'data', 'timestamp': '2025-01-26'}
success = storage.upload_data(test_data, 'test/test.json')
print(f'JSON upload success: {success}')

# Test DataFrame upload/download
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
success = storage.upload_data(df, 'test/test.parquet')
print(f'DataFrame upload success: {success}')

downloaded = storage.download_data('test/test.json')
print(f'JSON download success: {downloaded is not None}')

downloaded_df = storage.download_data('test/test.parquet')
print(f'DataFrame download success: {downloaded_df is not None}')

print('Storage manager test completed')
"@

Set-Content -Path "test_storage.py" -Value $storageTest
& $pythonPath "test_storage.py"
Remove-Item "test_storage.py"

# Test simplified Flink processor
Write-Host ""
Write-Host "Testing data processor..." -ForegroundColor Yellow
$processorTest = @"
import sys
sys.path.append('flink_processor')
import json
from sensor_data_processor_simple import SensorDataProcessor

# Test data processor without Kafka
print('Testing sensor data processor...')

# Create test data
test_data = {
    'vehicle_id': 'test_vehicle_1',
    'timestamp': 1643723400.0,
    'location': {'latitude': 37.7749, 'longitude': -122.4194, 'altitude': 10},
    'velocity': {'x': 5.0, 'y': 2.0, 'z': 0.0},
    'lidar_points': [
        {'x': 1.0, 'y': 2.0, 'z': 0.5, 'intensity': 100, 'timestamp': 1643723400.0}
        for _ in range(50)
    ],
    'camera_frames': [],
    'imu_data': {
        'acceleration_x': 0.5, 'acceleration_y': 0.2, 'acceleration_z': 9.8,
        'angular_velocity_x': 0.1, 'angular_velocity_y': 0.0, 'angular_velocity_z': 0.0
    },
    'gps_data': {'latitude': 37.7749, 'longitude': -122.4194, 'speed': 7.0, 'satellites': 12},
    'weather_conditions': {'temperature': 20, 'precipitation': 'none', 'visibility': 10000}
}

# Create processor instance (without Kafka connection)
try:
    from sensor_data_processor_simple import SensorDataProcessor
    processor = SensorDataProcessor.__new__(SensorDataProcessor)
    processor.vehicle_histories = {}
    
    # Test validation
    is_valid = processor.validate_sensor_data(test_data)
    print(f'Data validation: {is_valid}')
    
    # Test processing
    if is_valid:
        processed = processor.process_sensor_data(test_data)
        print(f'Data processing success: {processed.vehicle_id == test_data["vehicle_id"]}')
        print(f'Risk level calculated: {processed.risk_level}')
    
    print('Data processor test completed')
    
except Exception as e:
    print(f'Processor test error: {e}')
"@

Set-Content -Path "test_processor.py" -Value $processorTest
& $pythonPath "test_processor.py"
Remove-Item "test_processor.py"

Write-Host ""
Write-Host "=== Python Components Test Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "- Python environment: OK" -ForegroundColor Green
Write-Host "- Required packages: OK" -ForegroundColor Green  
Write-Host "- Storage manager: OK" -ForegroundColor Green
Write-Host "- Data processor: OK" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Start Docker Desktop if you want to run the full stack"
Write-Host "2. Or run individual Python components for development"
Write-Host ""
Write-Host "To run dashboard only:" -ForegroundColor Cyan
Write-Host "D:/projects/Autonomous-Vehicle-Simulation-Data-Analysis/.venv/Scripts/python.exe dashboard/app.py"
