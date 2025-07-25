# API Reference

## Core Classes

### VehicleSimulation

Main simulation class for autonomous vehicle scenarios.

```python
from autonomous_vehicle_simulation import VehicleSimulation

# Initialize simulation
sim = VehicleSimulation(
    num_vehicles=5,           # Number of vehicles
    duration=60,              # Simulation duration in seconds
    update_interval=0.1,      # Update frequency
    scenario="highway"        # Simulation scenario
)
```

#### Methods

##### `run()`
Executes the simulation and returns collected data.

**Returns**: `pandas.DataFrame` with simulation results

**Example**:
```python
data = sim.run()
print(data.head())
```

##### `get_vehicle_data(vehicle_id)`
Retrieves data for a specific vehicle.

**Parameters**:
- `vehicle_id` (int): Vehicle identifier

**Returns**: `dict` with vehicle metrics

##### `calculate_risk_scores()`
Computes safety risk scores for all vehicles.

**Returns**: `dict` mapping vehicle_id to risk_score

### Dashboard

Interactive web dashboard for real-time monitoring.

```python
from autonomous_vehicle_simulation.dashboard import run_dashboard, DashboardConfig

# Basic usage
run_dashboard()

# Advanced configuration
config = DashboardConfig(
    port=8050,
    debug=True,
    auto_refresh=True
)
run_dashboard(config)
```

#### Functions

##### `run_dashboard(config=None)`
Launches the web dashboard application.

**Parameters**:
- `config` (DashboardConfig, optional): Dashboard configuration

**Example**:
```python
run_dashboard()  # Access at http://localhost:8050
```

##### `get_live_metrics()`
Retrieves current simulation metrics.

**Returns**: `dict` with real-time data

### RiskAssessment

Safety analysis and collision detection system.

```python
from autonomous_vehicle_simulation import RiskAssessment

risk_analyzer = RiskAssessment()
```

#### Methods

##### `detect_collisions(vehicles)`
Identifies potential vehicle collisions.

**Parameters**:
- `vehicles` (list): List of vehicle objects

**Returns**: `list` of collision events

##### `calculate_safety_score(vehicle)`
Computes safety score for a vehicle.

**Parameters**:
- `vehicle` (Vehicle): Vehicle object

**Returns**: `float` safety score (0-100)

## Data Structures

### Vehicle

Represents an autonomous vehicle in simulation.

```python
class Vehicle:
    def __init__(self, vehicle_id, x=0, y=0, speed=0):
        self.id = vehicle_id
        self.position = (x, y)
        self.speed = speed
        self.heading = 0
        self.acceleration = 0
```

**Attributes**:
- `id` (int): Unique vehicle identifier
- `position` (tuple): (x, y) coordinates
- `speed` (float): Current velocity
- `heading` (float): Direction in degrees
- `acceleration` (float): Current acceleration

### SimulationData

Container for simulation results.

```python
class SimulationData:
    def __init__(self):
        self.vehicles = []
        self.events = []
        self.metrics = {}
        self.timestamp = datetime.now()
```

**Attributes**:
- `vehicles` (list): Vehicle state history
- `events` (list): Simulation events
- `metrics` (dict): Performance metrics
- `timestamp` (datetime): Simulation start time

## Configuration

### SimulationConfig

Configuration class for simulation parameters.

```python
from autonomous_vehicle_simulation import SimulationConfig

config = SimulationConfig(
    num_vehicles=10,
    duration=120,
    scenario="urban",
    weather="clear",
    traffic_density="medium"
)

sim = VehicleSimulation(config=config)
```

**Parameters**:
- `num_vehicles` (int): Number of vehicles (1-100)
- `duration` (int): Simulation time in seconds
- `scenario` (str): "highway", "urban", "parking"
- `weather` (str): "clear", "rain", "fog"
- `traffic_density` (str): "low", "medium", "high"

### DashboardConfig

Configuration for web dashboard.

```python
from autonomous_vehicle_simulation.dashboard import DashboardConfig

config = DashboardConfig(
    port=8050,
    host="localhost",
    debug=False,
    auto_refresh=True,
    refresh_interval=1000
)
```

**Parameters**:
- `port` (int): Server port (default: 8050)
- `host` (str): Server host (default: "localhost")
- `debug` (bool): Debug mode (default: False)
- `auto_refresh` (bool): Auto-refresh data (default: True)
- `refresh_interval` (int): Refresh rate in milliseconds

## Utility Functions

### Data Export

```python
from autonomous_vehicle_simulation.utils import export_data

# Export to CSV
export_data(data, "simulation_results.csv", format="csv")

# Export to JSON
export_data(data, "simulation_results.json", format="json")
```

### Visualization

```python
from autonomous_vehicle_simulation.visualization import plot_trajectories, plot_risk_heatmap

# Plot vehicle paths
plot_trajectories(data, save_path="trajectories.png")

# Generate risk visualization
plot_risk_heatmap(risk_data, save_path="risk_map.png")
```

## Error Handling

### Common Exceptions

#### `SimulationError`
Raised when simulation encounters an error.

```python
try:
    sim.run()
except SimulationError as e:
    print(f"Simulation failed: {e}")
```

#### `ConfigurationError`
Raised for invalid configuration parameters.

```python
try:
    config = SimulationConfig(num_vehicles=-1)  # Invalid
except ConfigurationError as e:
    print(f"Invalid config: {e}")
```

#### `DataExportError`
Raised when data export fails.

```python
try:
    export_data(data, "results.csv")
except DataExportError as e:
    print(f"Export failed: {e}")
```

## Examples

### Basic Simulation

```python
from autonomous_vehicle_simulation import VehicleSimulation

# Create and run simulation
sim = VehicleSimulation(num_vehicles=5, duration=30)
data = sim.run()

# Analyze results
print(f"Simulation completed with {len(data)} data points")
print(f"Average speed: {data['speed'].mean():.2f} m/s")
```

### Advanced Usage with Risk Analysis

```python
from autonomous_vehicle_simulation import VehicleSimulation, RiskAssessment

# Setup simulation
sim = VehicleSimulation(num_vehicles=10, duration=60)
risk_analyzer = RiskAssessment()

# Run simulation
data = sim.run()

# Analyze safety
collisions = risk_analyzer.detect_collisions(sim.vehicles)
safety_scores = risk_analyzer.calculate_safety_scores(sim.vehicles)

print(f"Detected {len(collisions)} potential collisions")
print(f"Average safety score: {sum(safety_scores.values())/len(safety_scores):.2f}")
```

### Dashboard Integration

```python
from autonomous_vehicle_simulation import VehicleSimulation
from autonomous_vehicle_simulation.dashboard import run_dashboard, DashboardConfig

# Run simulation in background
sim = VehicleSimulation(num_vehicles=5, duration=300)  # 5 minute simulation
data = sim.run_async()  # Non-blocking

# Start dashboard
config = DashboardConfig(port=8050, auto_refresh=True)
run_dashboard(config)  # Access at http://localhost:8050
```
