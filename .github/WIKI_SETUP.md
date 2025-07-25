# Wiki Setup Guide

## 📚 How to Create Your GitHub Wiki

### Step 1: Enable Wiki
1. Go to your repository Settings
2. Check ✅ "Wikis" in Features section
3. Click "Create the first page"

### Step 2: Recommended Wiki Pages

#### 🏠 **Home Page** (Auto-created)
```markdown
# Autonomous Vehicle Simulation Wiki

Welcome to the comprehensive documentation wiki for the Autonomous Vehicle Simulation package!

## 📚 Quick Links
- [Installation Guide](Installation)
- [Quick Start](Quick-Start) 
- [API Reference](API-Reference)
- [Examples](Examples)
- [Troubleshooting](Troubleshooting)
- [FAQ](FAQ)

## 🚀 Getting Started
New to the project? Start with our [Quick Start Guide](Quick-Start)!

## 🤝 Need Help?
- Check our [FAQ](FAQ)
- Browse [Troubleshooting](Troubleshooting)
- Ask in [Discussions](https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis/discussions)
```

#### ⚡ **Quick-Start** Page
```markdown
# Quick Start Guide

## Installation
```bash
pip install git+https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis.git
```

## Basic Usage
```python
from autonomous_vehicle_simulation import VehicleSimulation

# Create simulation
sim = VehicleSimulation(num_vehicles=5, duration=60)
data = sim.run()

# Launch dashboard
from autonomous_vehicle_simulation.dashboard import run_dashboard
run_dashboard()  # Access at http://localhost:8050
```

## Docker Setup
```bash
git clone https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis.git
cd Autonomous-Vehicle-Simulation-Data-Analysis
docker-compose up -d
```

Access:
- Dashboard: http://localhost:8050
- Grafana: http://localhost:3000
```

#### ❓ **FAQ** Page
```markdown
# Frequently Asked Questions

## Installation
**Q: How do I install the package?**
A: Use pip: `pip install git+https://github.com/alaminxtration/...`

**Q: Do I need Docker?**
A: Docker is optional but recommended for full-stack deployment.

## Usage
**Q: How do I start the dashboard?**
A: Run `run_dashboard()` from the package or access http://localhost:8050

**Q: Can I customize the simulation?**
A: Yes! See [Configuration Guide](Configuration) for details.

## Troubleshooting
**Q: Dashboard won't start**
A: Check if port 8050 is available and Python environment is activated.

**Q: No data showing**
A: Verify Redis is running: `docker ps | grep redis`
```

#### 🔧 **Configuration** Page
```markdown
# Configuration Guide

## Simulation Parameters
```python
sim = VehicleSimulation(
    num_vehicles=10,        # 1-100 vehicles
    duration=120,           # seconds
    update_interval=0.1,    # update frequency
    scenario="urban"        # highway/urban/parking
)
```

## Dashboard Configuration
```python
config = DashboardConfig(
    port=8050,
    debug=False,
    auto_refresh=True
)
run_dashboard(config)
```

## Docker Configuration
Edit `docker-compose.yml` to customize:
- Port mappings
- Resource limits
- Environment variables
```

#### 📊 **Examples** Page
```markdown
# Usage Examples

## Basic Simulation
```python
from autonomous_vehicle_simulation import VehicleSimulation

sim = VehicleSimulation(num_vehicles=5, duration=30)
data = sim.run()
print(f"Collected {len(data)} data points")
```

## Risk Analysis
```python
from autonomous_vehicle_simulation import RiskAssessment

risk_analyzer = RiskAssessment()
collisions = risk_analyzer.detect_collisions(vehicles)
safety_scores = risk_analyzer.calculate_safety_scores(vehicles)
```

## Data Export
```python
# Export to CSV
data.to_csv("simulation_results.csv")

# Export to JSON
export_data(data, "results.json", format="json")
```
```

### Step 3: Creating Wiki Pages
1. Go to your repository
2. Click **Wiki** tab
3. Click **Create the first page** or **New Page**
4. Copy content from above for each page
5. Save each page

### Step 4: Organize Wiki Sidebar
Create a custom sidebar by editing `_Sidebar.md`:
```markdown
## 📚 Navigation
- [Home](Home)
- [Quick Start](Quick-Start)
- [Installation](Installation)
- [Configuration](Configuration)
- [Examples](Examples)
- [API Reference](API-Reference)
- [FAQ](FAQ)
- [Troubleshooting](Troubleshooting)

## 🔗 External Links
- [GitHub Repo](https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis)
- [Issues](https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis/issues)
- [Discussions](https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis/discussions)
```
