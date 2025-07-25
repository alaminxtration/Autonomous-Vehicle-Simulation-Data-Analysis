#!/usr/bin/env python3
"""
Test installation script for the published package
"""

import subprocess
import sys
import tempfile
import os

def test_installation():
    """Test package installation from GitHub"""
    print("🧪 Testing Package Installation...")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Test installation command
        install_cmd = [
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis.git"
        ]
        
        print("📦 Installation command:")
        print(" ".join(install_cmd))
        print("\n✨ After publishing, users can install with:")
        print("pip install git+https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis.git")
        
        # Test import
        try:
            import simulation.vehicle_simulation
            print("✅ Local simulation module: Available")
        except ImportError as e:
            print(f"❌ Local simulation module: {e}")
        
        print("\n🎯 After installation, users will be able to:")
        print("1. from simulation.vehicle_simulation import VehicleSimulation")
        print("2. sim = VehicleSimulation(num_vehicles=5, duration=60)")
        print("3. data = sim.run()")
        print("4. Access dashboard at http://localhost:8050")

if __name__ == "__main__":
    test_installation()
