#!/usr/bin/env python3
"""
Package Validation and Publishing Checklist
Validates that the autonomous vehicle simulation package is ready for publishing
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a required file exists"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} - MISSING")
        return False

def check_package_structure():
    """Validate package structure"""
    print("🔍 Checking Package Structure...")
    
    required_files = [
        ("setup.py", "Package setup configuration"),
        ("README.md", "Main documentation"),
        ("LICENSE", "License file"),
        ("requirements.txt", "Dependencies"),
        ("CHANGELOG.md", "Version history"),
        ("CONTRIBUTING.md", "Contribution guidelines"),
        ("__init__.py", "Package initialization"),
        (".github/workflows/ci.yml", "CI/CD pipeline"),
        ("PUBLISHING_GUIDE.md", "Publishing documentation"),
    ]
    
    all_good = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            all_good = False
    
    return all_good

def check_docker_services():
    """Check Docker services status"""
    print("\n🐳 Checking Docker Services...")
    
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            
            services = ["redis", "grafana"]
            running_services = []
            
            for service in services:
                if service in output.lower():
                    running_services.append(service)
                    print(f"✅ {service.title()} container: Running")
                else:
                    print(f"❌ {service.title()} container: Not running")
            
            return len(running_services) > 0
        else:
            print("❌ Docker not available or not running")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        return False

def check_python_environment():
    """Check Python environment and dependencies"""
    print("\n🐍 Checking Python Environment...")
    
    print(f"✅ Python version: {sys.version}")
    
    # Check key dependencies
    required_packages = [
        "dash", "pandas", "numpy", "redis", 
        "plotly", "mlflow"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_project_functionality():
    """Test core project functionality"""
    print("\n🧪 Testing Core Functionality...")
    
    try:
        # Test Redis connection
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            print("✅ Redis connection: Working")
            redis_works = True
        except Exception as e:
            print(f"❌ Redis connection: Failed - {e}")
            redis_works = False
        
        # Test simulation import
        try:
            sys.path.append('.')
            from simulation.vehicle_simulation import VehicleSimulation
            print("✅ Vehicle simulation import: Working")
            sim_works = True
        except Exception as e:
            print(f"❌ Vehicle simulation import: Failed - {e}")
            sim_works = False
        
        # Test dashboard import
        try:
            import dashboard.simple_dashboard
            print("✅ Dashboard import: Working")
            dashboard_works = True
        except Exception as e:
            print(f"❌ Dashboard import: Failed - {e}")
            dashboard_works = False
        
        return redis_works and sim_works and dashboard_works
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def generate_package_info():
    """Generate package information summary"""
    print("\n📦 Package Information Summary...")
    
    info = {
        "name": "autonomous-vehicle-simulation",
        "version": "1.0.0",
        "description": "Comprehensive autonomous vehicle simulation and analysis framework",
        "author": "alaminxtration",
        "repository": "https://github.com/alaminxtration/Autonomous-Vehicle-Simulation-Data-Analysis",
        "license": "MIT",
        "python_requires": ">=3.8",
        "status": "Production Ready"
    }
    
    for key, value in info.items():
        print(f"  📋 {key.replace('_', ' ').title()}: {value}")
    
    return info

def check_git_status():
    """Check git repository status"""
    print("\n📝 Checking Git Status...")
    
    try:
        # Check if git repo
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout.strip():
                print("⚠️  Uncommitted changes found:")
                print(result.stdout)
                return False
            else:
                print("✅ Git repository: Clean, ready for release")
                return True
        else:
            print("❌ Not a git repository")
            return False
    except FileNotFoundError:
        print("❌ Git not installed")
        return False

def main():
    """Main validation function"""
    print("🚀 Autonomous Vehicle Simulation - Package Publishing Validation")
    print("=" * 70)
    
    # Run all checks
    checks = [
        ("Package Structure", check_package_structure()),
        ("Docker Services", check_docker_services()),
        ("Python Environment", check_python_environment()),
        ("Project Functionality", check_project_functionality()),
        ("Git Status", check_git_status()),
    ]
    
    # Generate package info
    package_info = generate_package_info()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nOverall Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 PACKAGE IS READY FOR PUBLISHING! 🎉")
        print("\nNext steps:")
        print("1. Create GitHub release (v1.0.0)")
        print("2. Tag the release")
        print("3. Publish to PyPI (optional)")
        print("4. Share with the community")
        print("\nSee PUBLISHING_GUIDE.md for detailed instructions.")
        return True
    else:
        print(f"\n⚠️  {total - passed} issues need to be resolved before publishing")
        print("Please fix the failing checks and run this script again.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
