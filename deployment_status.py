#!/usr/bin/env python3
"""
Autonomous Vehicle Simulation - Docker Deployment Status
"""
import subprocess
import requests
import redis
import os
from pathlib import Path

def check_docker_status():
    """Check Docker container status"""
    print("🐳 Docker Container Status")
    print("=" * 40)
    
    try:
        result = subprocess.run(['docker', 'ps', '--format', 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'], 
                              capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print(result.stdout)
            return "av-redis" in result.stdout
        else:
            print("❌ Docker command failed")
            return False
    except Exception as e:
        print(f"❌ Docker check failed: {e}")
        return False

def check_redis_connection():
    """Check Redis container connectivity"""
    print("🔗 Redis Container Connectivity")
    print("=" * 40)
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        ping_result = r.ping()
        info = r.info()
        
        print(f"✅ Redis Connection: {'Active' if ping_result else 'Failed'}")
        print(f"📈 Redis Version: {info.get('redis_version', 'Unknown')}")
        print(f"💾 Memory Used: {info.get('used_memory_human', 'Unknown')}")
        return ping_result
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def check_dashboard_status():
    """Check dashboard accessibility"""
    print("🌐 Dashboard Status")
    print("=" * 40)
    
    try:
        response = requests.get('http://localhost:8050', timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard: Running on http://localhost:8050")
            return True
        else:
            print(f"❌ Dashboard: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Dashboard: Not accessible (may not be running)")
        return False
    except Exception as e:
        print(f"❌ Dashboard check failed: {e}")
        return False

def check_data_files():
    """Check for simulation data files"""
    print("📊 Data Files Status")
    print("=" * 40)
    
    data_dir = Path("data/simulation")
    hdfs_dir = Path("data/hdfs_simulation")
    
    # Check data directories
    if data_dir.exists():
        json_files = list(data_dir.glob("*.json"))
        print(f"✅ Simulation data directory: {len(json_files)} files")
    else:
        print("❌ Simulation data directory: Not found")
    
    if hdfs_dir.exists():
        hdfs_files = list(hdfs_dir.rglob("*.json"))
        print(f"✅ HDFS simulation directory: {len(hdfs_files)} files")
    else:
        print("❌ HDFS simulation directory: Not found")
    
    return data_dir.exists() and hdfs_dir.exists()

def check_python_environment():
    """Check Python environment and packages"""
    print("🐍 Python Environment")
    print("=" * 40)
    
    # Check key packages
    packages = ['dash', 'redis', 'pandas', 'numpy', 'plotly']
    missing = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing.append(package)
    
    return len(missing) == 0

def main():
    """Run comprehensive status check"""
    print("🚗 Autonomous Vehicle Simulation - Docker Deployment Status")
    print("=" * 70)
    
    checks = [
        ("Docker Containers", check_docker_status()),
        ("Redis Connection", check_redis_connection()),
        ("Dashboard Service", check_dashboard_status()),
        ("Data Files", check_data_files()),
        ("Python Environment", check_python_environment())
    ]
    
    print("\n📋 Status Summary")
    print("=" * 40)
    
    all_good = True
    for check_name, status in checks:
        status_emoji = "✅" if status else "❌"
        print(f"{status_emoji} {check_name}: {'OK' if status else 'ISSUE'}")
        if not status:
            all_good = False
    
    print("\n🎯 Quick Access")
    print("=" * 40)
    print("🌐 Dashboard: http://localhost:8050")
    print("🐳 Redis: localhost:6379")
    print("📊 Data: data/simulation/")
    print("🔧 HDFS Sim: data/hdfs_simulation/")
    
    if all_good:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("Your Autonomous Vehicle Simulation is running perfectly with Docker Redis integration!")
    else:
        print("\n⚠️  Some issues detected. Check the status above.")
    
    return all_good

if __name__ == "__main__":
    main()
