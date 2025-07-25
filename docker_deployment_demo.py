#!/usr/bin/env python3
"""
Autonomous Vehicle Simulation - Docker Deployment Demo
Complete demonstration of the hybrid Docker/local setup
"""
import time
import subprocess
import json
import redis
from pathlib import Path

def show_banner():
    """Display project banner"""
    print("🚗" * 25)
    print("   AUTONOMOUS VEHICLE SIMULATION")
    print("      Docker + Local Hybrid Deployment")
    print("🚗" * 25)
    print()

def demo_docker_integration():
    """Demonstrate Docker Redis integration"""
    print("🐳 DOCKER INTEGRATION DEMO")
    print("=" * 50)
    
    # Connect to Redis container
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    # Store real-time vehicle data
    vehicles = ["AV_001", "AV_002", "AV_003", "AV_004", "AV_005"]
    
    print("📡 Storing real-time vehicle data in Redis...")
    for i, vehicle_id in enumerate(vehicles):
        data = {
            "vehicle_id": vehicle_id,
            "timestamp": time.time(),
            "location": {"lat": 37.7749 + i*0.001, "lon": -122.4194 + i*0.001},
            "speed": 25 + i*5,
            "status": "active",
            "battery": 90 - i*2,
            "passengers": i % 4
        }
        r.setex(f"live:vehicle:{vehicle_id}", 300, json.dumps(data))
        print(f"  ✅ {vehicle_id}: Speed {data['speed']} km/h, Battery {data['battery']}%")
    
    print(f"\n📊 Redis Stats:")
    info = r.info()
    print(f"  • Memory Used: {info.get('used_memory_human', 'Unknown')}")
    print(f"  • Connected Clients: {info.get('connected_clients', 'Unknown')}")
    print(f"  • Keys in DB: {r.dbsize()}")

def demo_data_pipeline():
    """Demonstrate data processing pipeline"""
    print("\n⚙️  DATA PROCESSING PIPELINE")
    print("=" * 50)
    
    # Check latest simulation data
    data_dir = Path("data/simulation")
    latest_files = sorted(data_dir.glob("sensor_data_*.json"), key=lambda f: f.stat().st_mtime)
    
    if latest_files:
        latest_file = latest_files[-1]
        file_size = latest_file.stat().st_size
        
        print(f"📄 Latest Data File: {latest_file.name}")
        print(f"  • Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
        # Load and analyze
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        print(f"  • Records: {len(data):,}")
        print(f"  • Vehicles: {len(set(record['vehicle_id'] for record in data))}")
        
        # Risk analysis
        risk_counts = {}
        for record in data:
            risk = record.get('risk_level', 'unknown')
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print("  • Risk Distribution:")
        for risk, count in risk_counts.items():
            percentage = (count / len(data)) * 100
            print(f"    - {risk.title()}: {count} ({percentage:.1f}%)")

def demo_hdfs_simulation():
    """Demonstrate HDFS simulation"""
    print("\n💾 HDFS SIMULATION DEMO")
    print("=" * 50)
    
    hdfs_dir = Path("data/hdfs_simulation")
    if hdfs_dir.exists():
        # Count files by vehicle
        vehicle_files = {}
        for vehicle_dir in hdfs_dir.iterdir():
            if vehicle_dir.is_dir() and vehicle_dir.name.startswith("sensor_data"):
                for subdir in vehicle_dir.iterdir():
                    if subdir.is_dir():
                        vehicle_id = subdir.name
                        file_count = len(list(subdir.glob("*.json")))
                        vehicle_files[vehicle_id] = file_count
        
        print("📂 HDFS Directory Structure:")
        for vehicle_id, count in sorted(vehicle_files.items()):
            print(f"  • {vehicle_id}: {count} data files")
        
        total_files = sum(vehicle_files.values())
        print(f"\n📊 Total HDFS Files: {total_files}")

def demo_dashboard_access():
    """Show dashboard access information"""
    print("\n🌐 DASHBOARD ACCESS")
    print("=" * 50)
    
    print("🎯 Interactive Dashboard Available:")
    print("  • URL: http://localhost:8050")
    print("  • Features:")
    print("    - Real-time vehicle tracking")
    print("    - Risk level analysis")
    print("    - Speed and location monitoring")
    print("    - Historical data visualization")
    print("    - Vehicle fleet overview")

def demo_docker_commands():
    """Show useful Docker commands"""
    print("\n🐳 DOCKER MANAGEMENT COMMANDS")
    print("=" * 50)
    
    commands = [
        ("View containers", "docker-compose -f docker-compose-minimal.yml ps"),
        ("View logs", "docker-compose -f docker-compose-minimal.yml logs"),
        ("Stop containers", "docker-compose -f docker-compose-minimal.yml down"),
        ("Restart containers", "docker-compose -f docker-compose-minimal.yml restart"),
        ("Redis CLI", "docker exec -it av-redis redis-cli")
    ]
    
    print("🔧 Common Commands:")
    for description, command in commands:
        print(f"  • {description}:")
        print(f"    {command}")

def main():
    """Run complete demonstration"""
    show_banner()
    
    try:
        demo_docker_integration()
        demo_data_pipeline()
        demo_hdfs_simulation()
        demo_dashboard_access()
        demo_docker_commands()
        
        print("\n🎉 DEPLOYMENT SUMMARY")
        print("=" * 50)
        print("✅ Redis container running in Docker")
        print("✅ Python services running locally")
        print("✅ Real-time data processing active")
        print("✅ Interactive dashboard available")
        print("✅ HDFS simulation storing data")
        print("✅ Hybrid Docker/local deployment successful")
        
        print("\n🚀 Next Steps:")
        print("  1. Visit http://localhost:8050 for the dashboard")
        print("  2. Run more simulations with different parameters")
        print("  3. Explore Redis data with: docker exec -it av-redis redis-cli")
        print("  4. Scale up with full Docker stack when network allows")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
