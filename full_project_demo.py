#!/usr/bin/env python3
"""
Complete Autonomous Vehicle Simulation Demo
Shows all features of the project step-by-step
"""
import time
import subprocess
import requests
import redis
import json
import pandas as pd
from pathlib import Path
import webbrowser

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"🚗 {title}")
    print("="*60)

def print_step(step_num, description):
    """Print formatted step"""
    print(f"\n{step_num}. {description}")
    print("-" * 40)

def check_prerequisites():
    """Check if all prerequisites are met"""
    print_header("CHECKING PREREQUISITES")
    
    checks = []
    
    # Check Docker
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✅ Docker: Available")
            checks.append(True)
        else:
            print("❌ Docker: Not available")
            checks.append(False)
    except:
        print("❌ Docker: Not found")
        checks.append(False)
    
    # Check Python packages
    packages = ['dash', 'redis', 'pandas', 'numpy', 'plotly']
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
            checks.append(True)
        except ImportError:
            print(f"❌ {package}: Missing")
            checks.append(False)
    
    return all(checks)

def start_docker_services():
    """Start Docker services"""
    print_header("STARTING DOCKER SERVICES")
    
    print_step(1, "Starting Redis container")
    try:
        result = subprocess.run([
            'docker-compose', '-f', 'docker-compose-minimal.yml', 'up', '-d'
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print("✅ Redis container started successfully")
            time.sleep(2)  # Give container time to start
            return True
        else:
            print(f"❌ Failed to start Redis: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error starting Docker services: {e}")
        return False

def test_redis_connection():
    """Test Redis connection"""
    print_step(2, "Testing Redis connection")
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        ping_result = r.ping()
        
        if ping_result:
            print("✅ Redis connection successful")
            
            # Store test data
            test_data = {
                "demo": "autonomous_vehicle_simulation",
                "timestamp": time.time(),
                "status": "running"
            }
            r.setex("demo:status", 60, json.dumps(test_data))
            print("✅ Test data stored in Redis")
            return True
        else:
            print("❌ Redis ping failed")
            return False
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def generate_simulation_data():
    """Generate simulation data"""
    print_header("GENERATING SIMULATION DATA")
    
    print_step(1, "Running vehicle simulation (60 seconds)")
    print("📡 Simulating 5 autonomous vehicles...")
    
    try:
        # Run simulation
        result = subprocess.run([
            '.venv\\Scripts\\python.exe', 'simulation\\vehicle_simulation.py', '60'
        ], capture_output=True, text=True, shell=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Simulation completed successfully")
            
            # Check generated files
            data_dir = Path("data/simulation")
            if data_dir.exists():
                json_files = list(data_dir.glob("*.json"))
                parquet_files = list(data_dir.glob("*.parquet"))
                
                print(f"📊 Generated files:")
                print(f"   • JSON files: {len(json_files)}")
                print(f"   • Parquet files: {len(parquet_files)}")
                
                if json_files:
                    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
                    file_size = latest_file.stat().st_size
                    print(f"   • Latest file: {latest_file.name} ({file_size:,} bytes)")
            
            return True
        else:
            print(f"❌ Simulation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False

def analyze_generated_data():
    """Analyze the generated data"""
    print_step(2, "Analyzing generated data")
    
    try:
        # Load latest data
        data_dir = Path("data/simulation")
        parquet_files = list(data_dir.glob("sensor_data_latest.parquet"))
        
        if parquet_files:
            df = pd.read_parquet(parquet_files[0])
            
            print(f"📈 Data Analysis Results:")
            print(f"   • Total records: {len(df):,}")
            print(f"   • Unique vehicles: {df['vehicle_id'].nunique()}")
            print(f"   • Time span: {df['timestamp'].max() - df['timestamp'].min():.0f} seconds")
            
            # Risk analysis
            risk_counts = df['risk_level'].value_counts()
            print(f"   • Risk distribution:")
            for risk, count in risk_counts.items():
                percentage = (count / len(df)) * 100
                print(f"     - {risk.title()}: {count} ({percentage:.1f}%)")
            
            # Speed analysis
            avg_speed = df['calculated_speed'].mean()
            max_speed = df['calculated_speed'].max()
            print(f"   • Average speed: {avg_speed:.2f} m/s")
            print(f"   • Maximum speed: {max_speed:.2f} m/s")
            
            return True
        else:
            print("❌ No parquet files found")
            return False
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        return False

def start_dashboard():
    """Start the dashboard"""
    print_header("STARTING INTERACTIVE DASHBOARD")
    
    print_step(1, "Launching dashboard server")
    
    try:
        # Start dashboard in background
        process = subprocess.Popen([
            '.venv\\Scripts\\python.exe', 'dashboard\\simple_dashboard.py'
        ], shell=True)
        
        print("🌐 Dashboard starting...")
        time.sleep(5)  # Give dashboard time to start
        
        # Check if dashboard is accessible
        try:
            response = requests.get('http://localhost:8050', timeout=10)
            if response.status_code == 200:
                print("✅ Dashboard is running on http://localhost:8050")
                return process
            else:
                print(f"❌ Dashboard returned status code: {response.status_code}")
                process.terminate()
                return None
        except requests.exceptions.ConnectionError:
            print("❌ Dashboard is not accessible")
            process.terminate()
            return None
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return None

def demonstrate_redis_integration():
    """Demonstrate Redis integration"""
    print_header("DEMONSTRATING REDIS INTEGRATION")
    
    print_step(1, "Storing real-time vehicle data in Redis")
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Simulate real-time vehicle updates
        vehicles = ["AV_001", "AV_002", "AV_003", "AV_004", "AV_005"]
        
        for i, vehicle_id in enumerate(vehicles):
            vehicle_data = {
                "vehicle_id": vehicle_id,
                "timestamp": time.time(),
                "location": {
                    "lat": 37.7749 + i * 0.001,
                    "lon": -122.4194 + i * 0.001
                },
                "speed": 25 + i * 5,
                "status": "active",
                "battery": 90 - i * 2,
                "risk_level": ["low", "medium", "high"][i % 3]
            }
            
            # Store with 5-minute TTL
            r.setex(f"live:vehicle:{vehicle_id}", 300, json.dumps(vehicle_data))
            print(f"   ✅ {vehicle_id}: Speed {vehicle_data['speed']} km/h, Battery {vehicle_data['battery']}%")
        
        # Show Redis stats
        info = r.info()
        print(f"\n📊 Redis Statistics:")
        print(f"   • Memory used: {info.get('used_memory_human', 'Unknown')}")
        print(f"   • Connected clients: {info.get('connected_clients', 'Unknown')}")
        print(f"   • Total keys: {r.dbsize()}")
        
        return True
    except Exception as e:
        print(f"❌ Redis integration failed: {e}")
        return False

def show_system_status():
    """Show complete system status"""
    print_header("SYSTEM STATUS OVERVIEW")
    
    try:
        # Run status check
        result = subprocess.run([
            '.venv\\Scripts\\python.exe', 'deployment_status.py'
        ], capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ Status check failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking system status: {e}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print_header("NEXT STEPS & USAGE GUIDE")
    
    print("🎯 Your Autonomous Vehicle Simulation is now fully operational!")
    print("\n📋 What you can do now:")
    
    print("\n1. 🌐 VIEW DASHBOARD:")
    print("   • Open: http://localhost:8050")
    print("   • Features: Real-time tracking, risk analysis, fleet overview")
    
    print("\n2. 🚗 RUN MORE SIMULATIONS:")
    print("   • Quick test: python simulation\\vehicle_simulation.py 120")
    print("   • Extended run: python simulation\\vehicle_simulation.py 600")
    print("   • Continuous: python simulation\\continuous_simulation.py")
    
    print("\n3. 🔍 EXPLORE DATA:")
    print("   • View files: dir data\\simulation\\")
    print("   • Analyze with Python/Pandas")
    print("   • Check Redis data: docker exec -it av-redis redis-cli")
    
    print("\n4. 🔧 SYSTEM MANAGEMENT:")
    print("   • Status check: python deployment_status.py")
    print("   • Docker logs: docker-compose -f docker-compose-minimal.yml logs")
    print("   • Stop system: docker-compose -f docker-compose-minimal.yml down")
    
    print("\n5. 📊 CUSTOM ANALYSIS:")
    print("   • Load data: pd.read_parquet('data/simulation/sensor_data_latest.parquet')")
    print("   • Custom risk models: Edit flink_processor\\sensor_data_processor_nokafka.py")
    print("   • Add sensors: Modify simulation\\vehicle_simulation.py")

def main():
    """Run complete demonstration"""
    print("🚗" * 20)
    print("AUTONOMOUS VEHICLE SIMULATION")
    print("Complete Project Demonstration")
    print("🚗" * 20)
    
    dashboard_process = None
    
    try:
        # Step 1: Prerequisites
        if not check_prerequisites():
            print("\n❌ Prerequisites not met. Please install missing components.")
            return False
        
        # Step 2: Start Docker services
        if not start_docker_services():
            print("\n❌ Failed to start Docker services.")
            return False
        
        # Step 3: Test Redis
        if not test_redis_connection():
            print("\n❌ Redis connection failed.")
            return False
        
        # Step 4: Generate data
        if not generate_simulation_data():
            print("\n❌ Data generation failed.")
            return False
        
        # Step 5: Analyze data
        if not analyze_generated_data():
            print("\n❌ Data analysis failed.")
            return False
        
        # Step 6: Start dashboard
        dashboard_process = start_dashboard()
        if not dashboard_process:
            print("\n⚠️  Dashboard failed to start, but other features are working.")
        
        # Step 7: Redis integration demo
        if not demonstrate_redis_integration():
            print("\n❌ Redis integration demo failed.")
            return False
        
        # Step 8: System status
        show_system_status()
        
        # Step 9: Next steps
        show_next_steps()
        
        print("\n🎉 DEMONSTRATION COMPLETE!")
        print("Your Autonomous Vehicle Simulation is ready for use!")
        
        if dashboard_process:
            print("\n💡 The dashboard is running. Press Ctrl+C to stop the demo.")
            try:
                # Open browser automatically
                webbrowser.open('http://localhost:8050')
                dashboard_process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping dashboard...")
                dashboard_process.terminate()
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        if dashboard_process:
            dashboard_process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if dashboard_process:
            dashboard_process.terminate()
        return False

if __name__ == "__main__":
    main()
