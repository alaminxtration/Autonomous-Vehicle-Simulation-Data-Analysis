#!/usr/bin/env python3
"""
Progressive Full Stack Deployment
Deploy services step by step to access all UIs
"""
import subprocess
import time
import requests
import webbrowser

def run_command(cmd, description, check_success=True):
    """Run a command and show results"""
    print(f"\n{'='*50}")
    print(f"🔧 {description}")
    print(f"{'='*50}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0 or not check_success:
            print("✅ SUCCESS!")
            if result.stdout.strip():
                print(f"Output: {result.stdout[:500]}")
            return True
        else:
            print("❌ FAILED!")
            if result.stderr.strip():
                print(f"Error: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def check_service(url, name, timeout=10):
    """Check if a service is accessible"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name}: ACCESSIBLE at {url}")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code} at {url}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {name}: NOT ACCESSIBLE at {url}")
        return False
    except Exception as e:
        print(f"❌ {name}: ERROR {e}")
        return False

def deploy_step_by_step():
    """Deploy services progressively"""
    print("🚀 PROGRESSIVE FULL STACK DEPLOYMENT")
    print("="*60)
    
    # Step 1: Stop current setup
    print("\n1️⃣ STOPPING CURRENT MINIMAL SETUP")
    run_command("docker-compose -f docker-compose-minimal.yml down", "Stop Redis-only setup", False)
    
    # Step 2: Try core services first (Kafka stack)
    print("\n2️⃣ DEPLOYING KAFKA STACK")
    success = run_command(
        "docker-compose up -d zookeeper kafka kafka-ui", 
        "Deploy Kafka + Zookeeper + UI",
        False
    )
    
    if success:
        time.sleep(15)  # Give services time to start
        print("\n📊 Checking Kafka UI...")
        if check_service("http://localhost:8080", "Kafka UI"):
            print("🎉 Kafka UI is ready!")
        else:
            print("⏳ Kafka UI still starting... try again in a few minutes")
    
    # Step 3: Try HDFS
    print("\n3️⃣ DEPLOYING HDFS")
    success = run_command(
        "docker-compose up -d namenode datanode", 
        "Deploy HDFS NameNode + DataNode",
        False
    )
    
    if success:
        time.sleep(20)  # HDFS takes longer to start
        print("\n📂 Checking HDFS...")
        if check_service("http://localhost:9870", "HDFS NameNode"):
            print("🎉 HDFS Web UI is ready!")
    
    # Step 4: Try Spark
    print("\n4️⃣ DEPLOYING SPARK")
    success = run_command(
        "docker-compose up -d spark-master spark-worker", 
        "Deploy Spark Master + Worker",
        False
    )
    
    if success:
        time.sleep(10)
        print("\n⚡ Checking Spark...")
        if check_service("http://localhost:8081", "Spark UI"):
            print("🎉 Spark UI is ready!")
    
    # Step 5: Try MLflow
    print("\n5️⃣ DEPLOYING MLFLOW")
    success = run_command(
        "docker-compose up -d mlflow", 
        "Deploy MLflow",
        False
    )
    
    if success:
        time.sleep(5)
        print("\n🤖 Checking MLflow...")
        if check_service("http://localhost:5000", "MLflow"):
            print("🎉 MLflow is ready!")
    
    # Step 6: Try Flink
    print("\n6️⃣ DEPLOYING FLINK")
    success = run_command(
        "docker-compose up -d flink-jobmanager flink-taskmanager", 
        "Deploy Flink JobManager + TaskManager",
        False
    )
    
    if success:
        time.sleep(10)
        print("\n🌊 Checking Flink...")
        if check_service("http://localhost:8082", "Flink Dashboard"):
            print("🎉 Flink Dashboard is ready!")
    
    # Step 7: Try Grafana + Prometheus
    print("\n7️⃣ DEPLOYING MONITORING")
    success = run_command(
        "docker-compose up -d prometheus grafana", 
        "Deploy Prometheus + Grafana",
        False
    )
    
    if success:
        time.sleep(10)
        print("\n📈 Checking Grafana...")
        if check_service("http://localhost:3000", "Grafana"):
            print("🎉 Grafana is ready! (Login: admin/admin)")
    
    # Step 8: Always ensure our dashboard is running
    print("\n8️⃣ ENSURING DASHBOARD IS RUNNING")
    # Start dashboard in background
    try:
        subprocess.Popen([
            ".venv\\Scripts\\python.exe", "dashboard\\simple_dashboard.py"
        ], shell=True)
        time.sleep(3)
        check_service("http://localhost:8050", "Vehicle Dashboard")
    except Exception as e:
        print(f"Dashboard start failed: {e}")

def show_access_summary():
    """Show summary of all accessible services"""
    print("\n" + "="*60)
    print("🌐 SERVICE ACCESS SUMMARY")
    print("="*60)
    
    services = [
        ("Vehicle Dashboard", "http://localhost:8050", "Real-time vehicle monitoring"),
        ("Kafka UI", "http://localhost:8080", "Message queue management"),
        ("MLflow", "http://localhost:5000", "ML model tracking"),
        ("Spark UI", "http://localhost:8081", "Big data processing"),
        ("HDFS", "http://localhost:9870", "Distributed file system"),
        ("Flink", "http://localhost:8082", "Stream processing"),
        ("Grafana", "http://localhost:3000", "Monitoring (admin/admin)")
    ]
    
    print("\n🔍 Checking all services...")
    accessible = []
    
    for name, url, description in services:
        if check_service(url, name, timeout=5):
            accessible.append((name, url, description))
    
    print(f"\n✅ ACCESSIBLE SERVICES ({len(accessible)}/{len(services)}):")
    for name, url, description in accessible:
        print(f"   • {name}: {url}")
        print(f"     └─ {description}")
    
    if accessible:
        print(f"\n🚀 QUICK ACCESS:")
        print("Open these URLs in your browser:")
        for name, url, _ in accessible:
            print(f"   {url}  # {name}")
    
    return accessible

def open_accessible_services(accessible_services):
    """Open accessible services in browser"""
    if not accessible_services:
        print("\n❌ No services are accessible yet")
        return
    
    print(f"\n🌐 Opening {len(accessible_services)} accessible services...")
    
    for name, url, _ in accessible_services:
        try:
            webbrowser.open(url)
            print(f"   🔗 Opened: {name}")
            time.sleep(1)  # Small delay between opens
        except Exception as e:
            print(f"   ❌ Failed to open {name}: {e}")

def show_docker_status():
    """Show Docker container status"""
    print("\n📦 DOCKER CONTAINER STATUS")
    print("="*50)
    run_command("docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'", "Container Status", False)

def main():
    """Main deployment process"""
    print("🚀 AUTONOMOUS VEHICLE SIMULATION")
    print("Full Stack Deployment & Access")
    print("="*60)
    
    try:
        # Step 1: Progressive deployment
        deploy_step_by_step()
        
        # Step 2: Show what's accessible
        accessible = show_access_summary()
        
        # Step 3: Show Docker status
        show_docker_status()
        
        # Step 4: Open services in browser
        if accessible:
            response = input(f"\n🌐 Open {len(accessible)} accessible services in browser? (y/n): ")
            if response.lower() in ['y', 'yes']:
                open_accessible_services(accessible)
        
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("="*60)
        print("\n💡 Tips:")
        print("• Services may take a few minutes to fully start")
        print("• If a service isn't accessible, wait and try again")
        print("• Use 'docker-compose logs <service>' to debug issues")
        print("• Use 'docker-compose down' to stop all services")
        
        print("\n🔧 Manual Commands:")
        print("• Check status: docker-compose ps")
        print("• View logs: docker-compose logs kafka-ui")
        print("• Restart service: docker-compose restart grafana")
        print("• Stop all: docker-compose down")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Deployment interrupted by user")
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")

if __name__ == "__main__":
    main()
