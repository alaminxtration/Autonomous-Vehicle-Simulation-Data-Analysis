#!/usr/bin/env python3
"""
Service Status Checker - Verifies all services are working
"""

import requests
import time

def check_service(name, url):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: {url} - Working!")
            return True
        else:
            print(f"⚠️  {name}: {url} - HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {url} - Failed ({str(e)})")
        return False

def main():
    print("🔍 Checking All AV Simulation Services")
    print("=" * 50)
    
    services = [
        ("Dashboard", "http://localhost:8050"),
        ("Kafka UI", "http://localhost:8080"),
        ("MLflow", "http://localhost:5000"),
        ("Spark UI", "http://localhost:8081"),
        ("HDFS", "http://localhost:9870"),
        ("Flink", "http://localhost:8082"),
        ("Grafana", "http://localhost:3000")
    ]
    
    working_count = 0
    total_count = len(services)
    
    for name, url in services:
        if check_service(name, url):
            working_count += 1
        time.sleep(0.5)  # Small delay between checks
    
    print("\n" + "=" * 50)
    print(f"📊 Service Status: {working_count}/{total_count} services working")
    
    if working_count == total_count:
        print("🎉 All services are operational!")
        print("\n🔗 Quick Access Links:")
        for name, url in services:
            auth_info = " (admin/admin)" if name == "Grafana" else ""
            print(f"   {name}: {url}{auth_info}")
    elif working_count > 0:
        print("⚠️  Some services are running")
    else:
        print("❌ No services are responding")
    
    return working_count == total_count

if __name__ == "__main__":
    main()
