#!/usr/bin/env python3
"""
Deploy All Services Script
Attempts to deploy the full AV simulation stack with fallbacks
"""

import subprocess
import time
import requests
import json
import os
from pathlib import Path

class ServiceDeployer:
    def __init__(self):
        self.services = {
            'Dashboard': {'port': 8050, 'url': 'http://localhost:8050', 'status': False},
            'Kafka UI': {'port': 8080, 'url': 'http://localhost:8080', 'status': False},
            'MLflow': {'port': 5000, 'url': 'http://localhost:5000', 'status': False},
            'Spark UI': {'port': 8081, 'url': 'http://localhost:8081', 'status': False},
            'HDFS': {'port': 9870, 'url': 'http://localhost:9870', 'status': False},
            'Flink': {'port': 8082, 'url': 'http://localhost:8082', 'status': False},
            'Grafana': {'port': 3000, 'url': 'http://localhost:3000', 'status': False, 'auth': 'admin/admin'},
            'Redis': {'port': 6379, 'url': 'redis://localhost:6379', 'status': False}
        }
        
    def check_port(self, port):
        """Check if a port is accessible"""
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
    
    def check_http_service(self, url):
        """Check if HTTP service is responding"""
        try:
            response = requests.get(url, timeout=5)
            return response.status_code < 500
        except:
            return False
    
    def run_command(self, command, description):
        """Run a command and return success status"""
        try:
            print(f"🔧 {description}...")
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"✅ {description} - Success")
                return True
            else:
                print(f"❌ {description} - Failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} - Timeout")
            return False
        except Exception as e:
            print(f"❌ {description} - Error: {e}")
            return False
    
    def deploy_with_docker_compose(self):
        """Try to deploy using docker-compose"""
        print("\n🐳 Attempting Docker Compose deployment...")
        
        # Try the main docker-compose file
        if self.run_command("docker-compose up -d", "Starting full stack with docker-compose"):
            time.sleep(30)  # Wait for services to start
            return True
        
        # Try the progressive file
        if self.run_command("docker-compose -f docker-compose-progressive.yml up -d", "Starting with progressive compose"):
            time.sleep(30)
            return True
        
        return False
    
    def deploy_individual_services(self):
        """Deploy services individually"""
        print("\n🚀 Deploying services individually...")
        
        # Kafka and Zookeeper
        kafka_commands = [
            "docker run -d --name zookeeper --network av-network -p 2181:2181 -e ZOOKEEPER_CLIENT_PORT=2181 -e ZOOKEEPER_TICK_TIME=2000 confluentinc/cp-zookeeper:7.4.0",
            "docker run -d --name kafka --network av-network -p 9092:9092 -e KAFKA_BROKER_ID=1 -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 confluentinc/cp-kafka:7.4.0",
            "docker run -d --name kafka-ui --network av-network -p 8080:8080 -e KAFKA_CLUSTERS_0_NAME=local -e KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS=kafka:9092 provectuslabs/kafka-ui:latest"
        ]
        
        # MLflow
        mlflow_command = "docker run -d --name mlflow --network av-network -p 5000:5000 mlflow/mlflow mlflow server --host 0.0.0.0 --port 5000"
        
        # Spark
        spark_commands = [
            "docker run -d --name spark-master --network av-network -p 8081:8080 bitnami/spark:latest spark-class org.apache.spark.deploy.master.Master",
            "docker run -d --name spark-worker --network av-network -e SPARK_MODE=worker -e SPARK_MASTER_URL=spark://spark-master:7077 bitnami/spark:latest"
        ]
        
        # HDFS
        hdfs_commands = [
            "docker run -d --name namenode --network av-network -p 9870:9870 bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8",
            "docker run -d --name datanode --network av-network bde2020/hadoop-datanode:2.0.0-hadoop3.2.1-java8"
        ]
        
        # Flink
        flink_commands = [
            "docker run -d --name flink-jobmanager --network av-network -p 8082:8081 flink:latest jobmanager",
            "docker run -d --name flink-taskmanager --network av-network flink:latest taskmanager"
        ]
        
        # Grafana
        grafana_command = "docker run -d --name grafana --network av-network -p 3000:3000 grafana/grafana:latest"
        
        # Try each service
        services_to_deploy = [
            ("Kafka Stack", kafka_commands),
            ("MLflow", [mlflow_command]),
            ("Spark", spark_commands),
            ("HDFS", hdfs_commands),
            ("Flink", flink_commands),
            ("Grafana", [grafana_command])
        ]
        
        for service_name, commands in services_to_deploy:
            print(f"\n📦 Deploying {service_name}...")
            for cmd in commands:
                if not self.run_command(cmd, f"Starting {service_name} component"):
                    print(f"⚠️  {service_name} deployment had issues, continuing...")
                    break
            time.sleep(5)
    
    def start_local_services(self):
        """Start local Python services"""
        print("\n🐍 Starting local Python services...")
        
        # Start the dashboard
        dashboard_cmd = ".venv\\Scripts\\python.exe dashboard\\storage_optimized_dashboard.py"
        if self.run_command(f"start /B {dashboard_cmd}", "Starting Dashboard"):
            self.services['Dashboard']['status'] = True
        
        # Start MLflow locally if Docker fails
        mlflow_cmd = ".venv\\Scripts\\python.exe -m mlflow server --host 0.0.0.0 --port 5000"
        self.run_command(f"start /B {mlflow_cmd}", "Starting local MLflow")
    
    def check_all_services(self):
        """Check status of all services"""
        print("\n🔍 Checking service status...")
        
        for service, config in self.services.items():
            port = config['port']
            url = config['url']
            
            if service == 'Redis':
                # Redis is already running
                if self.check_port(port):
                    config['status'] = True
                    print(f"✅ {service}: Running on port {port}")
                else:
                    print(f"❌ {service}: Not accessible on port {port}")
            else:
                if self.check_port(port):
                    if url.startswith('http') and self.check_http_service(url):
                        config['status'] = True
                        print(f"✅ {service}: Running at {url}")
                    elif not url.startswith('http'):
                        config['status'] = True
                        print(f"✅ {service}: Running on port {port}")
                    else:
                        print(f"⚠️  {service}: Port open but service not responding at {url}")
                else:
                    print(f"❌ {service}: Not accessible on port {port}")
    
    def create_service_links(self):
        """Create easy access links"""
        print("\n🔗 Service Access Links:")
        print("=" * 50)
        
        for service, config in self.services.items():
            if config['status'] and 'url' in config:
                auth_info = f" (Login: {config['auth']})" if 'auth' in config else ""
                print(f"✅ {service}: {config['url']}{auth_info}")
            else:
                print(f"❌ {service}: Not available")
    
    def deploy_all(self):
        """Main deployment function"""
        print("🚀 AV Simulation Full Stack Deployment")
        print("=" * 60)
        
        # Create network if it doesn't exist
        self.run_command("docker network create av-network", "Creating Docker network")
        
        # Try Docker Compose first
        if not self.deploy_with_docker_compose():
            print("\n📦 Docker Compose failed, trying individual deployment...")
            self.deploy_individual_services()
        
        # Start local services
        self.start_local_services()
        
        # Wait for services to fully start
        print("\n⏳ Waiting for services to initialize...")
        time.sleep(15)
        
        # Check all services
        self.check_all_services()
        
        # Create access links
        self.create_service_links()
        
        # Summary
        running_count = sum(1 for config in self.services.values() if config['status'])
        total_count = len(self.services)
        
        print(f"\n📊 Deployment Summary: {running_count}/{total_count} services running")
        
        if running_count == total_count:
            print("🎉 Full stack deployment successful!")
        elif running_count > total_count // 2:
            print("⚠️  Partial deployment - most services running")
        else:
            print("❌ Limited deployment - many services failed")
        
        return running_count, total_count

if __name__ == "__main__":
    deployer = ServiceDeployer()
    deployer.deploy_all()
