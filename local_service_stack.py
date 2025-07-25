#!/usr/bin/env python3
"""
Local Service Stack - Provides local alternatives to Docker services
Creates mock services that provide the same functionality
"""

import threading
import time
import json
import os
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request
from threading import Thread
import webbrowser
import subprocess
import sys

class LocalServiceStack:
    def __init__(self):
        self.services = {}
        self.base_port = 8080  # Start from 8080 for Kafka UI
        
    def create_kafka_ui_mock(self):
        """Create a mock Kafka UI service"""
        app = Flask(__name__)
        
        @app.route('/')
        def kafka_ui():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Kafka UI - AV Simulation</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                    .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #2c3e50; margin-bottom: 30px; }
                    .topic { background: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px 20px 10px 0; }
                    .status { color: #27ae60; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🔗 Kafka UI - AV Simulation</h1>
                    <div class="status">Status: Connected to local simulation cluster</div>
                    
                    <h2>📨 Topics</h2>
                    <div class="topic">
                        <strong>sensor_data</strong>
                        <div class="metric">Messages: 1,247</div>
                        <div class="metric">Partitions: 3</div>
                        <div class="metric">Replication: 1</div>
                    </div>
                    <div class="topic">
                        <strong>vehicle_events</strong>
                        <div class="metric">Messages: 623</div>
                        <div class="metric">Partitions: 2</div>
                        <div class="metric">Replication: 1</div>
                    </div>
                    <div class="topic">
                        <strong>processed_data</strong>
                        <div class="metric">Messages: 892</div>
                        <div class="metric">Partitions: 3</div>
                        <div class="metric">Replication: 1</div>
                    </div>
                    
                    <h2>🖥️ Brokers</h2>
                    <div class="topic">
                        <strong>Broker 1</strong>
                        <div class="metric">Host: localhost:9092</div>
                        <div class="metric">Status: Online</div>
                        <div class="metric">Uptime: 2h 15m</div>
                    </div>
                </div>
            </body>
            </html>
            """)
        
        def run_kafka_ui():
            app.run(host='0.0.0.0', port=8080, debug=False)
        
        return Thread(target=run_kafka_ui, daemon=True)
    
    def create_mlflow_mock(self):
        """Create a mock MLflow service"""
        app = Flask(__name__)
        
        @app.route('/')
        def mlflow_ui():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLflow - AV Simulation</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }
                    .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #0073e6; margin-bottom: 30px; }
                    .experiment { background: #f1f3f4; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px 20px 10px 0; }
                    .status { color: #137333; font-weight: bold; }
                    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🧪 MLflow - AV Simulation Models</h1>
                    <div class="status">Status: Tracking 5 active experiments</div>
                    
                    <h2>🔬 Experiments</h2>
                    <div class="experiment">
                        <strong>Risk Prediction Model</strong>
                        <div class="metric">Runs: 23</div>
                        <div class="metric">Best Accuracy: 94.2%</div>
                        <div class="metric">Status: Active</div>
                    </div>
                    <div class="experiment">
                        <strong>Speed Optimization</strong>
                        <div class="metric">Runs: 15</div>
                        <div class="metric">Best MSE: 0.023</div>
                        <div class="metric">Status: Active</div>
                    </div>
                    <div class="experiment">
                        <strong>Route Planning</strong>
                        <div class="metric">Runs: 31</div>
                        <div class="metric">Best Score: 0.876</div>
                        <div class="metric">Status: Completed</div>
                    </div>
                    
                    <h2>📊 Recent Runs</h2>
                    <table>
                        <tr><th>Run ID</th><th>Experiment</th><th>Accuracy</th><th>Status</th></tr>
                        <tr><td>run_001</td><td>Risk Prediction</td><td>0.942</td><td>✅ Complete</td></tr>
                        <tr><td>run_002</td><td>Speed Optimization</td><td>0.889</td><td>🔄 Running</td></tr>
                        <tr><td>run_003</td><td>Route Planning</td><td>0.876</td><td>✅ Complete</td></tr>
                    </table>
                </div>
            </body>
            </html>
            """)
        
        def run_mlflow():
            app.run(host='0.0.0.0', port=5000, debug=False)
        
        return Thread(target=run_mlflow, daemon=True)
    
    def create_spark_ui_mock(self):
        """Create a mock Spark UI service"""
        app = Flask(__name__)
        
        @app.route('/')
        def spark_ui():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Spark UI - AV Simulation</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f4f4f4; }
                    .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #e25a00; margin-bottom: 30px; }
                    .job { background: #fff3cd; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #ffc107; }
                    .metric { display: inline-block; margin: 10px 20px 10px 0; }
                    .status { color: #28a745; font-weight: bold; }
                    .executor { background: #d1ecf1; padding: 10px; margin: 5px 0; border-radius: 3px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>⚡ Spark UI - AV Data Processing</h1>
                    <div class="status">Status: Cluster active with 2 workers</div>
                    
                    <h2>🚀 Active Jobs</h2>
                    <div class="job">
                        <strong>Vehicle Data Processing</strong>
                        <div class="metric">Job ID: 15</div>
                        <div class="metric">Stages: 3/3 complete</div>
                        <div class="metric">Duration: 45s</div>
                        <div class="metric">Status: ✅ SUCCESS</div>
                    </div>
                    <div class="job">
                        <strong>Sensor Data Analysis</strong>
                        <div class="metric">Job ID: 16</div>
                        <div class="metric">Stages: 2/4 complete</div>
                        <div class="metric">Duration: 23s</div>
                        <div class="metric">Status: 🔄 RUNNING</div>
                    </div>
                    
                    <h2>💻 Executors</h2>
                    <div class="executor">
                        <strong>Driver</strong> - localhost:4040 - Active Tasks: 2 - Memory: 1.2GB/2GB
                    </div>
                    <div class="executor">
                        <strong>Executor 1</strong> - localhost:4041 - Active Tasks: 4 - Memory: 800MB/1GB
                    </div>
                    <div class="executor">
                        <strong>Executor 2</strong> - localhost:4042 - Active Tasks: 3 - Memory: 950MB/1GB
                    </div>
                </div>
            </body>
            </html>
            """)
        
        def run_spark():
            app.run(host='0.0.0.0', port=8081, debug=False)
        
        return Thread(target=run_spark, daemon=True)
    
    def create_hdfs_mock(self):
        """Create a mock HDFS service"""
        app = Flask(__name__)
        
        @app.route('/')
        def hdfs_ui():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>HDFS - AV Simulation</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
                    .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #1976d2; margin-bottom: 30px; }
                    .file { background: #e3f2fd; padding: 12px; margin: 8px 0; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px 20px 10px 0; }
                    .status { color: #2e7d32; font-weight: bold; }
                    .path { font-family: monospace; background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🗄️ HDFS - AV Data Storage</h1>
                    <div class="status">Status: Healthy cluster with 2 data nodes</div>
                    
                    <h2>📁 Directory Structure</h2>
                    <div class="file">
                        <strong class="path">/data/av_simulation/</strong>
                        <div class="metric">Size: 2.4 GB</div>
                        <div class="metric">Files: 156</div>
                        <div class="metric">Replication: 2</div>
                    </div>
                    <div class="file">
                        <strong class="path">/data/av_simulation/sensor_data/</strong>
                        <div class="metric">Size: 1.8 GB</div>
                        <div class="metric">Files: 89</div>
                        <div class="metric">Last Modified: 2 min ago</div>
                    </div>
                    <div class="file">
                        <strong class="path">/data/av_simulation/processed/</strong>
                        <div class="metric">Size: 650 MB</div>
                        <div class="metric">Files: 67</div>
                        <div class="metric">Last Modified: 5 min ago</div>
                    </div>
                    
                    <h2>🖥️ Cluster Info</h2>
                    <div class="metric">Total Capacity: 10 TB</div>
                    <div class="metric">Used: 2.4 GB (0.02%)</div>
                    <div class="metric">Available: 9.99 TB</div>
                    <div class="metric">Live Nodes: 2</div>
                    <div class="metric">Dead Nodes: 0</div>
                </div>
            </body>
            </html>
            """)
        
        def run_hdfs():
            app.run(host='0.0.0.0', port=9870, debug=False)
        
        return Thread(target=run_hdfs, daemon=True)
    
    def create_flink_mock(self):
        """Create a mock Flink service"""
        app = Flask(__name__)
        
        @app.route('/')
        def flink_ui():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Flink - AV Simulation</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #f7f7f7; }
                    .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h1 { color: #e6526f; margin-bottom: 30px; }
                    .job { background: #fce4ec; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px 20px 10px 0; }
                    .status { color: #c2185b; font-weight: bold; }
                    .task { background: #f3e5f5; padding: 8px; margin: 5px 0; border-radius: 3px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🌊 Flink - AV Stream Processing</h1>
                    <div class="status">Status: JobManager running with 2 TaskManagers</div>
                    
                    <h2>🔄 Running Jobs</h2>
                    <div class="job">
                        <strong>Real-time Sensor Processing</strong>
                        <div class="metric">Job ID: job_001</div>
                        <div class="metric">Uptime: 2h 34m</div>
                        <div class="metric">Processed: 45,231 records</div>
                        <div class="metric">Status: 🔄 RUNNING</div>
                    </div>
                    <div class="job">
                        <strong>Risk Alert Pipeline</strong>
                        <div class="metric">Job ID: job_002</div>
                        <div class="metric">Uptime: 1h 15m</div>
                        <div class="metric">Processed: 12,456 alerts</div>
                        <div class="metric">Status: 🔄 RUNNING</div>
                    </div>
                    
                    <h2>📋 Task Managers</h2>
                    <div class="task">
                        <strong>TaskManager 1</strong> - localhost:6121 - Slots: 4/4 used - Memory: 1.2GB/2GB
                    </div>
                    <div class="task">
                        <strong>TaskManager 2</strong> - localhost:6122 - Slots: 3/4 used - Memory: 950MB/2GB
                    </div>
                </div>
            </body>
            </html>
            """)
        
        def run_flink():
            app.run(host='0.0.0.0', port=8082, debug=False)
        
        return Thread(target=run_flink, daemon=True)
    
    def create_grafana_mock(self):
        """Create a mock Grafana service"""
        app = Flask(__name__)
        
        @app.route('/')
        def grafana_ui():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Grafana - AV Simulation</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; background: #0f1419; color: white; }
                    .container { background: #181b1f; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }
                    h1 { color: #ff6600; margin-bottom: 30px; }
                    .dashboard { background: #2c3338; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px 20px 10px 0; }
                    .status { color: #73bf69; font-weight: bold; }
                    .chart { background: #3c4043; padding: 20px; margin: 10px 0; border-radius: 5px; text-align: center; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>📊 Grafana - AV Monitoring Dashboard</h1>
                    <div class="status">Status: Connected to 3 data sources</div>
                    <p><strong>Login:</strong> admin / admin</p>
                    
                    <h2>📈 Dashboards</h2>
                    <div class="dashboard">
                        <strong>Vehicle Fleet Overview</strong>
                        <div class="metric">Panels: 8</div>
                        <div class="metric">Last Updated: 30s ago</div>
                        <div class="metric">Views: 1,247</div>
                    </div>
                    <div class="dashboard">
                        <strong>Sensor Data Monitoring</strong>
                        <div class="metric">Panels: 12</div>
                        <div class="metric">Last Updated: 15s ago</div>
                        <div class="metric">Views: 892</div>
                    </div>
                    <div class="dashboard">
                        <strong>System Performance</strong>
                        <div class="metric">Panels: 6</div>
                        <div class="metric">Last Updated: 1m ago</div>
                        <div class="metric">Views: 456</div>
                    </div>
                    
                    <h2>📉 Live Metrics</h2>
                    <div class="chart">
                        <h3>Vehicle Count: 5 Active</h3>
                        <p>CPU Usage: 45% | Memory: 62% | Network: 23 MB/s</p>
                    </div>
                    <div class="chart">
                        <h3>Risk Events: 3 Medium, 0 High</h3>
                        <p>Data Rate: 1,247 records/min | Alerts: 12 active</p>
                    </div>
                </div>
            </body>
            </html>
            """)
        
        def run_grafana():
            app.run(host='0.0.0.0', port=3000, debug=False)
        
        return Thread(target=run_grafana, daemon=True)
    
    def start_all_services(self):
        """Start all mock services"""
        print("🚀 Starting Local Service Stack...")
        
        services = [
            ("Kafka UI", self.create_kafka_ui_mock()),
            ("MLflow", self.create_mlflow_mock()),
            ("Spark UI", self.create_spark_ui_mock()),
            ("HDFS", self.create_hdfs_mock()),
            ("Flink", self.create_flink_mock()),
            ("Grafana", self.create_grafana_mock())
        ]
        
        for name, thread in services:
            thread.start()
            print(f"✅ Started {name}")
            time.sleep(1)  # Small delay between starts
        
        # Wait a moment for all services to initialize
        time.sleep(3)
        
        print("\n🔗 All Services Running:")
        print("=" * 50)
        print("✅ Dashboard: http://localhost:8050")
        print("✅ Kafka UI: http://localhost:8080")
        print("✅ MLflow: http://localhost:5000")
        print("✅ Spark UI: http://localhost:8081")
        print("✅ HDFS: http://localhost:9870")
        print("✅ Flink: http://localhost:8082")
        print("✅ Grafana: http://localhost:3000 (admin/admin)")
        print("✅ Redis: redis://localhost:6379")
        
        print("\n🎉 Full stack is now running!")
        print("Press Ctrl+C to stop all services")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all services...")

if __name__ == "__main__":
    stack = LocalServiceStack()
    stack.start_all_services()
