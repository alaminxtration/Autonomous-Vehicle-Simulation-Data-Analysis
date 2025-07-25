#!/usr/bin/env python3
"""
Test Redis Integration with Docker Container
"""
import redis
import json
import datetime
from typing import Dict, Any

def test_redis_connection():
    """Test Redis connection to Docker container"""
    print("🔗 Testing Redis Connection to Docker Container")
    print("=" * 50)
    
    try:
        # Connect to Redis container
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Test basic connectivity
        ping_result = r.ping()
        print(f"✅ Redis Ping: {ping_result}")
        
        # Test data storage and retrieval
        test_data = {
            "vehicle_id": "AV_TEST",
            "timestamp": datetime.datetime.now().isoformat(),
            "speed": 45.5,
            "location": {"lat": 37.7749, "lon": -122.4194},
            "status": "active"
        }
        
        # Store test data
        r.set("test:vehicle:AV_TEST", json.dumps(test_data))
        print("✅ Test data stored in Redis")
        
        # Retrieve test data
        retrieved_data = json.loads(r.get("test:vehicle:AV_TEST"))
        print("✅ Test data retrieved from Redis")
        print(f"📊 Retrieved: {retrieved_data}")
        
        # Test Redis info
        info = r.info()
        print(f"📈 Redis Version: {info.get('redis_version', 'Unknown')}")
        print(f"💾 Memory Used: {info.get('used_memory_human', 'Unknown')}")
        print(f"🔗 Connected Clients: {info.get('connected_clients', 'Unknown')}")
        
        # Clean up test data
        r.delete("test:vehicle:AV_TEST")
        print("🧹 Test data cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_simulation_data_storage():
    """Test storing simulation data in Redis"""
    print("\n📊 Testing Simulation Data Storage")
    print("=" * 50)
    
    try:
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Sample simulation data
        vehicles = ["AV_001", "AV_002", "AV_003"]
        
        for vehicle_id in vehicles:
            sensor_data = {
                "vehicle_id": vehicle_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "sensors": {
                    "camera": {"active": True, "quality": "HD"},
                    "lidar": {"range": 200, "accuracy": 0.95},
                    "gps": {"lat": 37.7749 + (hash(vehicle_id) % 100) / 10000, 
                           "lon": -122.4194 + (hash(vehicle_id) % 100) / 10000}
                },
                "metrics": {
                    "speed": 30 + (hash(vehicle_id) % 20),
                    "fuel_level": 85.5,
                    "battery": 92.3
                }
            }
            
            # Store in Redis with TTL
            key = f"vehicle:data:{vehicle_id}"
            r.setex(key, 3600, json.dumps(sensor_data))  # 1 hour TTL
            print(f"✅ Stored data for {vehicle_id}")
        
        # Retrieve and display stored data
        keys = r.keys("vehicle:data:*")
        print(f"📋 Found {len(keys)} vehicle data entries")
        
        for key in keys:
            data = json.loads(r.get(key))
            vehicle_id = data['vehicle_id']
            speed = data['metrics']['speed']
            print(f"🚗 {vehicle_id}: Speed={speed} km/h")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation data storage failed: {e}")
        return False

if __name__ == "__main__":
    print("🐳 Redis Docker Integration Test")
    print("=" * 50)
    
    # Test basic connection
    if test_redis_connection():
        print("✅ Basic Redis functionality working")
    else:
        print("❌ Basic Redis test failed")
        exit(1)
    
    # Test simulation data storage
    if test_simulation_data_storage():
        print("✅ Simulation data storage working")
    else:
        print("❌ Simulation data storage failed")
        exit(1)
    
    print("\n🎉 All Redis integration tests passed!")
    print("🔧 Redis container is ready for autonomous vehicle data")
