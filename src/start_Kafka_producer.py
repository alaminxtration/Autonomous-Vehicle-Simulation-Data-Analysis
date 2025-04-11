from kafka import KafkaProducer
import json

producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))
sensor_data = {"lidar": "point_cloud", "camera": "image", "timestamp": "2023-10-01T12:00:00"}
producer.send('sensor_topic', sensor_data)