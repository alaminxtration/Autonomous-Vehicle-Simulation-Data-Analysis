from kafka import KafkaConsumer
import json

consumer = KafkaConsumer('sensor_topic', bootstrap_servers='localhost:9092')
for message in consumer:
    data = json.loads(message.value)
    process_sensor_data(data)