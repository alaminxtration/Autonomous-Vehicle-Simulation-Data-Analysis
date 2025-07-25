FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV KAFKA_BOOTSTRAP_SERVERS=kafka:29092
ENV KAFKA_TOPIC=sensor_data
ENV VEHICLE_COUNT=5
ENV SEND_INTERVAL=1.0

# Run the producer
CMD ["python", "sensor_data_producer.py"]
