FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY .. .

# Create necessary directories
RUN mkdir -p data/input data/output data/simulation data/hdfs_simulation

# Expose port
EXPOSE 8050

# Run the simple dashboard (no Kafka dependencies)
CMD ["python", "dashboard/simple_dashboard.py"]
