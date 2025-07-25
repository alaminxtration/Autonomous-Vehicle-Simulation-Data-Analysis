import os
import json
import logging
import pickle
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

import redis
from concurrent.futures import ThreadPoolExecutor
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelInferenceService:
    """Real-time model inference service"""
    
    def __init__(self, mlflow_uri: str, model_run_id: str, redis_host: str = "localhost"):
        self.mlflow_uri = mlflow_uri
        self.model_run_id = model_run_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize MLflow
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = MlflowClient(mlflow_uri)
        
        # Initialize Redis for caching
        self.redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
        # Load models and scalers
        self.models = {}
        self.scalers = {}
        self.load_models()
        
        logger.info(f"Inference service initialized on device: {self.device}")
    
    def load_models(self):
        """Load trained models from MLflow"""
        try:
            # Load risk prediction model
            risk_model_uri = f"runs:/{self.model_run_id}/risk_prediction_model"
            self.models['risk_prediction'] = mlflow.pytorch.load_model(
                risk_model_uri, map_location=self.device
            )
            self.models['risk_prediction'].eval()
            
            # Load anomaly detection model  
            anomaly_model_uri = f"runs:/{self.model_run_id}/anomaly_detection_model"
            self.models['anomaly_detection'] = mlflow.pytorch.load_model(
                anomaly_model_uri, map_location=self.device
            )
            self.models['anomaly_detection'].eval()
            
            # Load scalers
            risk_scaler_path = self.client.download_artifacts(self.model_run_id, "feature_scaler.pkl")
            with open(risk_scaler_path, "rb") as f:
                self.scalers['risk_prediction'] = pickle.load(f)
            
            anomaly_scaler_path = self.client.download_artifacts(self.model_run_id, "anomaly_scaler.pkl")
            with open(anomaly_scaler_path, "rb") as f:
                self.scalers['anomaly_detection'] = pickle.load(f)
            
            logger.info("Models and scalers loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def extract_features_for_risk_prediction(self, sensor_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for risk prediction"""
        try:
            features = []
            
            # Basic features
            features.append(sensor_data.get('calculated_speed', 0))
            features.append(sensor_data.get('lidar_stats', {}).get('point_count', 0))
            features.append(sensor_data.get('lidar_stats', {}).get('avg_intensity', 0))
            features.append(sensor_data.get('lidar_stats', {}).get('max_distance', 0))
            features.append(sensor_data.get('lidar_density', 0))
            features.append(sensor_data.get('weather_severity', 0))
            features.append(sensor_data.get('speed_variance', 0))
            
            # Time-based features
            timestamp = sensor_data.get('timestamp', time.time())
            dt = datetime.fromtimestamp(timestamp)
            features.append(dt.hour)
            
            features.append(sensor_data.get('anomaly_count', 0))
            
            return np.array(features, dtype=np.float32).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return np.zeros((1, 9), dtype=np.float32)
    
    def predict_risk_level(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict risk level for sensor data"""
        try:
            # Extract features
            features = self.extract_features_for_risk_prediction(sensor_data)
            
            # Scale features
            features_scaled = self.scalers['risk_prediction'].transform(features)
            
            # Convert to tensor
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.models['risk_prediction'](features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            risk_levels = ['low', 'medium', 'high']
            predicted_risk = risk_levels[min(predicted_class, len(risk_levels) - 1)]
            
            return {
                'predicted_risk_level': predicted_risk,
                'confidence': round(confidence, 4),
                'probabilities': {
                    risk_levels[i]: round(probabilities[0][i].item(), 4)
                    for i in range(len(risk_levels))
                }
            }
            
        except Exception as e:
            logger.error(f"Risk prediction error: {e}")
            return {
                'predicted_risk_level': 'unknown',
                'confidence': 0.0,
                'probabilities': {}
            }
    
    def detect_anomaly(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in sensor data"""
        try:
            # Extract numerical features
            features = []
            for key, value in sensor_data.items():
                if isinstance(value, (int, float)):
                    features.append(value)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float)):
                            features.append(subvalue)
            
            # Limit to expected number of features
            features = features[:20]
            if len(features) < 20:
                features.extend([0.0] * (20 - len(features)))
            
            features_array = np.array(features, dtype=np.float32).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scalers['anomaly_detection'].transform(features_array)
            
            # Convert to tensor
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            
            # Get reconstruction
            with torch.no_grad():
                reconstructed = self.models['anomaly_detection'](features_tensor)
                reconstruction_error = torch.mean((features_tensor - reconstructed) ** 2).item()
            
            # Determine anomaly (threshold can be tuned)
            anomaly_threshold = 0.1
            is_anomaly = reconstruction_error > anomaly_threshold
            
            return {
                'is_anomaly': is_anomaly,
                'reconstruction_error': round(reconstruction_error, 6),
                'anomaly_score': min(reconstruction_error / anomaly_threshold, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {
                'is_anomaly': False,
                'reconstruction_error': 0.0,
                'anomaly_score': 0.0
            }
    
    def cache_prediction(self, vehicle_id: str, prediction: Dict[str, Any], ttl: int = 300):
        """Cache prediction results"""
        try:
            cache_key = f"prediction:{vehicle_id}:{int(time.time())}"
            self.redis_client.setex(
                cache_key, 
                ttl, 
                json.dumps(prediction, default=str)
            )
        except Exception as e:
            logger.warning(f"Failed to cache prediction: {e}")
    
    def get_cached_prediction(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction"""
        try:
            # Look for recent predictions
            pattern = f"prediction:{vehicle_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                # Get the most recent one
                latest_key = max(keys)
                cached_data = self.redis_client.get(latest_key)
                if cached_data:
                    return json.loads(cached_data)
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get cached prediction: {e}")
            return None

class KafkaInferenceProcessor:
    """Kafka-based real-time inference processor"""
    
    def __init__(self, kafka_servers: str, input_topic: str, output_topic: str,
                 mlflow_uri: str, model_run_id: str, group_id: str = "inference-service"):
        
        self.kafka_servers = kafka_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.group_id = group_id
        
        # Initialize inference service
        self.inference_service = ModelInferenceService(mlflow_uri, model_run_id)
        
        # Initialize Kafka consumer and producer
        self.consumer = None
        self.producer = None
        self.setup_kafka()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Metrics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = time.time()
    
    def setup_kafka(self):
        """Setup Kafka consumer and producer"""
        try:
            self.consumer = KafkaConsumer(
                self.input_topic,
                bootstrap_servers=self.kafka_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda m: m.decode('utf-8') if m else None,
                auto_offset_reset='latest',
                enable_auto_commit=True,
                consumer_timeout_ms=1000
            )
            
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: str(k).encode('utf-8'),
                retries=3,
                acks='all'
            )
            
            logger.info(f"Kafka setup complete. Consuming from {self.input_topic}, "
                       f"producing to {self.output_topic}")
            
        except Exception as e:
            logger.error(f"Kafka setup failed: {e}")
            raise
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single sensor data record"""
        try:
            vehicle_id = sensor_data.get('vehicle_id')
            timestamp = sensor_data.get('timestamp', time.time())
            
            # Check cache first
            cached_result = self.inference_service.get_cached_prediction(vehicle_id)
            if cached_result and time.time() - cached_result.get('timestamp', 0) < 30:
                logger.debug(f"Using cached prediction for vehicle {vehicle_id}")
                return cached_result
            
            # Perform inference
            risk_prediction = self.inference_service.predict_risk_level(sensor_data)
            anomaly_detection = self.inference_service.detect_anomaly(sensor_data)
            
            # Combine results
            prediction_result = {
                'vehicle_id': vehicle_id,
                'timestamp': timestamp,
                'inference_timestamp': time.time(),
                'original_data': sensor_data,
                'risk_prediction': risk_prediction,
                'anomaly_detection': anomaly_detection,
                'processing_latency_ms': round((time.time() - timestamp) * 1000, 2)
            }
            
            # Cache result
            self.inference_service.cache_prediction(vehicle_id, prediction_result)
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            self.error_count += 1
            return {
                'vehicle_id': sensor_data.get('vehicle_id', 'unknown'),
                'timestamp': sensor_data.get('timestamp', time.time()),
                'error': str(e),
                'processing_failed': True
            }
    
    def send_prediction(self, prediction: Dict[str, Any]):
        """Send prediction result to output topic"""
        try:
            key = f"{prediction['vehicle_id']}_{int(prediction['timestamp'])}"
            
            future = self.producer.send(
                self.output_topic,
                key=key,
                value=prediction
            )
            
            # Wait for confirmation
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Sent prediction to {record_metadata.topic}:"
                        f"{record_metadata.partition}:{record_metadata.offset}")
            
        except KafkaError as e:
            logger.error(f"Failed to send prediction: {e}")
            self.error_count += 1
    
    def log_metrics(self):
        """Log processing metrics"""
        runtime = time.time() - self.start_time
        rate = self.processed_count / runtime if runtime > 0 else 0
        
        logger.info(f"Processed: {self.processed_count}, "
                   f"Errors: {self.error_count}, "
                   f"Rate: {rate:.2f} msg/sec, "
                   f"Runtime: {runtime:.1f}s")
    
    def run(self):
        """Run the inference processor"""
        logger.info("Starting real-time inference processor...")
        
        try:
            last_metrics_log = time.time()
            
            for message in self.consumer:
                try:
                    # Process sensor data
                    sensor_data = message.value
                    prediction = self.process_sensor_data(sensor_data)
                    
                    # Send prediction result
                    self.send_prediction(prediction)
                    
                    self.processed_count += 1
                    
                    # Log metrics periodically
                    if time.time() - last_metrics_log > 60:  # Every minute
                        self.log_metrics()
                        last_metrics_log = time.time()
                    
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    self.error_count += 1
                    
        except KeyboardInterrupt:
            logger.info("Stopping inference processor...")
        except Exception as e:
            logger.error(f"Processor error: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
            self.executor.shutdown(wait=True)
            
            # Final metrics
            self.log_metrics()
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

def main():
    """Main function"""
    # Configuration
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    INPUT_TOPIC = os.getenv('INPUT_TOPIC', 'processed_sensor_data')
    OUTPUT_TOPIC = os.getenv('OUTPUT_TOPIC', 'inference_results')
    MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    MODEL_RUN_ID = os.getenv('MODEL_RUN_ID', 'latest')
    REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
    
    # Get latest model run if not specified
    if MODEL_RUN_ID == 'latest':
        client = MlflowClient(MLFLOW_URI)
        experiment = client.get_experiment_by_name("risk_prediction")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                MODEL_RUN_ID = runs[0].info.run_id
                logger.info(f"Using latest model run: {MODEL_RUN_ID}")
    
    # Create and run processor
    processor = KafkaInferenceProcessor(
        kafka_servers=KAFKA_SERVERS,
        input_topic=INPUT_TOPIC,
        output_topic=OUTPUT_TOPIC,
        mlflow_uri=MLFLOW_URI,
        model_run_id=MODEL_RUN_ID,
        group_id="inference-service"
    )
    
    processor.run()

if __name__ == "__main__":
    main()
