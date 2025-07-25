import os
import logging
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
# Simplified HDFS access using requests for WebHDFS API
import requests

from kafka import KafkaConsumer
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3StorageManager:
    """AWS S3 storage management"""
    
    def __init__(self, bucket_name: str, aws_access_key: str = None, 
                 aws_secret_key: str = None, region: str = 'us-east-1'):
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        session = boto3.Session(
            aws_access_key_id=aws_access_key or os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=aws_secret_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=region
        )
        
        self.s3_client = session.client('s3')
        self.s3_resource = session.resource('s3')
        
        # Ensure bucket exists
        self.create_bucket_if_not_exists()
    
    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket {self.bucket_name} exists")
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    logger.info(f"Created S3 bucket {self.bucket_name}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket: {create_error}")
                    raise
            else:
                logger.error(f"Error accessing bucket: {e}")
                raise
    
    def upload_json(self, data: Dict[str, Any], key: str, metadata: Dict[str, str] = None) -> bool:
        """Upload JSON data to S3"""
        try:
            json_data = json.dumps(data, default=str, indent=2)
            
            extra_args = {
                'ContentType': 'application/json',
                'Metadata': metadata or {}
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json_data,
                **extra_args
            )
            
            logger.debug(f"Uploaded JSON to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload JSON to S3: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, key: str, format: str = 'parquet',
                        metadata: Dict[str, str] = None) -> bool:
        """Upload DataFrame to S3"""
        try:
            # Convert DataFrame to bytes
            if format.lower() == 'parquet':
                buffer = df.to_parquet(index=False)
                content_type = 'application/octet-stream'
            elif format.lower() == 'csv':
                buffer = df.to_csv(index=False).encode('utf-8')
                content_type = 'text/csv'
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            extra_args = {
                'ContentType': content_type,
                'Metadata': metadata or {}
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer,
                **extra_args
            )
            
            logger.debug(f"Uploaded DataFrame to s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame to S3: {e}")
            return False
    
    def upload_file(self, local_path: str, s3_key: str, metadata: Dict[str, str] = None) -> bool:
        """Upload local file to S3"""
        try:
            extra_args = {'Metadata': metadata or {}}
            
            self.s3_client.upload_file(
                local_path, 
                self.bucket_name, 
                s3_key,
                ExtraArgs=extra_args
            )
            
            logger.info(f"Uploaded {local_path} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return False
    
    def download_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            json_data = json.loads(response['Body'].read().decode('utf-8'))
            return json_data
            
        except Exception as e:
            logger.error(f"Failed to download JSON from S3: {e}")
            return None
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download file from S3 to local path"""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file from S3: {e}")
            return False
    
    def list_objects(self, prefix: str = '', max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List objects in S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )
            
            objects = []
            for obj in response.get('Contents', []):
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'etag': obj['ETag']
                })
            
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []
    
    def delete_object(self, key: str) -> bool:
        """Delete object from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            logger.info(f"Deleted s3://{self.bucket_name}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete S3 object: {e}")
            return False

class HDFSStorageManager:
    """HDFS storage management using WebHDFS REST API"""
    
    def __init__(self, hdfs_url: str = 'http://namenode:9870', user: str = 'root'):
        self.hdfs_url = hdfs_url.rstrip('/')
        self.webhdfs_url = f"{self.hdfs_url}/webhdfs/v1"
        self.user = user
        
        try:
            # Test connection by listing root directory
            response = requests.get(
                f"{self.webhdfs_url}/?op=LISTSTATUS&user.name={self.user}",
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"Connected to HDFS WebHDFS at {hdfs_url}")
            else:
                raise ConnectionError(f"HDFS connection failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to HDFS: {e}")
            # Don't raise exception, allow fallback to local storage
            logger.warning("HDFS connection failed, will use local storage as fallback")
    
    def create_directory(self, path: str) -> bool:
        """Create directory in HDFS"""
        try:
            self.client.makedirs(path, permission=755)
            logger.info(f"Created HDFS directory: {path}")
            return True
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.error(f"Failed to create HDFS directory: {e}")
                return False
            return True
    
    def upload_json(self, data: Dict[str, Any], hdfs_path: str, overwrite: bool = True) -> bool:
        """Upload JSON data to HDFS"""
        try:
            json_data = json.dumps(data, default=str, indent=2)
            
            with self.client.write(hdfs_path, overwrite=overwrite, encoding='utf-8') as writer:
                writer.write(json_data)
            
            logger.debug(f"Uploaded JSON to HDFS: {hdfs_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload JSON to HDFS: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, hdfs_path: str, format: str = 'parquet',
                        overwrite: bool = True) -> bool:
        """Upload DataFrame to HDFS"""
        try:
            if format.lower() == 'parquet':
                write_dataframe(self.client, hdfs_path, df, overwrite=overwrite)
            elif format.lower() == 'csv':
                csv_data = df.to_csv(index=False)
                with self.client.write(hdfs_path, overwrite=overwrite, encoding='utf-8') as writer:
                    writer.write(csv_data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.debug(f"Uploaded DataFrame to HDFS: {hdfs_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame to HDFS: {e}")
            return False
    
    def upload_file(self, local_path: str, hdfs_path: str, overwrite: bool = True) -> bool:
        """Upload local file to HDFS"""
        try:
            self.client.upload(hdfs_path, local_path, overwrite=overwrite)
            logger.info(f"Uploaded {local_path} to HDFS: {hdfs_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file to HDFS: {e}")
            return False
    
    def download_json(self, hdfs_path: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from HDFS"""
        try:
            with self.client.read(hdfs_path, encoding='utf-8') as reader:
                json_data = json.loads(reader.read())
            return json_data
            
        except Exception as e:
            logger.error(f"Failed to download JSON from HDFS: {e}")
            return None
    
    def download_dataframe(self, hdfs_path: str, format: str = 'parquet') -> Optional[pd.DataFrame]:
        """Download DataFrame from HDFS"""
        try:
            if format.lower() == 'parquet':
                df = read_dataframe(self.client, hdfs_path)
            elif format.lower() == 'csv':
                with self.client.read(hdfs_path, encoding='utf-8') as reader:
                    df = pd.read_csv(reader)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to download DataFrame from HDFS: {e}")
            return None
    
    def download_file(self, hdfs_path: str, local_path: str, overwrite: bool = True) -> bool:
        """Download file from HDFS to local path"""
        try:
            self.client.download(hdfs_path, local_path, overwrite=overwrite)
            logger.info(f"Downloaded HDFS {hdfs_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file from HDFS: {e}")
            return False
    
    def list_directory(self, path: str = '/') -> List[Dict[str, Any]]:
        """List contents of HDFS directory"""
        try:
            contents = []
            for item in self.client.list(path, status=True):
                contents.append({
                    'name': item[0],
                    'type': item[1]['type'],
                    'size': item[1]['length'],
                    'modification_time': item[1]['modificationTime'],
                    'permission': item[1]['permission'],
                    'owner': item[1]['owner'],
                    'group': item[1]['group']
                })
            return contents
            
        except Exception as e:
            logger.error(f"Failed to list HDFS directory: {e}")
            return []
    
    def delete_path(self, path: str, recursive: bool = False) -> bool:
        """Delete path from HDFS"""
        try:
            self.client.delete(path, recursive=recursive)
            logger.info(f"Deleted HDFS path: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete HDFS path: {e}")
            return False

class StreamingStorageService:
    """Service for storing streaming data to both S3 and HDFS"""
    
    def __init__(self, s3_bucket: str, hdfs_base_path: str = '/data/streaming',
                 kafka_servers: str = 'localhost:9092', topics: List[str] = None):
        
        # Initialize storage managers
        self.s3_manager = S3StorageManager(s3_bucket)
        self.hdfs_manager = HDFSStorageManager()
        
        # Kafka configuration
        self.kafka_servers = kafka_servers
        self.topics = topics or ['inference_results', 'processed_sensor_data']
        
        # Storage configuration
        self.hdfs_base_path = hdfs_base_path
        self.s3_base_prefix = 'streaming-data'
        
        # Batch configuration
        self.batch_size = 100
        self.batch_timeout = 60  # seconds
        
        # Storage buffers
        self.storage_buffers = {topic: [] for topic in self.topics}
        self.last_flush_time = {topic: time.time() for topic in self.topics}
        
        # Threading
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Streaming storage service initialized")
    
    def create_partitioned_path(self, base_path: str, timestamp: float, topic: str) -> str:
        """Create partitioned path based on timestamp"""
        dt = datetime.fromtimestamp(timestamp)
        return f"{base_path}/{topic}/year={dt.year}/month={dt.month:02d}/day={dt.day:02d}/hour={dt.hour:02d}"
    
    def store_batch_to_s3(self, data_batch: List[Dict[str, Any]], topic: str) -> bool:
        """Store batch of data to S3"""
        try:
            if not data_batch:
                return True
            
            # Create timestamp-based key
            timestamp = data_batch[0].get('timestamp', time.time())
            dt = datetime.fromtimestamp(timestamp)
            
            s3_key = f"{self.s3_base_prefix}/{topic}/{dt.year}/{dt.month:02d}/{dt.day:02d}/" \
                    f"{dt.hour:02d}/batch_{int(timestamp)}_{len(data_batch)}.json"
            
            # Prepare batch data
            batch_metadata = {
                'batch_size': str(len(data_batch)),
                'topic': topic,
                'timestamp': str(timestamp),
                'created_at': datetime.now().isoformat()
            }
            
            batch_data = {
                'metadata': batch_metadata,
                'records': data_batch
            }
            
            return self.s3_manager.upload_json(batch_data, s3_key, batch_metadata)
            
        except Exception as e:
            logger.error(f"Failed to store batch to S3: {e}")
            return False
    
    def store_batch_to_hdfs(self, data_batch: List[Dict[str, Any]], topic: str) -> bool:
        """Store batch of data to HDFS"""
        try:
            if not data_batch:
                return True
            
            # Convert to DataFrame
            df = pd.DataFrame(data_batch)
            
            # Create partitioned path
            timestamp = data_batch[0].get('timestamp', time.time())
            hdfs_path = self.create_partitioned_path(self.hdfs_base_path, timestamp, topic)
            
            # Create directory if needed
            self.hdfs_manager.create_directory(hdfs_path)
            
            # Store as parquet
            filename = f"batch_{int(timestamp)}_{len(data_batch)}.parquet"
            full_path = f"{hdfs_path}/{filename}"
            
            return self.hdfs_manager.upload_dataframe(df, full_path, format='parquet')
            
        except Exception as e:
            logger.error(f"Failed to store batch to HDFS: {e}")
            return False
    
    def flush_topic_buffer(self, topic: str):
        """Flush buffer for a specific topic"""
        if topic not in self.storage_buffers or not self.storage_buffers[topic]:
            return
        
        data_batch = self.storage_buffers[topic].copy()
        self.storage_buffers[topic].clear()
        self.last_flush_time[topic] = time.time()
        
        logger.info(f"Flushing {len(data_batch)} records for topic {topic}")
        
        # Store to both S3 and HDFS in parallel
        futures = []
        futures.append(self.executor.submit(self.store_batch_to_s3, data_batch, topic))
        futures.append(self.executor.submit(self.store_batch_to_hdfs, data_batch, topic))
        
        # Wait for completion
        for future in as_completed(futures):
            try:
                success = future.result(timeout=30)
                if not success:
                    logger.warning(f"Storage operation failed for topic {topic}")
            except Exception as e:
                logger.error(f"Storage operation error for topic {topic}: {e}")
    
    def add_to_buffer(self, data: Dict[str, Any], topic: str):
        """Add data to storage buffer"""
        if topic not in self.storage_buffers:
            self.storage_buffers[topic] = []
        
        self.storage_buffers[topic].append(data)
        
        # Check if buffer should be flushed
        buffer_size = len(self.storage_buffers[topic])
        time_since_flush = time.time() - self.last_flush_time[topic]
        
        if buffer_size >= self.batch_size or time_since_flush >= self.batch_timeout:
            self.flush_topic_buffer(topic)
    
    def process_kafka_stream(self):
        """Process Kafka stream and store data"""
        try:
            consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.kafka_servers,
                group_id='storage-service',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            logger.info(f"Started consuming from topics: {self.topics}")
            
            while not self.stop_event.is_set():
                try:
                    message_batch = consumer.poll(timeout_ms=1000)
                    
                    for topic_partition, messages in message_batch.items():
                        topic = topic_partition.topic
                        
                        for message in messages:
                            self.add_to_buffer(message.value, topic)
                    
                    # Check for timeouts
                    current_time = time.time()
                    for topic in self.topics:
                        if current_time - self.last_flush_time[topic] >= self.batch_timeout:
                            if self.storage_buffers[topic]:
                                self.flush_topic_buffer(topic)
                
                except Exception as e:
                    logger.error(f"Kafka processing error: {e}")
                    time.sleep(5)
            
            # Final flush
            for topic in self.topics:
                self.flush_topic_buffer(topic)
            
            consumer.close()
            
        except Exception as e:
            logger.error(f"Kafka stream processing failed: {e}")
            raise
    
    def start(self):
        """Start the streaming storage service"""
        logger.info("Starting streaming storage service...")
        
        # Start Kafka processing in a separate thread
        kafka_thread = threading.Thread(target=self.process_kafka_stream)
        kafka_thread.daemon = True
        kafka_thread.start()
        
        return kafka_thread
    
    def stop(self):
        """Stop the streaming storage service"""
        logger.info("Stopping streaming storage service...")
        self.stop_event.set()
        
        # Final flush
        for topic in self.topics:
            self.flush_topic_buffer(topic)
        
        self.executor.shutdown(wait=True)

def main():
    """Main function for running storage service"""
    # Configuration
    S3_BUCKET = os.getenv('S3_BUCKET', 'av-simulation-data')
    HDFS_BASE_PATH = os.getenv('HDFS_BASE_PATH', '/data/streaming')
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    TOPICS = os.getenv('KAFKA_TOPICS', 'inference_results,processed_sensor_data').split(',')
    
    # Create storage service
    storage_service = StreamingStorageService(
        s3_bucket=S3_BUCKET,
        hdfs_base_path=HDFS_BASE_PATH,
        kafka_servers=KAFKA_SERVERS,
        topics=TOPICS
    )
    
    try:
        # Start service
        kafka_thread = storage_service.start()
        
        # Keep running
        kafka_thread.join()
        
    except KeyboardInterrupt:
        logger.info("Stopping storage service...")
    except Exception as e:
        logger.error(f"Storage service error: {e}")
        raise
    finally:
        storage_service.stop()

if __name__ == "__main__":
    main()
