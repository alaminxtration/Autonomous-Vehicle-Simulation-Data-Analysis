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
# Simple file-based storage for development/demo
import shutil

from kafka import KafkaConsumer
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3StorageManager:
    """AWS S3 storage management"""
    
    def __init__(self, bucket_name: str, region: str = 'us-east-1', 
                 access_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.bucket_name = bucket_name
        self.region = region
        
        # Initialize S3 client
        try:
            if access_key and secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=access_key,
                    aws_secret_access_key=secret_key,
                    region_name=region
                )
            else:
                # Use default credentials (AWS CLI, IAM role, etc.)
                self.s3_client = boto3.client('s3', region_name=region)
            
            # Test connection by checking if bucket exists
            self.s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Connected to S3 bucket: {bucket_name}")
            
        except NoCredentialsError:
            logger.warning("AWS credentials not found. S3 functionality will be limited.")
            self.s3_client = None
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.warning(f"S3 bucket {bucket_name} not found. Will create if needed.")
            else:
                logger.error(f"Failed to connect to S3: {e}")
            self.s3_client = None
        except Exception as e:
            logger.error(f"S3 initialization error: {e}")
            self.s3_client = None
    
    def create_bucket(self) -> bool:
        """Create S3 bucket if it doesn't exist"""
        if not self.s3_client:
            return False
            
        try:
            if self.region != 'us-east-1':
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            else:
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            
            logger.info(f"Created S3 bucket: {self.bucket_name}")
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'BucketAlreadyOwnedByYou':
                return True
            else:
                logger.error(f"Failed to create S3 bucket: {e}")
                return False
    
    def upload_json(self, data: Dict[str, Any], s3_key: str) -> bool:
        """Upload JSON data to S3"""
        if not self.s3_client:
            return False
            
        try:
            json_data = json.dumps(data, default=str, indent=2)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data,
                ContentType='application/json'
            )
            
            logger.debug(f"Uploaded JSON to S3: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload JSON to S3: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, s3_key: str, format: str = 'parquet') -> bool:
        """Upload DataFrame to S3"""
        if not self.s3_client:
            return False
            
        try:
            if format.lower() == 'parquet':
                parquet_buffer = df.to_parquet(index=False)
                content_type = 'application/octet-stream'
            elif format.lower() == 'csv':
                parquet_buffer = df.to_csv(index=False).encode('utf-8')
                content_type = 'text/csv'
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=parquet_buffer,
                ContentType=content_type
            )
            
            logger.debug(f"Uploaded DataFrame to S3: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame to S3: {e}")
            return False
    
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload local file to S3"""
        if not self.s3_client:
            return False
            
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logger.info(f"Uploaded {local_path} to S3: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file to S3: {e}")
            return False
    
    def download_json(self, s3_key: str) -> Optional[Dict[str, Any]]:
        """Download JSON data from S3"""
        if not self.s3_client:
            return None
            
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            json_data = json.loads(response['Body'].read().decode('utf-8'))
            
            logger.debug(f"Downloaded JSON from S3: s3://{self.bucket_name}/{s3_key}")
            return json_data
            
        except Exception as e:
            logger.error(f"Failed to download JSON from S3: {e}")
            return None
    
    def download_dataframe(self, s3_key: str, format: str = 'parquet') -> Optional[pd.DataFrame]:
        """Download DataFrame from S3"""
        if not self.s3_client:
            return None
            
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            
            if format.lower() == 'parquet':
                df = pd.read_parquet(response['Body'])
            elif format.lower() == 'csv':
                df = pd.read_csv(response['Body'])
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.debug(f"Downloaded DataFrame from S3: s3://{self.bucket_name}/{s3_key}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download DataFrame from S3: {e}")
            return None
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """Download file from S3 to local path"""
        if not self.s3_client:
            return False
            
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file from S3: {e}")
            return False
    
    def list_objects(self, prefix: str = '') -> List[str]:
        """List objects in S3 bucket with optional prefix"""
        if not self.s3_client:
            return []
            
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            objects = []
            if 'Contents' in response:
                objects = [obj['Key'] for obj in response['Contents']]
            
            logger.debug(f"Listed {len(objects)} objects from S3 with prefix '{prefix}'")
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            return []
    
    def delete_object(self, s3_key: str) -> bool:
        """Delete object from S3"""
        if not self.s3_client:
            return False
            
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Deleted S3 object: s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete S3 object: {e}")
            return False

class HDFSStorageManager:
    """File-based storage manager (HDFS simulation for development)"""
    
    def __init__(self, base_path: str = './data/hdfs', user: str = 'root'):
        self.base_path = Path(base_path)
        self.user = user
        
        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized HDFS storage at: {self.base_path.absolute()}")
    
    def create_directory(self, path: str) -> bool:
        """Create directory"""
        try:
            full_path = self.base_path / path.lstrip('/')
            full_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {full_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create directory: {e}")
            return False
    
    def upload_json(self, data: Dict[str, Any], hdfs_path: str, overwrite: bool = True) -> bool:
        """Upload JSON data"""
        try:
            full_path = self.base_path / hdfs_path.lstrip('/')
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if full_path.exists() and not overwrite:
                logger.warning(f"File exists and overwrite=False: {full_path}")
                return False
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, default=str, indent=2)
            
            logger.debug(f"Uploaded JSON to: {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload JSON: {e}")
            return False
    
    def upload_dataframe(self, df: pd.DataFrame, hdfs_path: str, format: str = 'parquet',
                        overwrite: bool = True) -> bool:
        """Upload DataFrame"""
        try:
            full_path = self.base_path / hdfs_path.lstrip('/')
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if full_path.exists() and not overwrite:
                logger.warning(f"File exists and overwrite=False: {full_path}")
                return False
            
            if format.lower() == 'parquet':
                df.to_parquet(full_path, index=False)
            elif format.lower() == 'csv':
                df.to_csv(full_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.debug(f"Uploaded DataFrame to: {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload DataFrame: {e}")
            return False
    
    def upload_file(self, local_path: str, hdfs_path: str, overwrite: bool = True) -> bool:
        """Upload local file"""
        try:
            full_path = self.base_path / hdfs_path.lstrip('/')
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if full_path.exists() and not overwrite:
                logger.warning(f"File exists and overwrite=False: {full_path}")
                return False
            
            shutil.copy2(local_path, full_path)
            logger.info(f"Uploaded {local_path} to: {full_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            return False
    
    def download_json(self, hdfs_path: str) -> Optional[Dict[str, Any]]:
        """Download JSON data"""
        try:
            full_path = self.base_path / hdfs_path.lstrip('/')
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                return None
            
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.debug(f"Downloaded JSON from: {full_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download JSON: {e}")
            return None
    
    def download_dataframe(self, hdfs_path: str, format: str = 'parquet') -> Optional[pd.DataFrame]:
        """Download DataFrame"""
        try:
            full_path = self.base_path / hdfs_path.lstrip('/')
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                return None
            
            if format.lower() == 'parquet':
                df = pd.read_parquet(full_path)
            elif format.lower() == 'csv':
                df = pd.read_csv(full_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.debug(f"Downloaded DataFrame from: {full_path}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download DataFrame: {e}")
            return None
    
    def download_file(self, hdfs_path: str, local_path: str) -> bool:
        """Download file to local path"""
        try:
            full_path = self.base_path / hdfs_path.lstrip('/')
            
            if not full_path.exists():
                logger.warning(f"File not found: {full_path}")
                return False
            
            shutil.copy2(full_path, local_path)
            logger.info(f"Downloaded {full_path} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return False
    
    def list_files(self, hdfs_path: str = '') -> List[str]:
        """List files in directory"""
        try:
            full_path = self.base_path / hdfs_path.lstrip('/')
            
            if not full_path.exists():
                return []
            
            files = []
            for item in full_path.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(self.base_path)
                    files.append(str(rel_path))
            
            logger.debug(f"Listed {len(files)} files from: {full_path}")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def delete_file(self, hdfs_path: str) -> bool:
        """Delete file"""
        try:
            full_path = self.base_path / hdfs_path.lstrip('/')
            
            if full_path.exists():
                if full_path.is_file():
                    full_path.unlink()
                else:
                    shutil.rmtree(full_path)
                logger.info(f"Deleted: {full_path}")
                return True
            else:
                logger.warning(f"File not found: {full_path}")
                return False
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False

class UnifiedStorageManager:
    """Unified storage manager supporting both S3 and HDFS"""
    
    def __init__(self, 
                 s3_bucket: Optional[str] = None,
                 s3_region: str = 'us-east-1',
                 hdfs_path: str = './data/hdfs',
                 enable_s3: bool = True,
                 enable_hdfs: bool = True):
        
        self.storage_managers = {}
        
        # Initialize S3 if enabled
        if enable_s3 and s3_bucket:
            try:
                self.storage_managers['s3'] = S3StorageManager(
                    bucket_name=s3_bucket,
                    region=s3_region
                )
                logger.info("S3 storage manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize S3: {e}")
        
        # Initialize HDFS if enabled
        if enable_hdfs:
            try:
                self.storage_managers['hdfs'] = HDFSStorageManager(base_path=hdfs_path)
                logger.info("HDFS storage manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize HDFS: {e}")
        
        self.default_storage = 'hdfs' if 'hdfs' in self.storage_managers else 's3'
    
    def upload_data(self, data: Any, path: str, storage_type: str = None, 
                   format: str = 'auto', **kwargs) -> bool:
        """Upload data to specified storage"""
        storage_type = storage_type or self.default_storage
        
        if storage_type not in self.storage_managers:
            logger.error(f"Storage type {storage_type} not available")
            return False
        
        manager = self.storage_managers[storage_type]
        
        # Determine data type and upload method
        if isinstance(data, dict):
            return manager.upload_json(data, path, **kwargs)
        elif isinstance(data, pd.DataFrame):
            fmt = format if format != 'auto' else 'parquet'
            return manager.upload_dataframe(data, path, format=fmt, **kwargs)
        elif isinstance(data, (str, Path)) and Path(data).exists():
            return manager.upload_file(str(data), path, **kwargs)
        else:
            logger.error(f"Unsupported data type: {type(data)}")
            return False
    
    def download_data(self, path: str, storage_type: str = None, format: str = 'auto'):
        """Download data from specified storage"""
        storage_type = storage_type or self.default_storage
        
        if storage_type not in self.storage_managers:
            logger.error(f"Storage type {storage_type} not available")
            return None
        
        manager = self.storage_managers[storage_type]
        
        # Try to determine format from file extension
        if format == 'auto':
            if path.endswith('.json'):
                format = 'json'
            elif path.endswith(('.parquet', '.pq')):
                format = 'parquet'
            elif path.endswith('.csv'):
                format = 'csv'
            else:
                format = 'json'  # Default
        
        if format == 'json':
            return manager.download_json(path)
        elif format in ['parquet', 'csv']:
            return manager.download_dataframe(path, format=format)
        else:
            logger.error(f"Unsupported format: {format}")
            return None

class StreamingDataArchiver:
    """Archive streaming data from Kafka to storage"""
    
    def __init__(self, storage_manager: UnifiedStorageManager,
                 bootstrap_servers: str = 'localhost:9092',
                 topics: List[str] = None,
                 batch_size: int = 1000,
                 batch_timeout: int = 300):  # 5 minutes
        
        self.storage_manager = storage_manager
        self.bootstrap_servers = bootstrap_servers
        self.topics = topics or ['sensor_data', 'processed_sensor_data', 'inference_results']
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.running = False
        
        # Initialize consumers for each topic
        self.consumers = {}
        for topic in self.topics:
            try:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=bootstrap_servers,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    group_id=f'archiver_{topic}',
                    auto_offset_reset='latest',
                    enable_auto_commit=True
                )
                self.consumers[topic] = consumer
                logger.info(f"Created consumer for topic: {topic}")
            except Exception as e:
                logger.error(f"Failed to create consumer for {topic}: {e}")
    
    def start_archiving(self):
        """Start archiving process"""
        self.running = True
        
        # Start archiving thread for each topic
        threads = []
        for topic in self.topics:
            if topic in self.consumers:
                thread = threading.Thread(
                    target=self._archive_topic,
                    args=(topic,),
                    daemon=True
                )
                thread.start()
                threads.append(thread)
                logger.info(f"Started archiving thread for topic: {topic}")
        
        return threads
    
    def stop_archiving(self):
        """Stop archiving process"""
        self.running = False
        
        # Close consumers
        for consumer in self.consumers.values():
            consumer.close()
        
        logger.info("Stopped data archiving")
    
    def _archive_topic(self, topic: str):
        """Archive data from a specific topic"""
        consumer = self.consumers[topic]
        batch = []
        last_save_time = time.time()
        
        logger.info(f"Starting archival for topic: {topic}")
        
        try:
            while self.running:
                # Poll for messages
                message_batch = consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        batch.append({
                            'timestamp': message.timestamp / 1000,  # Convert to seconds
                            'key': message.key.decode('utf-8') if message.key else None,
                            'value': message.value,
                            'topic': message.topic,
                            'partition': message.partition,
                            'offset': message.offset
                        })
                
                # Save batch if conditions met
                current_time = time.time()
                if (len(batch) >= self.batch_size or 
                    (batch and current_time - last_save_time >= self.batch_timeout)):
                    
                    self._save_batch(topic, batch)
                    batch = []
                    last_save_time = current_time
                
        except Exception as e:
            logger.error(f"Error in archival thread for {topic}: {e}")
        finally:
            # Save any remaining batch
            if batch:
                self._save_batch(topic, batch)
    
    def _save_batch(self, topic: str, batch: List[Dict[str, Any]]):
        """Save batch of messages to storage"""
        try:
            # Create DataFrame from batch
            df = pd.DataFrame(batch)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"streaming/{topic}/{timestamp}.parquet"
            
            # Upload to storage
            success = self.storage_manager.upload_data(
                df, filename, format='parquet'
            )
            
            if success:
                logger.info(f"Archived {len(batch)} messages from {topic} to {filename}")
            else:
                logger.error(f"Failed to archive batch for {topic}")
                
        except Exception as e:
            logger.error(f"Error saving batch for {topic}: {e}")

def main():
    """Main function for testing storage manager"""
    
    # Configuration
    S3_BUCKET = os.getenv('S3_BUCKET', 'av-simulation-data')
    HDFS_PATH = os.getenv('HDFS_BASE_PATH', './data/hdfs')
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    TOPICS = os.getenv('KAFKA_TOPICS', 'inference_results,processed_sensor_data').split(',')
    
    logger.info("Starting Storage Manager")
    logger.info(f"S3 Bucket: {S3_BUCKET}")
    logger.info(f"HDFS Path: {HDFS_PATH}")
    logger.info(f"Kafka Servers: {KAFKA_SERVERS}")
    logger.info(f"Topics: {TOPICS}")
    
    # Initialize unified storage manager
    storage_manager = UnifiedStorageManager(
        s3_bucket=S3_BUCKET,
        hdfs_path=HDFS_PATH,
        enable_s3=True,
        enable_hdfs=True
    )
    
    # Initialize streaming data archiver
    archiver = StreamingDataArchiver(
        storage_manager=storage_manager,
        bootstrap_servers=KAFKA_SERVERS,
        topics=TOPICS,
        batch_size=int(os.getenv('BATCH_SIZE', '1000')),
        batch_timeout=int(os.getenv('BATCH_TIMEOUT', '300'))
    )
    
    # Start archiving
    try:
        threads = archiver.start_archiving()
        
        # Keep running
        while True:
            time.sleep(60)
            logger.info("Storage manager running...")
            
    except KeyboardInterrupt:
        logger.info("Shutting down storage manager...")
        archiver.stop_archiving()
    except Exception as e:
        logger.error(f"Storage manager error: {e}")
        archiver.stop_archiving()

if __name__ == "__main__":
    main()
