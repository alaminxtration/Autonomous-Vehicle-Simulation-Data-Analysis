"""
Storage Manager for Autonomous Vehicle Simulation - No Kafka Version
Unified interface for S3, HDFS, and local storage
"""

import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import boto3
from botocore.exceptions import ClientError
import requests
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3StorageManager:
    """Manages S3 storage operations"""
    
    def __init__(self, bucket_name: str = "av-simulation-bucket", region: str = "us-west-2"):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = None
        self._setup_s3_client()
    
    def _setup_s3_client(self):
        """Setup S3 client with credentials"""
        try:
            self.s3_client = boto3.client('s3', region_name=self.region)
            logger.info(f"S3 client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"S3 client setup failed: {e}")
            self.s3_client = None
    
    def upload_data(self, data: Any, key: str) -> bool:
        """Upload data to S3"""
        if not self.s3_client:
            return False
        
        try:
            if isinstance(data, pd.DataFrame):
                # Upload DataFrame as parquet
                buffer = data.to_parquet()
                self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=buffer)
            elif isinstance(data, dict):
                # Upload dictionary as JSON
                json_data = json.dumps(data)
                self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=json_data)
            else:
                # Upload as pickle
                pickle_data = pickle.dumps(data)
                self.s3_client.put_object(Bucket=self.bucket_name, Key=key, Body=pickle_data)
            
            logger.info(f"Successfully uploaded to S3: {key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload to S3 {key}: {e}")
            return False
    
    def download_data(self, key: str) -> Optional[Any]:
        """Download data from S3"""
        if not self.s3_client:
            return None
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            data = response['Body'].read()
            
            if key.endswith('.json'):
                return json.loads(data)
            elif key.endswith('.parquet'):
                return pd.read_parquet(data)
            else:
                return pickle.loads(data)
                
        except ClientError as e:
            logger.error(f"Failed to download from S3 {key}: {e}")
            return None

class HDFSStorageManager:
    """Simplified HDFS storage using local filesystem"""
    
    def __init__(self, base_path: str = "data/hdfs_simulation"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"HDFS simulation initialized at: {self.base_path}")
    
    def upload_data(self, data: Any, path: str) -> bool:
        """Upload data to simulated HDFS"""
        try:
            full_path = self.base_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if isinstance(data, pd.DataFrame):
                if path.endswith('.parquet'):
                    data.to_parquet(full_path, index=False)
                else:
                    data.to_parquet(full_path.with_suffix('.parquet'), index=False)
            elif isinstance(data, dict):
                if path.endswith('.json'):
                    with open(full_path, 'w') as f:
                        json.dump(data, f, indent=2)
                else:
                    with open(full_path.with_suffix('.json'), 'w') as f:
                        json.dump(data, f, indent=2)
            else:
                with open(full_path.with_suffix('.pickle'), 'wb') as f:
                    pickle.dump(data, f)
            
            logger.info(f"Successfully uploaded to HDFS simulation: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload to HDFS simulation {path}: {e}")
            return False
    
    def download_data(self, path: str) -> Optional[Any]:
        """Download data from simulated HDFS"""
        try:
            full_path = self.base_path / path
            
            if not full_path.exists():
                # Try with different extensions
                for ext in ['.json', '.parquet', '.pickle']:
                    alt_path = full_path.with_suffix(ext)
                    if alt_path.exists():
                        full_path = alt_path
                        break
                else:
                    logger.warning(f"File not found in HDFS simulation: {path}")
                    return None
            
            if full_path.suffix == '.json':
                with open(full_path, 'r') as f:
                    return json.load(f)
            elif full_path.suffix == '.parquet':
                return pd.read_parquet(full_path)
            elif full_path.suffix == '.pickle':
                with open(full_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(full_path, 'r') as f:
                    return f.read()
                    
        except Exception as e:
            logger.error(f"Failed to download from HDFS simulation {path}: {e}")
            return None

class UnifiedStorageManager:
    """Unified storage manager combining S3 and HDFS"""
    
    def __init__(self, enable_s3: bool = False, enable_hdfs: bool = True):
        self.s3_manager = S3StorageManager() if enable_s3 else None
        self.hdfs_manager = HDFSStorageManager() if enable_hdfs else None
        
        if not enable_s3 and not enable_hdfs:
            logger.warning("No storage backends enabled!")
    
    def upload_data(self, data: Any, path: str, storage_type: str = "auto") -> bool:
        """Upload data to specified storage or both"""
        success = False
        
        if storage_type in ["auto", "s3"] and self.s3_manager:
            success |= self.s3_manager.upload_data(data, path)
        
        if storage_type in ["auto", "hdfs"] and self.hdfs_manager:
            success |= self.hdfs_manager.upload_data(data, path)
        
        return success
    
    def download_data(self, path: str, storage_type: str = "auto") -> Optional[Any]:
        """Download data from specified storage"""
        
        if storage_type in ["auto", "hdfs"] and self.hdfs_manager:
            data = self.hdfs_manager.download_data(path)
            if data is not None:
                return data
        
        if storage_type in ["auto", "s3"] and self.s3_manager:
            data = self.s3_manager.download_data(path)
            if data is not None:
                return data
        
        return None
    
    def list_files(self, prefix: str = "", storage_type: str = "hdfs") -> List[str]:
        """List files in storage"""
        if storage_type == "hdfs" and self.hdfs_manager:
            base_path = self.hdfs_manager.base_path / prefix
            if base_path.exists():
                return [str(p.relative_to(self.hdfs_manager.base_path)) 
                       for p in base_path.rglob("*") if p.is_file()]
        
        return []

class StreamingDataArchiver:
    """Archives streaming data for batch processing - No Kafka Version"""
    
    def __init__(self, storage_manager: UnifiedStorageManager):
        self.storage_manager = storage_manager
        self.archive_batch_size = 100
        self.current_batch = []
        logger.info("Streaming data archiver initialized (no Kafka)")
    
    def archive_sensor_data(self, sensor_data: Dict[str, Any]) -> bool:
        """Archive sensor data batch"""
        try:
            self.current_batch.append(sensor_data)
            
            if len(self.current_batch) >= self.archive_batch_size:
                return self._flush_batch()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive sensor data: {e}")
            return False
    
    def _flush_batch(self) -> bool:
        """Flush current batch to storage"""
        if not self.current_batch:
            return True
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_path = f"sensor_data/batch_{timestamp}.json"
            
            success = self.storage_manager.upload_data(self.current_batch, batch_path)
            
            if success:
                logger.info(f"Archived batch of {len(self.current_batch)} records to {batch_path}")
                self.current_batch = []
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            return False
    
    def force_flush(self) -> bool:
        """Force flush current batch"""
        return self._flush_batch()

# Test function
def test_storage_managers():
    """Test storage managers"""
    print("Testing storage managers...")
    
    # Test unified storage
    storage = UnifiedStorageManager(enable_s3=False, enable_hdfs=True)
    
    # Test data
    test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
    test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    
    # Test uploads
    json_success = storage.upload_data(test_data, "test/test_data.json")
    df_success = storage.upload_data(test_df, "test/test_df.parquet")
    
    print(f"JSON upload: {json_success}")
    print(f"DataFrame upload: {df_success}")
    
    # Test downloads
    downloaded_json = storage.download_data("test/test_data.json")
    downloaded_df = storage.download_data("test/test_df.parquet")
    
    print(f"JSON download: {downloaded_json is not None}")
    print(f"DataFrame download: {downloaded_df is not None}")
    
    # Test archiver
    archiver = StreamingDataArchiver(storage)
    archiver_success = archiver.archive_sensor_data(test_data)
    archiver.force_flush()
    
    print(f"Archiver test: {archiver_success}")
    print("Storage manager tests completed")

if __name__ == "__main__":
    test_storage_managers()
