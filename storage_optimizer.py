#!/usr/bin/env python3
"""
Storage Optimizer - Reduces local storage usage for AV Simulation
Implements compression, cleanup, and efficient data management
"""

import os
import gzip
import shutil
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import tempfile
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StorageOptimizer:
    """Optimizes local storage usage for the AV simulation"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.compressed_path = self.base_path / "compressed"
        self.archive_path = self.base_path / "archive"
        self.db_path = self.base_path / "simulation.db"
        
        # Create directories
        self.compressed_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.init_database()
        
        logger.info(f"Storage optimizer initialized for {self.base_path}")
    
    def init_database(self):
        """Initialize SQLite database for efficient data storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    data_type TEXT NOT NULL,
                    compressed_data BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vehicle_timestamp 
                ON sensor_data(vehicle_id, timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON sensor_data(timestamp)
            """)
            
            logger.info("Database initialized")
    
    def compress_json_file(self, file_path: Path) -> Optional[Path]:
        """Compress a JSON file using gzip"""
        try:
            compressed_path = self.compressed_path / f"{file_path.stem}.json.gz"
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Get compression ratio
            original_size = file_path.stat().st_size
            compressed_size = compressed_path.stat().st_size
            ratio = (1 - compressed_size / original_size) * 100
            
            logger.info(f"Compressed {file_path.name}: {ratio:.1f}% reduction")
            return compressed_path
            
        except Exception as e:
            logger.error(f"Failed to compress {file_path}: {e}")
            return None
    
    def store_data_in_db(self, data: Dict[str, Any], vehicle_id: str, data_type: str = "sensor"):
        """Store data efficiently in SQLite database"""
        try:
            # Compress data as JSON
            json_data = json.dumps(data, separators=(',', ':'))  # Compact JSON
            compressed_data = gzip.compress(json_data.encode('utf-8'))
            
            metadata = {
                'original_size': len(json_data),
                'compressed_size': len(compressed_data),
                'compression_ratio': (1 - len(compressed_data) / len(json_data)) * 100
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO sensor_data 
                    (vehicle_id, timestamp, data_type, compressed_data, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    vehicle_id,
                    data.get('timestamp', datetime.now().timestamp()),
                    data_type,
                    compressed_data,
                    json.dumps(metadata)
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store data in database: {e}")
            return False
    
    def get_data_from_db(self, vehicle_id: str = None, 
                        start_time: float = None, 
                        end_time: float = None,
                        limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve data from database with optional filtering"""
        try:
            query = "SELECT vehicle_id, timestamp, compressed_data FROM sensor_data WHERE 1=1"
            params = []
            
            if vehicle_id:
                query += " AND vehicle_id = ?"
                params.append(vehicle_id)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                results = []
                
                for row in cursor.fetchall():
                    vehicle_id, timestamp, compressed_data = row
                    
                    # Decompress data
                    json_data = gzip.decompress(compressed_data).decode('utf-8')
                    data = json.loads(json_data)
                    
                    results.append({
                        'vehicle_id': vehicle_id,
                        'timestamp': timestamp,
                        **data
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve data from database: {e}")
            return []
    
    def cleanup_old_files(self, days_old: int = 7):
        """Remove files older than specified days"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cutoff_timestamp = cutoff_date.timestamp()
        
        removed_count = 0
        freed_space = 0
        
        # Clean up JSON files
        for json_file in self.base_path.glob("**/*.json"):
            try:
                if json_file.stat().st_mtime < cutoff_timestamp:
                    file_size = json_file.stat().st_size
                    json_file.unlink()
                    removed_count += 1
                    freed_space += file_size
            except Exception as e:
                logger.warning(f"Could not remove {json_file}: {e}")
        
        # Clean up old database entries
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM sensor_data 
                WHERE timestamp < ?
            """, (cutoff_timestamp,))
        
        logger.info(f"Cleaned up {removed_count} files, freed {freed_space / (1024*1024):.2f} MB")
        return removed_count, freed_space
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get current storage usage statistics"""
        try:
            total_size = 0
            file_count = 0
            
            for file_path in self.base_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            # Database stats
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM sensor_data")
                db_records = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT SUM(LENGTH(compressed_data)) FROM sensor_data
                """)
                db_size = cursor.fetchone()[0] or 0
            
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'total_files': file_count,
                'database_records': db_records,
                'database_size_mb': db_size / (1024 * 1024),
                'storage_path': str(self.base_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def compress_existing_data(self):
        """Compress all existing JSON files"""
        json_files = list(self.base_path.glob("**/*.json"))
        compressed_count = 0
        total_savings = 0
        
        for json_file in json_files:
            try:
                # Skip if already compressed
                compressed_name = self.compressed_path / f"{json_file.stem}.json.gz"
                if compressed_name.exists():
                    continue
                
                original_size = json_file.stat().st_size
                compressed_path = self.compress_json_file(json_file)
                
                if compressed_path:
                    compressed_size = compressed_path.stat().st_size
                    savings = original_size - compressed_size
                    total_savings += savings
                    compressed_count += 1
                    
                    # Remove original if compression successful
                    json_file.unlink()
                    
            except Exception as e:
                logger.warning(f"Failed to compress {json_file}: {e}")
        
        logger.info(f"Compressed {compressed_count} files, saved {total_savings / (1024*1024):.2f} MB")
        return compressed_count, total_savings
    
    def migrate_to_efficient_storage(self):
        """Migrate existing data to more efficient storage"""
        json_files = list(self.base_path.glob("**/*.json"))
        migrated_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle both single records and arrays
                if isinstance(data, list):
                    for record in data:
                        vehicle_id = record.get('vehicle_id', 'unknown')
                        self.store_data_in_db(record, vehicle_id)
                        migrated_count += 1
                else:
                    vehicle_id = data.get('vehicle_id', 'unknown')
                    self.store_data_in_db(data, vehicle_id)
                    migrated_count += 1
                
                # Move original to archive
                archive_path = self.archive_path / json_file.name
                shutil.move(str(json_file), str(archive_path))
                
            except Exception as e:
                logger.warning(f"Failed to migrate {json_file}: {e}")
        
        logger.info(f"Migrated {migrated_count} records to efficient storage")
        return migrated_count
    
    def export_data_for_analysis(self, vehicle_id: str = None, 
                                hours_back: int = 24) -> Optional[pd.DataFrame]:
        """Export recent data as DataFrame for analysis"""
        try:
            start_time = (datetime.now() - timedelta(hours=hours_back)).timestamp()
            data = self.get_data_from_db(
                vehicle_id=vehicle_id,
                start_time=start_time,
                limit=10000
            )
            
            if data:
                df = pd.DataFrame(data)
                logger.info(f"Exported {len(df)} records for analysis")
                return df
            else:
                logger.warning("No data found for export")
                return None
                
        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return None

def optimize_storage():
    """Main storage optimization function"""
    optimizer = StorageOptimizer()
    
    print("🗂️  AV Simulation Storage Optimizer")
    print("=" * 50)
    
    # Get current stats
    stats = optimizer.get_storage_stats()
    print(f"📊 Current storage usage: {stats.get('total_size_mb', 0):.2f} MB")
    print(f"📁 Total files: {stats.get('total_files', 0)}")
    print(f"🗃️  Database records: {stats.get('database_records', 0)}")
    
    # Compress existing data
    print("\n🗜️  Compressing existing files...")
    compressed_count, savings = optimizer.compress_existing_data()
    
    # Migrate to efficient storage
    print("\n📦 Migrating to efficient storage...")
    migrated_count = optimizer.migrate_to_efficient_storage()
    
    # Clean up old files
    print("\n🧹 Cleaning up old files...")
    removed_count, freed_space = optimizer.cleanup_old_files(days_old=3)
    
    # Final stats
    final_stats = optimizer.get_storage_stats()
    print(f"\n✅ Optimization complete!")
    print(f"💾 Storage reduced to: {final_stats.get('total_size_mb', 0):.2f} MB")
    print(f"🗜️  Compression savings: {savings / (1024*1024):.2f} MB")
    print(f"🗃️  Records in database: {final_stats.get('database_records', 0)}")
    
    return optimizer

if __name__ == "__main__":
    optimize_storage()
