import os
import sys
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, when, isnan, isnull, count, sum as spark_sum,
    avg, max as spark_max, min as spark_min, stddev,
    from_json, to_json, struct, array, explode,
    udf, window, current_timestamp, unix_timestamp,
    regexp_replace, split, size, collect_list
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, ArrayType, MapType, TimestampType, BooleanType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler, MinMaxScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataProcessor:
    """Spark-based sensor data preprocessing pipeline"""
    
    def __init__(self, app_name: str = "SensorDataProcessor"):
        self.spark = self._create_spark_session(app_name)
        self.sensor_schema = self._define_sensor_schema()
        
    def _create_spark_session(self, app_name: str) -> SparkSession:
        """Create Spark session with optimized configurations"""
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.parquet.compression.codec", "snappy") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/spark-checkpoints") \
            .getOrCreate()
        
        spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Created Spark session: {app_name}")
        return spark
    
    def _define_sensor_schema(self) -> StructType:
        """Define schema for sensor data"""
        lidar_point_schema = StructType([
            StructField("x", DoubleType(), True),
            StructField("y", DoubleType(), True),
            StructField("z", DoubleType(), True),
            StructField("intensity", DoubleType(), True),
            StructField("timestamp", DoubleType(), True)
        ])
        
        camera_frame_schema = StructType([
            StructField("frame_id", StringType(), True),
            StructField("timestamp", DoubleType(), True),
            StructField("camera_position", StringType(), True),
            StructField("width", IntegerType(), True),
            StructField("height", IntegerType(), True),
            StructField("format", StringType(), True),
            StructField("data_size", IntegerType(), True),
            StructField("exposure", DoubleType(), True),
            StructField("iso", IntegerType(), True),
            StructField("focal_length", DoubleType(), True),
            StructField("aperture", DoubleType(), True)
        ])
        
        return StructType([
            StructField("vehicle_id", StringType(), False),
            StructField("timestamp", DoubleType(), False),
            StructField("processing_timestamp", StringType(), True),
            StructField("location", MapType(StringType(), DoubleType()), False),
            StructField("velocity", MapType(StringType(), DoubleType()), False),
            StructField("calculated_speed", DoubleType(), True),
            StructField("lidar_points", ArrayType(lidar_point_schema), True),
            StructField("lidar_stats", MapType(StringType(), DoubleType()), True),
            StructField("camera_frames", ArrayType(camera_frame_schema), True),
            StructField("imu_data", MapType(StringType(), DoubleType()), True),
            StructField("gps_data", MapType(StringType(), DoubleType()), True),
            StructField("weather_conditions", MapType(StringType(), StringType()), True),
            StructField("risk_assessment", MapType(StringType(), StringType()), True),
            StructField("anomalies", ArrayType(MapType(StringType(), StringType())), True),
            StructField("anomaly_count", IntegerType(), True)
        ])
    
    def read_from_kafka(self, kafka_servers: str, topic: str) -> DataFrame:
        """Read streaming data from Kafka"""
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_servers) \
            .option("subscribe", topic) \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .load()
        
        # Parse JSON value
        parsed_df = df.select(
            col("key").cast("string").alias("message_key"),
            col("timestamp").alias("kafka_timestamp"),
            from_json(col("value").cast("string"), self.sensor_schema).alias("data")
        ).select("message_key", "kafka_timestamp", "data.*")
        
        return parsed_df
    
    def read_from_hdfs(self, hdfs_path: str, format: str = "parquet") -> DataFrame:
        """Read batch data from HDFS"""
        logger.info(f"Reading {format} data from {hdfs_path}")
        
        if format.lower() == "parquet":
            df = self.spark.read.parquet(hdfs_path)
        elif format.lower() == "json":
            df = self.spark.read.option("multiline", "true").json(hdfs_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return df
    
    def clean_data(self, df: DataFrame) -> DataFrame:
        """Clean and validate sensor data"""
        logger.info("Starting data cleaning...")
        
        # Remove duplicates
        df_dedup = df.dropDuplicates(["vehicle_id", "timestamp"])
        
        # Filter out null vehicle_ids and timestamps
        df_clean = df_dedup.filter(
            col("vehicle_id").isNotNull() & 
            col("timestamp").isNotNull()
        )
        
        # Remove outliers in speed (> 50 m/s is unrealistic for test vehicles)
        df_clean = df_clean.filter(col("calculated_speed") <= 50.0)
        
        # Validate coordinates
        df_clean = df_clean.filter(
            (col("location.latitude").between(-90, 90)) &
            (col("location.longitude").between(-180, 180))
        )
        
        # Clean weather data
        df_clean = df_clean.withColumn(
            "weather_conditions",
            when(col("weather_conditions").isNull(), 
                 struct(
                     col("temperature").cast(DoubleType()).alias("temperature"),
                     col("humidity").cast(DoubleType()).alias("humidity"),
                     col("visibility").cast(DoubleType()).alias("visibility"),
                     col("precipitation").alias("precipitation")
                 )
            ).otherwise(col("weather_conditions"))
        )
        
        logger.info("Data cleaning completed")
        return df_clean
    
    def feature_engineering(self, df: DataFrame) -> DataFrame:
        """Create additional features for ML training"""
        logger.info("Starting feature engineering...")
        
        # Time-based features
        df_features = df.withColumn(
            "datetime", 
            (col("timestamp").cast("timestamp"))
        ).withColumn(
            "hour_of_day", 
            col("datetime").substr(12, 2).cast(IntegerType())
        ).withColumn(
            "day_of_week", 
            date_format(col("datetime"), "EEEE")
        )
        
        # Speed categories
        df_features = df_features.withColumn(
            "speed_category",
            when(col("calculated_speed") < 5, "stationary")
            .when(col("calculated_speed") < 15, "slow")
            .when(col("calculated_speed") < 25, "medium")
            .otherwise("fast")
        )
        
        # Risk level encoding
        df_features = df_features.withColumn(
            "risk_level_numeric",
            when(col("risk_assessment.risk_level") == "low", 1)
            .when(col("risk_assessment.risk_level") == "medium", 2)
            .when(col("risk_assessment.risk_level") == "high", 3)
            .otherwise(0)
        )
        
        # LiDAR density features
        df_features = df_features.withColumn(
            "lidar_density",
            col("lidar_stats.point_count") / 
            (col("lidar_stats.max_distance") * col("lidar_stats.max_distance"))
        )
        
        # Weather severity score
        df_features = df_features.withColumn(
            "weather_severity",
            when(col("weather_conditions.precipitation") == "heavy_rain", 0.8)
            .when(col("weather_conditions.precipitation") == "snow", 0.9)
            .when(col("weather_conditions.precipitation") == "light_rain", 0.4)
            .when(col("weather_conditions.precipitation") == "fog", 0.6)
            .otherwise(0.1)
        )
        
        # Movement consistency (difference from average speed)
        window_spec = Window.partitionBy("vehicle_id").orderBy("timestamp").rowsBetween(-5, 0)
        df_features = df_features.withColumn(
            "avg_speed_window",
            avg("calculated_speed").over(window_spec)
        ).withColumn(
            "speed_variance",
            abs(col("calculated_speed") - col("avg_speed_window"))
        )
        
        logger.info("Feature engineering completed")
        return df_features
    
    def aggregate_by_vehicle(self, df: DataFrame) -> DataFrame:
        """Create vehicle-level aggregations"""
        logger.info("Creating vehicle aggregations...")
        
        vehicle_agg = df.groupBy("vehicle_id").agg(
            count("*").alias("total_records"),
            avg("calculated_speed").alias("avg_speed"),
            spark_max("calculated_speed").alias("max_speed"),
            spark_min("calculated_speed").alias("min_speed"),
            stddev("calculated_speed").alias("speed_stddev"),
            avg("lidar_stats.point_count").alias("avg_lidar_points"),
            avg("risk_assessment.risk_score").alias("avg_risk_score"),
            spark_sum("anomaly_count").alias("total_anomalies"),
            collect_list("location").alias("trajectory_points")
        )
        
        return vehicle_agg
    
    def create_ml_features(self, df: DataFrame) -> DataFrame:
        """Prepare features for ML training"""
        logger.info("Creating ML feature vectors...")
        
        # Select numerical features
        feature_cols = [
            "calculated_speed", "lidar_stats.point_count", "lidar_stats.avg_intensity",
            "lidar_stats.max_distance", "risk_level_numeric", "lidar_density",
            "weather_severity", "speed_variance", "hour_of_day"
        ]
        
        # Fill null values
        df_ml = df.fillna(0, subset=feature_cols)
        
        # Create feature vector
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features"
        )
        
        df_assembled = assembler.transform(df_ml)
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        scaler_model = scaler.fit(df_assembled)
        df_scaled = scaler_model.transform(df_assembled)
        
        return df_scaled
    
    def cluster_analysis(self, df: DataFrame) -> DataFrame:
        """Perform clustering analysis on sensor data"""
        logger.info("Performing clustering analysis...")
        
        # Prepare ML features
        df_ml = self.create_ml_features(df)
        
        # K-means clustering
        kmeans = KMeans(
            featuresCol="scaled_features",
            predictionCol="cluster",
            k=5,
            seed=42
        )
        
        model = kmeans.fit(df_ml)
        df_clustered = model.transform(df_ml)
        
        # Evaluate clustering
        evaluator = ClusteringEvaluator(
            featuresCol="scaled_features",
            predictionCol="cluster"
        )
        
        silhouette = evaluator.evaluate(df_clustered)
        logger.info(f"Clustering silhouette score: {silhouette}")
        
        return df_clustered
    
    def write_to_hdfs(self, df: DataFrame, output_path: str, mode: str = "append"):
        """Write processed data to HDFS"""
        logger.info(f"Writing data to HDFS: {output_path}")
        
        df.write \
            .mode(mode) \
            .option("compression", "snappy") \
            .parquet(output_path)
        
        logger.info(f"Data written successfully to {output_path}")
    
    def write_streaming_to_hdfs(self, df: DataFrame, output_path: str, 
                               checkpoint_path: str, trigger_interval: str = "60 seconds"):
        """Write streaming data to HDFS"""
        logger.info(f"Starting streaming write to HDFS: {output_path}")
        
        query = df.writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", output_path) \
            .option("checkpointLocation", checkpoint_path) \
            .trigger(processingTime=trigger_interval) \
            .start()
        
        return query
    
    def generate_data_quality_report(self, df: DataFrame) -> Dict[str, Any]:
        """Generate data quality report"""
        logger.info("Generating data quality report...")
        
        total_records = df.count()
        
        # Null counts
        null_counts = {}
        for col_name in df.columns:
            null_count = df.filter(col(col_name).isNull()).count()
            null_counts[col_name] = {
                'null_count': null_count,
                'null_percentage': round((null_count / total_records) * 100, 2)
            }
        
        # Speed statistics
        speed_stats = df.select(
            avg("calculated_speed").alias("avg_speed"),
            spark_max("calculated_speed").alias("max_speed"),
            spark_min("calculated_speed").alias("min_speed"),
            stddev("calculated_speed").alias("speed_stddev")
        ).collect()[0].asDict()
        
        # Vehicle count
        unique_vehicles = df.select("vehicle_id").distinct().count()
        
        # Anomaly statistics
        anomaly_stats = df.agg(
            spark_sum("anomaly_count").alias("total_anomalies"),
            avg("anomaly_count").alias("avg_anomalies_per_record")
        ).collect()[0].asDict()
        
        report = {
            'total_records': total_records,
            'unique_vehicles': unique_vehicles,
            'null_counts': null_counts,
            'speed_statistics': speed_stats,
            'anomaly_statistics': anomaly_stats,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def close(self):
        """Close Spark session"""
        self.spark.stop()
        logger.info("Spark session closed")

def main():
    """Main function for batch processing"""
    # Configuration
    HDFS_INPUT_PATH = os.getenv('HDFS_INPUT_PATH', 'hdfs://namenode:9000/data/raw/sensor_data')
    HDFS_OUTPUT_PATH = os.getenv('HDFS_OUTPUT_PATH', 'hdfs://namenode:9000/data/processed/sensor_data')
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'processed_sensor_data')
    
    processor = SensorDataProcessor("Sensor Data Batch Processing")
    
    try:
        # Read data from HDFS
        df_raw = processor.read_from_hdfs(HDFS_INPUT_PATH, format="parquet")
        
        # Process data
        df_clean = processor.clean_data(df_raw)
        df_features = processor.feature_engineering(df_clean)
        df_clustered = processor.cluster_analysis(df_features)
        
        # Create aggregations
        df_vehicle_agg = processor.aggregate_by_vehicle(df_features)
        
        # Generate quality report
        quality_report = processor.generate_data_quality_report(df_clean)
        logger.info(f"Data quality report: {quality_report}")
        
        # Write processed data
        processor.write_to_hdfs(df_clustered, HDFS_OUTPUT_PATH + "/enriched")
        processor.write_to_hdfs(df_vehicle_agg, HDFS_OUTPUT_PATH + "/vehicle_aggregations")
        
        logger.info("Batch processing completed successfully")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise
    finally:
        processor.close()

def streaming_main():
    """Main function for streaming processing"""
    KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
    INPUT_TOPIC = os.getenv('INPUT_TOPIC', 'processed_sensor_data')
    HDFS_OUTPUT_PATH = os.getenv('HDFS_OUTPUT_PATH', 'hdfs://namenode:9000/data/streaming/sensor_data')
    CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH', 'hdfs://namenode:9000/checkpoints/sensor_processing')
    
    processor = SensorDataProcessor("Sensor Data Streaming Processing")
    
    try:
        # Read streaming data from Kafka
        df_stream = processor.read_from_kafka(KAFKA_SERVERS, INPUT_TOPIC)
        
        # Process streaming data
        df_clean = processor.clean_data(df_stream)
        df_features = processor.feature_engineering(df_clean)
        
        # Write to HDFS
        query = processor.write_streaming_to_hdfs(
            df_features, 
            HDFS_OUTPUT_PATH, 
            CHECKPOINT_PATH
        )
        
        logger.info("Streaming processing started")
        query.awaitTermination()
        
    except Exception as e:
        logger.error(f"Streaming processing failed: {e}")
        raise
    finally:
        processor.close()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "streaming":
        streaming_main()
    else:
        main()
