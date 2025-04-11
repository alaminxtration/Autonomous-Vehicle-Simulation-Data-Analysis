from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()
df = spark.read.format("csv").option("header", "true").load("hdfs://path_to_data.csv")
df = df.filter(df["lidar"] != "null")
df.show()