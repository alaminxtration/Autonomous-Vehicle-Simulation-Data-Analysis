import boto3

s3 = boto3.client('s3')
s3.upload_file("local_file_path", "bucket_name", "s3_file_path")