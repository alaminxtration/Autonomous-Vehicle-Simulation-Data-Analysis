FROM python:3.9-slim

WORKDIR /mlflow

RUN pip install --no-cache-dir \
    mlflow==2.8.1 \
    boto3==1.29.7 \
    psycopg2-binary==2.9.7 \
    pymysql==1.1.0

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
