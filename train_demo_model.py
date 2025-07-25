import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create experiment
experiment_name = "autonomous_vehicle_demo"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

mlflow.set_experiment(experiment_name)

# Load training data
print("Loading training data...")
df = pd.read_parquet("data/input/training_data.parquet")

# Prepare features
feature_columns = [
    'calculated_speed', 'lidar_stats_point_count', 'lidar_stats_avg_intensity',
    'lidar_stats_max_distance', 'lidar_density', 'weather_severity',
    'speed_variance', 'hour_of_day', 'anomaly_count'
]

X = df[feature_columns].fillna(0)
y = df['risk_level_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "risk_prediction_model")
    
    print(f"Model trained with accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save feature names for inference
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(feature_columns, f)
    mlflow.log_artifact("feature_names.pkl")
    
    print("Model logged to MLflow successfully!")

print("Demo model training completed!")
