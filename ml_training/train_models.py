import os
import sys
import logging
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import detection
import torchvision.models as models

import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorDataset(Dataset):
    """Custom dataset for sensor data"""
    
    def __init__(self, data: pd.DataFrame, features: List[str], target: str, transform=None):
        self.data = data
        self.features = features
        self.target = target
        self.transform = transform
        
        # Prepare feature matrix
        self.X = self.data[features].values.astype(np.float32)
        self.y = self.data[target].values.astype(np.float32)
        
        # Scale features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.y[idx], dtype=torch.float32)
        
        if self.transform:
            features = self.transform(features)
        
        return features, target

class RiskPredictionModel(nn.Module):
    """Neural network for predicting vehicle risk levels"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, dropout_rate: float = 0.3):
        super(RiskPredictionModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AnomalyDetectionModel(nn.Module):
    """Autoencoder for anomaly detection"""
    
    def __init__(self, input_size: int, encoding_dim: int):
        super(AnomalyDetectionModel, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim * 4),
            nn.ReLU(),
            nn.Linear(encoding_dim * 4, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim * 4),
            nn.ReLU(),
            nn.Linear(encoding_dim * 4, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SpeedPredictionModel(nn.Module):
    """LSTM model for speed prediction"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(SpeedPredictionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last output
        return out

class MLTrainer:
    """ML model trainer with MLflow integration"""
    
    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.client = MlflowClient(mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and prepare data for training"""
        logger.info(f"Loading data from {data_path}")
        
        # Load data (assuming parquet format from Spark processing)
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Basic data statistics
        data_stats = {
            'total_records': len(df),
            'unique_vehicles': df['vehicle_id'].nunique() if 'vehicle_id' in df.columns else 0,
            'features_count': len(df.columns),
            'missing_values': df.isnull().sum().to_dict()
        }
        
        logger.info(f"Data loaded: {data_stats}")
        return df, data_stats
    
    def create_risk_prediction_features(self, df: pd.DataFrame) -> Tuple[List[str], str]:
        """Create features for risk prediction"""
        feature_cols = [
            'calculated_speed', 'lidar_stats_point_count', 'lidar_stats_avg_intensity',
            'lidar_stats_max_distance', 'lidar_density', 'weather_severity',
            'speed_variance', 'hour_of_day', 'anomaly_count'
        ]
        
        # Flatten nested columns if needed
        for col in df.columns:
            if 'lidar_stats.' in col:
                new_col = col.replace('.', '_')
                df[new_col] = df[col]
        
        # Create risk level target
        if 'risk_assessment_risk_level' in df.columns:
            target_col = 'risk_assessment_risk_level'
        else:
            # Create synthetic target based on speed and anomalies
            df['risk_level'] = 0
            df.loc[df['calculated_speed'] > 20, 'risk_level'] = 1
            df.loc[df['anomaly_count'] > 0, 'risk_level'] = 1
            df.loc[(df['calculated_speed'] > 30) | (df['anomaly_count'] > 2), 'risk_level'] = 2
            target_col = 'risk_level'
        
        # Select available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        return available_features, target_col
    
    def train_risk_prediction_model(self, df: pd.DataFrame, experiment_name: str = "risk_prediction"):
        """Train risk prediction model"""
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"risk_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Prepare features
            features, target = self.create_risk_prediction_features(df)
            
            # Parameters
            params = {
                'hidden_sizes': [128, 64, 32],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 100,
                'patience': 10
            }
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("features", features)
            mlflow.log_param("n_features", len(features))
            
            # Prepare data
            df_clean = df[features + [target]].dropna()
            
            # Encode target if categorical
            if df_clean[target].dtype == 'object':
                le = LabelEncoder()
                df_clean[target] = le.fit_transform(df_clean[target])
                num_classes = len(le.classes_)
                mlflow.log_param("label_encoder_classes", le.classes_.tolist())
            else:
                num_classes = int(df_clean[target].max()) + 1
            
            # Create dataset
            dataset = SensorDataset(df_clean, features, target)
            
            # Train/validation split
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            
            # Create model
            model = RiskPredictionModel(
                input_size=len(features),
                hidden_sizes=params['hidden_sizes'],
                num_classes=num_classes,
                dropout_rate=params['dropout_rate']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(params['epochs']):
                # Training
                model.train()
                train_loss = 0.0
                for features_batch, targets_batch in train_loader:
                    features_batch = features_batch.to(self.device)
                    targets_batch = targets_batch.long().to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(features_batch)
                    loss = criterion(outputs, targets_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for features_batch, targets_batch in val_loader:
                        features_batch = features_batch.to(self.device)
                        targets_batch = targets_batch.long().to(self.device)
                        
                        outputs = model(features_batch)
                        loss = criterion(outputs, targets_batch)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += targets_batch.size(0)
                        correct += (predicted == targets_batch).sum().item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                accuracy = correct / total
                
                # Log metrics
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", accuracy, step=epoch)
                
                scheduler.step(avg_val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{params['epochs']}: "
                              f"Train Loss: {avg_train_loss:.4f}, "
                              f"Val Loss: {avg_val_loss:.4f}, "
                              f"Val Accuracy: {accuracy:.4f}")
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), "best_risk_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= params['patience']:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Load best model
            model.load_state_dict(torch.load("best_risk_model.pth"))
            
            # Final evaluation
            model.eval()
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for features_batch, targets_batch in val_loader:
                    features_batch = features_batch.to(self.device)
                    outputs = model(features_batch)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_targets.extend(targets_batch.numpy())
            
            # Calculate final metrics
            final_accuracy = accuracy_score(all_targets, all_predictions)
            final_precision = precision_score(all_targets, all_predictions, average='weighted')
            final_recall = recall_score(all_targets, all_predictions, average='weighted')
            final_f1 = f1_score(all_targets, all_predictions, average='weighted')
            
            # Log final metrics
            mlflow.log_metric("final_accuracy", final_accuracy)
            mlflow.log_metric("final_precision", final_precision)
            mlflow.log_metric("final_recall", final_recall)
            mlflow.log_metric("final_f1", final_f1)
            
            # Log model
            mlflow.pytorch.log_model(model, "risk_prediction_model")
            
            # Log feature scaler
            with open("feature_scaler.pkl", "wb") as f:
                pickle.dump(dataset.scaler, f)
            mlflow.log_artifact("feature_scaler.pkl")
            
            logger.info(f"Risk prediction model training completed. "
                       f"Final accuracy: {final_accuracy:.4f}")
            
            return model, dataset.scaler
    
    def train_anomaly_detection_model(self, df: pd.DataFrame, experiment_name: str = "anomaly_detection"):
        """Train anomaly detection model"""
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Parameters
            params = {
                'encoding_dim': 32,
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 50
            }
            
            mlflow.log_params(params)
            
            # Prepare features (use numerical columns)
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numerical_cols if not col.startswith('target')][:20]  # Limit features
            
            df_clean = df[feature_cols].dropna()
            
            # Normalize data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df_clean)
            
            # Convert to tensor
            data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
            
            # Create data loader
            dataset = torch.utils.data.TensorDataset(data_tensor)
            train_loader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)
            
            # Create model
            model = AnomalyDetectionModel(
                input_size=len(feature_cols),
                encoding_dim=params['encoding_dim']
            ).to(self.device)
            
            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            
            # Training loop
            for epoch in range(params['epochs']):
                model.train()
                total_loss = 0.0
                
                for batch in train_loader:
                    data = batch[0].to(self.device)
                    
                    optimizer.zero_grad()
                    reconstructed = model(data)
                    loss = criterion(reconstructed, data)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(train_loader)
                mlflow.log_metric("train_loss", avg_loss, step=epoch)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{params['epochs']}: Loss: {avg_loss:.6f}")
            
            # Log model and scaler
            mlflow.pytorch.log_model(model, "anomaly_detection_model")
            
            with open("anomaly_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            mlflow.log_artifact("anomaly_scaler.pkl")
            
            logger.info("Anomaly detection model training completed")
            
            return model, scaler

def main():
    """Main training function"""
    # Configuration
    DATA_PATH = os.getenv('DATA_PATH', '/data/processed/sensor_data')
    MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    
    trainer = MLTrainer(MLFLOW_URI)
    
    try:
        # Load data
        df, data_stats = trainer.prepare_data(DATA_PATH)
        
        # Train risk prediction model
        logger.info("Training risk prediction model...")
        risk_model, risk_scaler = trainer.train_risk_prediction_model(df)
        
        # Train anomaly detection model
        logger.info("Training anomaly detection model...")
        anomaly_model, anomaly_scaler = trainer.train_anomaly_detection_model(df)
        
        logger.info("All model training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
