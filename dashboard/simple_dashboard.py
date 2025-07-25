"""
Simple Dashboard for Autonomous Vehicle Simulation - No Kafka Version
Displays vehicle data, sensor readings, and risk analysis
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Autonomous Vehicle Simulation Dashboard"

# Sample data for demonstration
def generate_sample_data():
    """Generate sample vehicle data"""
    np.random.seed(42)
    vehicles = ['AV_001', 'AV_002', 'AV_003', 'AV_004', 'AV_005']
    timestamps = pd.date_range(start=datetime.now() - timedelta(hours=1), 
                              end=datetime.now(), freq='30S')
    
    data = []
    for timestamp in timestamps:
        for vehicle in vehicles:
            record = {
                'vehicle_id': vehicle,
                'timestamp': timestamp,
                'latitude': 37.7749 + np.random.normal(0, 0.01),
                'longitude': -122.4194 + np.random.normal(0, 0.01),
                'speed': max(0, np.random.normal(15, 5)),  # km/h
                'risk_level': np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.25, 0.05]),
                'battery_level': max(0, min(100, np.random.normal(75, 15))),
                'lidar_points': np.random.randint(50, 200),
                'gps_satellites': np.random.randint(8, 16),
                'temperature': np.random.normal(20, 5)
            }
            data.append(record)
    
    return pd.DataFrame(data)

# Load or generate data
try:
    if os.path.exists('data/input/test_data.parquet'):
        df = pd.read_parquet('data/input/test_data.parquet')
        # Convert to proper format
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['speed'] = df.get('calculated_speed', df.get('speed', 15)) * 3.6  # m/s to km/h
        df['latitude'] = 37.7749
        df['longitude'] = -122.4194
        df['battery_level'] = np.random.uniform(50, 100, len(df))
        df['lidar_points'] = np.random.randint(50, 200, len(df))
        df['gps_satellites'] = np.random.randint(8, 16, len(df))
        df['temperature'] = np.random.normal(20, 5, len(df))
    else:
        df = generate_sample_data()
except Exception as e:
    print(f"Error loading data, generating sample: {e}")
    df = generate_sample_data()

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚗 Autonomous Vehicle Simulation Dashboard", 
                   className="text-center mb-4",
                   style={'color': '#2c3e50'})
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Fleet Overview", className="card-title"),
                    html.P(f"Total Vehicles: {df['vehicle_id'].nunique()}", className="card-text"),
                    html.P(f"Data Points: {len(df):,}", className="card-text"),
                    html.P(f"Time Range: {df['timestamp'].min().strftime('%H:%M')} - {df['timestamp'].max().strftime('%H:%M')}", 
                           className="card-text")
                ])
            ], color="info", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("⚠️ Risk Levels", className="card-title"),
                    html.P(f"High Risk: {(df['risk_level'] == 'high').sum()}", 
                           className="card-text text-danger"),
                    html.P(f"Medium Risk: {(df['risk_level'] == 'medium').sum()}", 
                           className="card-text text-warning"),
                    html.P(f"Low Risk: {(df['risk_level'] == 'low').sum()}", 
                           className="card-text text-success")
                ])
            ], color="warning", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🚄 Speed Stats", className="card-title"),
                    html.P(f"Avg Speed: {df['speed'].mean():.1f} km/h", className="card-text"),
                    html.P(f"Max Speed: {df['speed'].max():.1f} km/h", className="card-text"),
                    html.P(f"Min Speed: {df['speed'].min():.1f} km/h", className="card-text")
                ])
            ], color="success", outline=True)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🔋 System Health", className="card-title"),
                    html.P(f"Avg Battery: {df['battery_level'].mean():.1f}%", className="card-text"),
                    html.P(f"LiDAR Points: {df['lidar_points'].mean():.0f}", className="card-text"),
                    html.P(f"GPS Satellites: {df['gps_satellites'].mean():.0f}", className="card-text")
                ])
            ], color="primary", outline=True)
        ], width=3)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="speed-timeline")
        ], width=6),
        dbc.Col([
            dcc.Graph(id="risk-distribution")
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dcc.Graph(id="vehicle-map")
        ], width=6),
        dbc.Col([
            dcc.Graph(id="sensor-health")
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.H4("📈 Real-time Metrics"),
            dcc.Interval(id='interval-component', interval=5000, n_intervals=0),
            html.Div(id='live-metrics')
        ])
    ])
    
], fluid=True)

# Callbacks
@app.callback(
    Output('speed-timeline', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_speed_timeline(n):
    fig = px.line(df, x='timestamp', y='speed', color='vehicle_id',
                  title='Vehicle Speed Over Time',
                  labels={'speed': 'Speed (km/h)', 'timestamp': 'Time'})
    fig.update_layout(height=400)
    return fig

@app.callback(
    Output('risk-distribution', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_risk_distribution(n):
    risk_counts = df['risk_level'].value_counts()
    fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                 title='Risk Level Distribution',
                 color_discrete_map={'low': '#28a745', 'medium': '#ffc107', 'high': '#dc3545'})
    fig.update_layout(height=400)
    return fig

@app.callback(
    Output('vehicle-map', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_vehicle_map(n):
    latest_positions = df.groupby('vehicle_id').last().reset_index()
    
    fig = px.scatter_mapbox(latest_positions, 
                           lat='latitude', lon='longitude',
                           color='risk_level',
                           color_discrete_map={'low': '#28a745', 'medium': '#ffc107', 'high': '#dc3545'},
                           hover_data=['vehicle_id', 'speed', 'battery_level'],
                           title='Current Vehicle Positions',
                           zoom=12)
    
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=37.7749, lon=-122.4194)),
        height=400
    )
    return fig

@app.callback(
    Output('sensor-health', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_sensor_health(n):
    latest_data = df.groupby('vehicle_id').last().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=latest_data['vehicle_id'],
        y=latest_data['battery_level'],
        mode='markers+lines',
        name='Battery Level (%)',
        line=dict(color='#007bff')
    ))
    
    # Add secondary y-axis for LiDAR points
    fig.add_trace(go.Scatter(
        x=latest_data['vehicle_id'],
        y=latest_data['lidar_points'],
        mode='markers+lines',
        name='LiDAR Points',
        yaxis='y2',
        line=dict(color='#28a745')
    ))
    
    fig.update_layout(
        title='Sensor Health Status',
        xaxis_title='Vehicle ID',
        yaxis=dict(title='Battery Level (%)', side='left'),
        yaxis2=dict(title='LiDAR Points', side='right', overlaying='y'),
        height=400
    )
    
    return fig

@app.callback(
    Output('live-metrics', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_live_metrics(n):
    current_time = datetime.now().strftime('%H:%M:%S')
    
    metrics = [
        dbc.Alert(f"🕒 Last Update: {current_time}", color="info"),
        dbc.Alert(f"✅ System Status: All systems operational", color="success"),
        dbc.Alert(f"📡 Data Processing: {len(df):,} records processed", color="primary")
    ]
    
    return metrics

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚗 Autonomous Vehicle Simulation Dashboard")
    print("="*50)
    print(f"📊 Loaded {len(df):,} data points for {df['vehicle_id'].nunique()} vehicles")
    print(f"⏰ Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("🌐 Starting dashboard on http://localhost:8050")
    print("="*50 + "\n")
    
    app.run_server(debug=True, host='localhost', port=8050)
