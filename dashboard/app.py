import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc

from kafka import KafkaConsumer
import threading
import queue
import time
from collections import deque, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataManager:
    """Manages real-time data from Kafka"""
    
    def __init__(self, kafka_servers: str, topics: List[str], max_records: int = 1000):
        self.kafka_servers = kafka_servers
        self.topics = topics
        self.max_records = max_records
        
        # Data storage
        self.data_queues = {topic: deque(maxlen=max_records) for topic in topics}
        self.latest_data = {topic: {} for topic in topics}
        self.metrics = defaultdict(lambda: defaultdict(list))
        
        # Threading
        self.stop_event = threading.Event()
        self.data_lock = threading.Lock()
        
        # Start data collection
        self.start_data_collection()
    
    def start_data_collection(self):
        """Start Kafka data collection in background thread"""
        def collect_data():
            try:
                consumer = KafkaConsumer(
                    *self.topics,
                    bootstrap_servers=self.kafka_servers,
                    group_id='dashboard-consumer',
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    auto_offset_reset='latest',
                    consumer_timeout_ms=1000
                )
                
                logger.info(f"Started collecting data from topics: {self.topics}")
                
                while not self.stop_event.is_set():
                    try:
                        message_batch = consumer.poll(timeout_ms=1000)
                        
                        with self.data_lock:
                            for topic_partition, messages in message_batch.items():
                                topic = topic_partition.topic
                                
                                for message in messages:
                                    data = message.value
                                    
                                    # Add timestamp if not present
                                    if 'dashboard_timestamp' not in data:
                                        data['dashboard_timestamp'] = time.time()
                                    
                                    # Store data
                                    self.data_queues[topic].append(data)
                                    self.latest_data[topic] = data
                                    
                                    # Update metrics
                                    self.update_metrics(topic, data)
                    
                    except Exception as e:
                        logger.error(f"Data collection error: {e}")
                        time.sleep(5)
                
                consumer.close()
                
            except Exception as e:
                logger.error(f"Kafka consumer error: {e}")
        
        thread = threading.Thread(target=collect_data, daemon=True)
        thread.start()
    
    def update_metrics(self, topic: str, data: Dict[str, Any]):
        """Update real-time metrics"""
        timestamp = data.get('dashboard_timestamp', time.time())
        
        if topic == 'inference_results':
            # Risk prediction metrics
            risk_data = data.get('risk_prediction', {})
            self.metrics[topic]['risk_scores'].append({
                'timestamp': timestamp,
                'vehicle_id': data.get('vehicle_id'),
                'risk_level': risk_data.get('predicted_risk_level'),
                'confidence': risk_data.get('confidence', 0)
            })
            
            # Anomaly detection metrics
            anomaly_data = data.get('anomaly_detection', {})
            self.metrics[topic]['anomalies'].append({
                'timestamp': timestamp,
                'vehicle_id': data.get('vehicle_id'),
                'is_anomaly': anomaly_data.get('is_anomaly', False),
                'anomaly_score': anomaly_data.get('anomaly_score', 0)
            })
            
        elif topic == 'processed_sensor_data':
            # Vehicle metrics
            self.metrics[topic]['vehicles'].append({
                'timestamp': timestamp,
                'vehicle_id': data.get('vehicle_id'),
                'speed': data.get('calculated_speed', 0),
                'location': data.get('location', {}),
                'lidar_points': data.get('lidar_stats', {}).get('point_count', 0)
            })
    
    def get_recent_data(self, topic: str, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get recent data for a topic"""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.data_lock:
            recent_data = [
                record for record in self.data_queues[topic]
                if record.get('dashboard_timestamp', 0) > cutoff_time
            ]
        
        return recent_data
    
    def get_vehicle_summary(self) -> Dict[str, Any]:
        """Get vehicle summary statistics"""
        recent_inference = self.get_recent_data('inference_results', 10)
        recent_sensor = self.get_recent_data('processed_sensor_data', 10)
        
        vehicle_stats = defaultdict(lambda: {
            'last_seen': 0,
            'risk_level': 'unknown',
            'speed': 0,
            'anomalies': 0,
            'location': {}
        })
        
        # Process inference results
        for record in recent_inference:
            vehicle_id = record.get('vehicle_id')
            timestamp = record.get('dashboard_timestamp', 0)
            
            if timestamp > vehicle_stats[vehicle_id]['last_seen']:
                vehicle_stats[vehicle_id]['last_seen'] = timestamp
                
                risk_data = record.get('risk_prediction', {})
                vehicle_stats[vehicle_id]['risk_level'] = risk_data.get('predicted_risk_level', 'unknown')
                
                anomaly_data = record.get('anomaly_detection', {})
                if anomaly_data.get('is_anomaly', False):
                    vehicle_stats[vehicle_id]['anomalies'] += 1
        
        # Process sensor data
        for record in recent_sensor:
            vehicle_id = record.get('vehicle_id')
            vehicle_stats[vehicle_id]['speed'] = record.get('calculated_speed', 0)
            vehicle_stats[vehicle_id]['location'] = record.get('location', {})
        
        return dict(vehicle_stats)
    
    def stop(self):
        """Stop data collection"""
        self.stop_event.set()

# Initialize data manager
KAFKA_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
TOPICS = ['inference_results', 'processed_sensor_data']
data_manager = RealTimeDataManager(KAFKA_SERVERS, TOPICS)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Autonomous Vehicle Monitoring Dashboard"

# Define layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🚗 Autonomous Vehicle Monitoring Dashboard", 
                   className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Summary cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="active-vehicles", children="0", className="text-primary"),
                    html.P("Active Vehicles", className="card-text")
                ])
            ])
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="high-risk-vehicles", children="0", className="text-danger"),
                    html.P("High Risk Vehicles", className="card-text")
                ])
            ])
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="anomaly-count", children="0", className="text-warning"),
                    html.P("Anomalies Detected", className="card-text")
                ])
            ])
        ], md=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="avg-speed", children="0", className="text-info"),
                    html.P("Average Speed (m/s)", className="card-text")
                ])
            ])
        ], md=3)
    ], className="mb-4"),
    
    # Real-time charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Real-time Risk Assessment"),
                dbc.CardBody([
                    dcc.Graph(id="risk-timeline")
                ])
            ])
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Vehicle Locations"),
                dbc.CardBody([
                    dcc.Graph(id="vehicle-map")
                ])
            ])
        ], md=6)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Speed Distribution"),
                dbc.CardBody([
                    dcc.Graph(id="speed-distribution")
                ])
            ])
        ], md=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Anomaly Detection"),
                dbc.CardBody([
                    dcc.Graph(id="anomaly-chart")
                ])
            ])
        ], md=6)
    ], className="mb-4"),
    
    # Vehicle details table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Vehicle Status Table"),
                dbc.CardBody([
                    html.Div(id="vehicle-table")
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=2000,  # Update every 2 seconds
        n_intervals=0
    )
    
], fluid=True)

# Callbacks
@app.callback(
    [Output('active-vehicles', 'children'),
     Output('high-risk-vehicles', 'children'),
     Output('anomaly-count', 'children'),
     Output('avg-speed', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_summary_cards(n):
    """Update summary statistics cards"""
    try:
        vehicle_summary = data_manager.get_vehicle_summary()
        
        # Calculate metrics
        active_vehicles = len(vehicle_summary)
        high_risk_vehicles = sum(1 for v in vehicle_summary.values() 
                               if v.get('risk_level') == 'high')
        total_anomalies = sum(v.get('anomalies', 0) for v in vehicle_summary.values())
        avg_speed = np.mean([v.get('speed', 0) for v in vehicle_summary.values()]) \
                   if vehicle_summary else 0
        
        return (
            str(active_vehicles),
            str(high_risk_vehicles),
            str(total_anomalies),
            f"{avg_speed:.1f}"
        )
        
    except Exception as e:
        logger.error(f"Summary cards update error: {e}")
        return "0", "0", "0", "0"

@app.callback(
    Output('risk-timeline', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_risk_timeline(n):
    """Update risk assessment timeline"""
    try:
        recent_data = data_manager.get_recent_data('inference_results', 10)
        
        if not recent_data:
            return go.Figure().add_annotation(
                text="No data available", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Prepare data
        timestamps = []
        risk_scores = []
        vehicle_ids = []
        risk_levels = []
        
        for record in recent_data:
            timestamp = datetime.fromtimestamp(record.get('dashboard_timestamp', 0))
            risk_data = record.get('risk_prediction', {})
            
            timestamps.append(timestamp)
            risk_scores.append(risk_data.get('confidence', 0))
            vehicle_ids.append(record.get('vehicle_id', 'unknown'))
            risk_levels.append(risk_data.get('predicted_risk_level', 'unknown'))
        
        # Create figure
        fig = go.Figure()
        
        # Color mapping for risk levels
        color_map = {'low': 'green', 'medium': 'orange', 'high': 'red', 'unknown': 'gray'}
        
        for risk_level in set(risk_levels):
            mask = [rl == risk_level for rl in risk_levels]
            fig.add_trace(go.Scatter(
                x=[t for t, m in zip(timestamps, mask) if m],
                y=[s for s, m in zip(risk_scores, mask) if m],
                mode='markers',
                name=f'{risk_level.capitalize()} Risk',
                marker=dict(color=color_map.get(risk_level, 'gray'), size=8),
                text=[v for v, m in zip(vehicle_ids, mask) if m],
                hovertemplate='<b>%{text}</b><br>Confidence: %{y:.2f}<br>Time: %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Risk Assessment Timeline",
            xaxis_title="Time",
            yaxis_title="Confidence Score",
            yaxis=dict(range=[0, 1]),
            height=400,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Risk timeline update error: {e}")
        return go.Figure()

@app.callback(
    Output('vehicle-map', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_vehicle_map(n):
    """Update vehicle location map"""
    try:
        vehicle_summary = data_manager.get_vehicle_summary()
        
        if not vehicle_summary:
            return go.Figure().add_annotation(
                text="No vehicle location data", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Prepare map data
        lats = []
        lons = []
        vehicle_ids = []
        risk_levels = []
        speeds = []
        
        for vehicle_id, data in vehicle_summary.items():
            location = data.get('location', {})
            if location.get('latitude') and location.get('longitude'):
                lats.append(location['latitude'])
                lons.append(location['longitude'])
                vehicle_ids.append(vehicle_id)
                risk_levels.append(data.get('risk_level', 'unknown'))
                speeds.append(data.get('speed', 0))
        
        if not lats:
            return go.Figure().add_annotation(
                text="No valid location data", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create map
        color_map = {'low': 'green', 'medium': 'orange', 'high': 'red', 'unknown': 'gray'}
        colors = [color_map.get(risk, 'gray') for risk in risk_levels]
        
        fig = go.Figure(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
            ),
            text=vehicle_ids,
            customdata=list(zip(risk_levels, speeds)),
            hovertemplate='<b>%{text}</b><br>Risk: %{customdata[0]}<br>Speed: %{customdata[1]:.1f} m/s<extra></extra>'
        ))
        
        # Center map on average location
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=12
            ),
            height=400,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Vehicle map update error: {e}")
        return go.Figure()

@app.callback(
    Output('speed-distribution', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_speed_distribution(n):
    """Update speed distribution chart"""
    try:
        recent_data = data_manager.get_recent_data('processed_sensor_data', 5)
        
        if not recent_data:
            return go.Figure().add_annotation(
                text="No speed data available", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        speeds = [record.get('calculated_speed', 0) for record in recent_data]
        
        fig = go.Figure(data=[go.Histogram(
            x=speeds,
            nbinsx=20,
            marker_color='lightblue',
            opacity=0.7
        )])
        
        fig.update_layout(
            title="Vehicle Speed Distribution",
            xaxis_title="Speed (m/s)",
            yaxis_title="Count",
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Speed distribution update error: {e}")
        return go.Figure()

@app.callback(
    Output('anomaly-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_anomaly_chart(n):
    """Update anomaly detection chart"""
    try:
        recent_data = data_manager.get_recent_data('inference_results', 10)
        
        if not recent_data:
            return go.Figure().add_annotation(
                text="No anomaly data available", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Prepare data
        timestamps = []
        anomaly_scores = []
        is_anomaly = []
        vehicle_ids = []
        
        for record in recent_data:
            timestamp = datetime.fromtimestamp(record.get('dashboard_timestamp', 0))
            anomaly_data = record.get('anomaly_detection', {})
            
            timestamps.append(timestamp)
            anomaly_scores.append(anomaly_data.get('anomaly_score', 0))
            is_anomaly.append(anomaly_data.get('is_anomaly', False))
            vehicle_ids.append(record.get('vehicle_id', 'unknown'))
        
        # Create figure
        fig = go.Figure()
        
        # Normal data points
        normal_mask = [not a for a in is_anomaly]
        fig.add_trace(go.Scatter(
            x=[t for t, m in zip(timestamps, normal_mask) if m],
            y=[s for s, m in zip(anomaly_scores, normal_mask) if m],
            mode='markers',
            name='Normal',
            marker=dict(color='green', size=6),
            text=[v for v, m in zip(vehicle_ids, normal_mask) if m],
            hovertemplate='<b>%{text}</b><br>Score: %{y:.3f}<br>Time: %{x}<extra></extra>'
        ))
        
        # Anomaly data points
        anomaly_mask = is_anomaly
        fig.add_trace(go.Scatter(
            x=[t for t, m in zip(timestamps, anomaly_mask) if m],
            y=[s for s, m in zip(anomaly_scores, anomaly_mask) if m],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x'),
            text=[v for v, m in zip(vehicle_ids, anomaly_mask) if m],
            hovertemplate='<b>%{text}</b><br>Score: %{y:.3f}<br>Time: %{x}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Anomaly Detection Over Time",
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            height=400,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Anomaly chart update error: {e}")
        return go.Figure()

@app.callback(
    Output('vehicle-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_vehicle_table(n):
    """Update vehicle status table"""
    try:
        vehicle_summary = data_manager.get_vehicle_summary()
        
        if not vehicle_summary:
            return html.P("No vehicle data available")
        
        # Prepare table data
        table_data = []
        for vehicle_id, data in vehicle_summary.items():
            last_seen = datetime.fromtimestamp(data.get('last_seen', 0))
            location = data.get('location', {})
            
            table_data.append({
                'Vehicle ID': vehicle_id,
                'Risk Level': data.get('risk_level', 'unknown').capitalize(),
                'Speed (m/s)': f"{data.get('speed', 0):.1f}",
                'Location': f"{location.get('latitude', 0):.4f}, {location.get('longitude', 0):.4f}",
                'Anomalies': data.get('anomalies', 0),
                'Last Seen': last_seen.strftime('%H:%M:%S')
            })
        
        # Create table
        df = pd.DataFrame(table_data)
        
        table = dbc.Table.from_dataframe(
            df, 
            striped=True, 
            bordered=True, 
            hover=True,
            responsive=True,
            size='sm'
        )
        
        return table
        
    except Exception as e:
        logger.error(f"Vehicle table update error: {e}")
        return html.P(f"Error loading table: {e}")

if __name__ == '__main__':
    app.run_server(
        debug=os.getenv('DEBUG', 'False').lower() == 'true',
        host='0.0.0.0',
        port=int(os.getenv('DASH_PORT', 8050))
    )
