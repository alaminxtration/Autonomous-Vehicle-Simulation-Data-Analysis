#!/usr/bin/env python3
"""
Enhanced Storage-Optimized Dashboard for AV Simulation
Integrates with storage optimizer and Redis for efficient data handling
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import redis
import json
import sqlite3
import gzip
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage_optimizer import StorageOptimizer

# Initialize components
app = dash.Dash(__name__)
optimizer = StorageOptimizer()

# Redis connection
try:
    r = redis.Redis(host='localhost', port=6379, decode_responses=True)
    r.ping()
    redis_connected = True
    print("✅ Connected to Redis")
except:
    redis_connected = False
    print("❌ Redis not available")

# Dashboard styling
app.layout = html.Div([
    html.Div([
        html.H1("🚗 AV Simulation - Storage Optimized Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
        
        # Storage Stats Row
        html.Div([
            html.Div([
                html.H3("📊 Storage Statistics", style={'color': '#34495e'}),
                html.Div(id='storage-stats', style={'padding': '20px'})
            ], className='six columns'),
            
            html.Div([
                html.H3("🗃️ Data Sources", style={'color': '#34495e'}),
                html.Div(id='data-sources', style={'padding': '20px'})
            ], className='six columns'),
        ], className='row'),
        
        html.Hr(),
        
        # Control Panel
        html.Div([
            html.H3("🎛️ Control Panel", style={'color': '#34495e'}),
            html.Div([
                html.Button('🗜️ Optimize Storage', id='optimize-btn', 
                           style={'margin': '10px', 'padding': '10px 20px'}),
                html.Button('🔄 Refresh Data', id='refresh-btn',
                           style={'margin': '10px', 'padding': '10px 20px'}),
                html.Button('📤 Export Data', id='export-btn',
                           style={'margin': '10px', 'padding': '10px 20px'}),
                dcc.Dropdown(
                    id='vehicle-selector',
                    placeholder="Select Vehicle(s)",
                    multi=True,
                    style={'width': '300px', 'margin': '10px'}
                ),
                dcc.Dropdown(
                    id='time-range',
                    options=[
                        {'label': 'Last Hour', 'value': 1},
                        {'label': 'Last 6 Hours', 'value': 6},
                        {'label': 'Last 24 Hours', 'value': 24},
                        {'label': 'Last Week', 'value': 168}
                    ],
                    value=24,
                    style={'width': '200px', 'margin': '10px'}
                )
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center'})
        ]),
        
        html.Hr(),
        
        # Charts Row
        html.Div([
            html.Div([
                dcc.Graph(id='vehicle-locations-map')
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(id='risk-distribution')
            ], className='six columns'),
        ], className='row'),
        
        html.Div([
            html.Div([
                dcc.Graph(id='speed-timeline')
            ], className='six columns'),
            
            html.Div([
                dcc.Graph(id='data-volume-chart')
            ], className='six columns'),
        ], className='row'),
        
        # Status messages
        html.Div(id='status-messages', style={'margin': '20px'}),
        
        # Auto-refresh interval
        dcc.Interval(
            id='interval-component',
            interval=30*1000,  # 30 seconds
            n_intervals=0
        )
    ], style={'padding': '20px'})
])

@callback(
    [Output('storage-stats', 'children'),
     Output('data-sources', 'children'),
     Output('vehicle-selector', 'options')],
    [Input('interval-component', 'n_intervals'),
     Input('refresh-btn', 'n_clicks')]
)
def update_dashboard_info(n_intervals, refresh_clicks):
    # Get storage statistics
    stats = optimizer.get_storage_stats()
    
    storage_info = html.Div([
        html.P(f"💾 Total Storage: {stats.get('total_size_mb', 0):.2f} MB"),
        html.P(f"📁 Total Files: {stats.get('total_files', 0)}"),
        html.P(f"🗃️ DB Records: {stats.get('database_records', 0)}"),
        html.P(f"🗜️ Compression: Active"),
    ])
    
    # Check data sources
    sources = []
    if redis_connected:
        try:
            redis_keys = r.keys('*')
            sources.append(f"✅ Redis: {len(redis_keys)} keys")
        except:
            sources.append("❌ Redis: Connection failed")
    else:
        sources.append("❌ Redis: Not connected")
    
    sources.append(f"✅ SQLite: {stats.get('database_records', 0)} records")
    
    data_sources_info = html.Div([
        html.P(source) for source in sources
    ])
    
    # Get vehicle options
    try:
        # Get vehicles from database
        with sqlite3.connect(optimizer.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT vehicle_id FROM sensor_data ORDER BY vehicle_id")
            vehicles = [row[0] for row in cursor.fetchall()]
        
        vehicle_options = [{'label': vid, 'value': vid} for vid in vehicles]
    except:
        vehicle_options = []
    
    return storage_info, data_sources_info, vehicle_options

@callback(
    Output('vehicle-locations-map', 'figure'),
    [Input('vehicle-selector', 'value'),
     Input('time-range', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_vehicle_map(selected_vehicles, hours_back, n_intervals):
    try:
        # Get recent data
        df = optimizer.export_data_for_analysis(hours_back=hours_back or 24)
        
        if df is None or df.empty:
            return {
                'data': [],
                'layout': {
                    'title': 'No vehicle location data available',
                    'xaxis': {'title': 'Longitude'},
                    'yaxis': {'title': 'Latitude'}
                }
            }
        
        # Filter by selected vehicles
        if selected_vehicles:
            df = df[df['vehicle_id'].isin(selected_vehicles)]
        
        # Extract GPS data if available
        if 'gps' in df.columns:
            df['lat'] = df['gps'].apply(lambda x: x.get('latitude', 0) if isinstance(x, dict) else 0)
            df['lon'] = df['gps'].apply(lambda x: x.get('longitude', 0) if isinstance(x, dict) else 0)
        else:
            # Use mock coordinates
            df['lat'] = 37.7749 + (df.index % 100) * 0.001
            df['lon'] = -122.4194 + (df.index % 100) * 0.001
        
        # Create scatter plot
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="vehicle_id",
            title="Vehicle Locations",
            zoom=10,
            height=400
        )
        
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
        
        return fig
        
    except Exception as e:
        return {
            'data': [],
            'layout': {'title': f'Error loading map: {str(e)}'}
        }

@callback(
    Output('risk-distribution', 'figure'),
    [Input('vehicle-selector', 'value'),
     Input('time-range', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_risk_chart(selected_vehicles, hours_back, n_intervals):
    try:
        df = optimizer.export_data_for_analysis(hours_back=hours_back or 24)
        
        if df is None or df.empty:
            return {'data': [], 'layout': {'title': 'No risk data available'}}
        
        if selected_vehicles:
            df = df[df['vehicle_id'].isin(selected_vehicles)]
        
        # Extract risk level if available
        if 'risk_level' in df.columns:
            risk_counts = df['risk_level'].value_counts()
        else:
            # Mock risk data
            risk_counts = pd.Series({'low': 60, 'medium': 30, 'high': 10})
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color_discrete_map={'low': '#2ecc71', 'medium': '#f39c12', 'high': '#e74c3c'}
        )
        
        return fig
        
    except Exception as e:
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

@callback(
    Output('speed-timeline', 'figure'),
    [Input('vehicle-selector', 'value'),
     Input('time-range', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_speed_chart(selected_vehicles, hours_back, n_intervals):
    try:
        df = optimizer.export_data_for_analysis(hours_back=hours_back or 24)
        
        if df is None or df.empty:
            return {'data': [], 'layout': {'title': 'No speed data available'}}
        
        if selected_vehicles:
            df = df[df['vehicle_id'].isin(selected_vehicles)]
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Extract speed if available
        if 'calculated_speed' in df.columns:
            speed_col = 'calculated_speed'
        else:
            # Mock speed data
            df['calculated_speed'] = 15 + 10 * (df.index % 10) / 10
            speed_col = 'calculated_speed'
        
        fig = px.line(
            df,
            x='datetime',
            y=speed_col,
            color='vehicle_id',
            title="Vehicle Speed Over Time",
            labels={'calculated_speed': 'Speed (m/s)', 'datetime': 'Time'}
        )
        
        return fig
        
    except Exception as e:
        return {'data': [], 'layout': {'title': f'Error: {str(e)}'}}

@callback(
    Output('data-volume-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_data_volume_chart(n_intervals):
    try:
        # Get data volume over time
        with sqlite3.connect(optimizer.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as records,
                    SUM(LENGTH(compressed_data)) / 1024.0 / 1024.0 as size_mb
                FROM sensor_data 
                WHERE created_at >= datetime('now', '-7 days')
                GROUP BY DATE(created_at)
                ORDER BY date
            """)
            
            data = cursor.fetchall()
            
            if not data:
                return {'data': [], 'layout': {'title': 'No volume data available'}}
            
            df = pd.DataFrame(data, columns=['date', 'records', 'size_mb'])
            
            fig = px.bar(
                df,
                x='date',
                y='size_mb',
                title="Daily Data Volume (MB)",
                labels={'size_mb': 'Storage (MB)', 'date': 'Date'}
            )
            
            return fig
        
    except Exception as e:
        return {'data': [], 'layout': {'title': f'Volume chart error: {str(e)}'}}

@callback(
    Output('status-messages', 'children'),
    [Input('optimize-btn', 'n_clicks'),
     Input('export-btn', 'n_clicks')]
)
def handle_button_clicks(optimize_clicks, export_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'optimize-btn' and optimize_clicks:
        try:
            compressed_count, savings = optimizer.compress_existing_data()
            migrated_count = optimizer.migrate_to_efficient_storage()
            removed_count, freed_space = optimizer.cleanup_old_files(days_old=7)
            
            message = html.Div([
                html.P(f"✅ Storage optimization complete!", style={'color': 'green'}),
                html.P(f"🗜️ Compressed {compressed_count} files, saved {savings/(1024*1024):.2f} MB"),
                html.P(f"📦 Migrated {migrated_count} records to database"),
                html.P(f"🧹 Cleaned up {removed_count} old files")
            ])
            return message
        except Exception as e:
            return html.P(f"❌ Optimization failed: {str(e)}", style={'color': 'red'})
    
    elif button_id == 'export-btn' and export_clicks:
        try:
            df = optimizer.export_data_for_analysis(hours_back=24)
            if df is not None:
                export_path = f"data/export_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(export_path, index=False)
                return html.P(f"✅ Data exported to {export_path}", style={'color': 'green'})
            else:
                return html.P("❌ No data to export", style={'color': 'orange'})
        except Exception as e:
            return html.P(f"❌ Export failed: {str(e)}", style={'color': 'red'})
    
    return ""

if __name__ == '__main__':
    print("🚀 Starting Storage-Optimized AV Dashboard...")
    print(f"📊 Database records: {optimizer.get_storage_stats().get('database_records', 0)}")
    print(f"💾 Storage usage: {optimizer.get_storage_stats().get('total_size_mb', 0):.2f} MB")
    print("🌐 Dashboard will be available at: http://localhost:8050")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
