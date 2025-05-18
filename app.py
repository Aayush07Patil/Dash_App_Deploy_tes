import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from dash.dependencies import Input, Output 
from collections import defaultdict
import pandas as pd
import numpy as np
from flask import request, jsonify
from plotly.subplots import make_subplots
from urllib.parse import parse_qs
import os
import logging
import json
import datetime
import threading
import time
import requests

# Import local modules
import utils
import main
import visualization

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dash_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # This line is crucial for Azure deployment

# Initialize global variables to store data from .NET
global_containers_df = None  # Will store the containers DataFrame 
global_products_df = None    # Will store the products DataFrame

# Initialize global variables for visualization results
global_placed_products = []
global_containers = []
global_blocked_for_ULD = []
global_placed_ulds = []
global_processed = False    # Flag to track if data has been processed

# Flask route to receive data from ASP.NET
@app.server.route('/update_data', methods=['POST'])
def update_data():
    global global_containers_df, global_products_df, global_processed
    global global_placed_products, global_containers, global_blocked_for_ULD, global_placed_ulds
    
    try:
        # Log the request
        logger.info(f"Received data from .NET frontend at {datetime.datetime.now()}")
        
        # Get data from POST request
        data = request.get_json()
        
        # Log the data size
        logger.info(f"Received {len(data['containers'])} containers and {len(data['products'])} products")
        
        # Save the full data to file for inspection
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'received_data')
        os.makedirs(data_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = os.path.join(data_dir, f'data_{timestamp}.json')
        
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved received data to {data_file}")
        
        # Log sample of first container and product for quick inspection
        if data['containers']:
            logger.info(f"First container sample: {json.dumps(data['containers'][0], indent=2)}")
        
        if data['products']:
            logger.info(f"First product sample: {json.dumps(data['products'][0], indent=2)}")
        
        # Convert to dataframes
        global_containers_df = pd.DataFrame(data['containers'])
        global_products_df = pd.DataFrame(data['products'])
        
        # Log dataframe columns
        logger.info(f"Container DataFrame columns: {global_containers_df.columns.tolist()}")
        logger.info(f"Product DataFrame columns: {global_products_df.columns.tolist()}")
        
        # Reset the processed flag so we'll reprocess the data
        global_processed = False
        
        # Process the data right away
        logger.info("Starting data processing...")
        ensure_data_processed()
        logger.info(f"Data processing complete. Placed products: {len(global_placed_products)}")
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)})

@app.server.route('/api/healthcheck', methods=['GET'])
def health_check():
    # Always return healthy response, even during processing
    return jsonify({'status': 'healthy'}), 200

# NEW COMBINED ENDPOINT: Get both container table and summary in one call
@app.server.route('/get_container_data', methods=['GET'])
def get_container_data():
    global global_placed_products, global_processed
    
    if not global_processed or not global_placed_products:
        return jsonify({'status': 'error', 'message': 'No data available yet'})
    
    try:
        # Create the container summary
        container_summary = utils.create_container_product_summary(global_placed_products)
        
        # Create the table data
        table_df = utils.create_container_summary_table(container_summary)
        
        # Combine both in a single response
        response_data = {
            'status': 'success',
            'container_summary': container_summary,
            'table_data': table_df.to_dict(orient='records')
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error creating combined container data: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)})

# Function to process data if it hasn't been processed yet
def ensure_data_processed():
    global global_containers_df, global_products_df, global_processed
    global global_placed_products, global_containers, global_blocked_for_ULD, global_placed_ulds
    
    # If we have data but haven't processed it yet
    if not global_processed and global_containers_df is not None and global_products_df is not None:
        try:
            logger.info("Processing container and product data...")
            # Run your existing processing function
            global_placed_products, global_containers, global_blocked_for_ULD, global_placed_ulds = main.main(
                global_containers_df, global_products_df
            )
            global_processed = True
            logger.info(f"Processing complete. Results: {len(global_placed_products)} placed products, "
                     f"{len(global_containers)} containers, {len(global_blocked_for_ULD)} blocked containers")
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}", exc_info=True)
            # Initialize with empty results if processing fails
            global_placed_products, global_containers = [], []
            global_blocked_for_ULD, global_placed_ulds = [], []
            global_processed = True  # Mark as processed to avoid repeated errors

# Add a debug endpoint to check the current state
@app.server.route('/debug', methods=['GET'])
def debug_info():
    global global_containers_df, global_products_df, global_processed
    global global_placed_products, global_containers, global_blocked_for_ULD, global_placed_ulds
    
    info = {
        'timestamp': datetime.datetime.now().isoformat(),
        'has_container_data': global_containers_df is not None,
        'has_product_data': global_products_df is not None,
        'data_processed': global_processed,
        'placed_products_count': len(global_placed_products),
        'containers_count': len(global_containers) if global_containers else 0,
        'blocked_containers_count': len(global_blocked_for_ULD) if global_blocked_for_ULD else 0,
        'placed_ulds_count': len(global_placed_ulds) if global_placed_ulds else 0
    }
    
    # Add container columns if available
    if global_containers_df is not None:
        info['container_columns'] = global_containers_df.columns.tolist()
    
    # Add product columns if available
    if global_products_df is not None:
        info['product_columns'] = global_products_df.columns.tolist()

    # Add simplified container list with essential info
    if global_containers:
        container_list = []
        for container in global_containers:
            # Create a simplified container summary
            container_summary = {
                'id': container['id'],
                'ULDCategory': container['ULDCategory'],
                'Type': container['Type'],
                'Dimensions': f"{container['Length']}x{container['Width']}x{container['Height']}",
                'Volume': round(container['Volume'], 2)
            }
            container_list.append(container_summary)
        
        # Sort by container ID for easier reading
        container_list = sorted(container_list, key=lambda x: x['id'])
        
        # Add to main info
        info['container_list'] = container_list
    
    # Add simplified list of blocked containers (if any)
    if global_blocked_for_ULD:
        blocked_list = []
        for container in global_blocked_for_ULD:
            blocked_summary = {
                'id': container['id'],
                'ULDCategory': container['ULDCategory'],
                'Type': container['Type']
            }
            blocked_list.append(blocked_summary)
        
        info['blocked_container_list'] = blocked_list
    
    return jsonify(info)

@app.server.route('/view_data', methods=['GET'])
def view_data():
    global global_containers_df, global_products_df
    
    if global_containers_df is None or global_products_df is None:
        return jsonify({"error": "No data available yet"})
    
    # Convert dataframes to JSON
    containers_json = global_containers_df.to_dict(orient='records')
    products_json = global_products_df.to_dict(orient='records')
    
    # Return the full data
    return jsonify({
        "containers": containers_json,
        "products": products_json
    })

# Define the app layout
app.layout = html.Div([    
    # Only the graph for displaying the 3D container visualization
    dcc.Graph(id='container-graph', style={'height': '100vh'}),
    
    # Store the URL params in the dcc.Location component
    dcc.Location(id='url', refresh=False),
    
    # Interval component with max_intervals to prevent continuous refreshing
    dcc.Interval(
        id='interval-component',
        interval=2*1000,  # in milliseconds
        n_intervals=0,
        max_intervals=0  # Stop refreshing after 3 intervals
    )
])

# Callback to update the visualization based on URL parameters
@app.callback(
    Output('container-graph', 'figure'),
    [Input('url', 'search'),
     Input('interval-component', 'n_intervals')]
)
def update_container_visualization(search, n_intervals):
    # Log callback execution
    logger.info(f"Callback triggered. Search: {search}, Intervals: {n_intervals}")
    
    # Check if data has been received
    if not global_processed:
        logger.info("No processed data available yet. Returning empty figure.")
        # No data yet, show empty figure
        return go.Figure()
    
    # Parse URL parameters to get container ID
    query_params = parse_qs(search.lstrip('?'))
    selected_container = None
    
    if 'container' in query_params:
        try:
            # No need to convert to int, use as string
            selected_container = query_params['container'][0]
            logger.info(f"Selected container from URL: {selected_container}")
        except (IndexError):
            logger.warning("Error parsing container ID from URL")
            pass
    
    logger.info(f"Creating visualization for container: {selected_container if selected_container else 'all'}")
    
    # Use the visualization function with the global data
    fig = visualization.visualize_specific_containers_with_plotly(
        global_containers, 
        global_placed_products, 
        global_blocked_for_ULD, 
        global_placed_ulds, 
        selected_container
    )
    
    logger.info("Visualization created successfully")
    return fig

def start_self_ping():
    """Start a background thread that pings the app to keep it responsive"""
    def ping_thread():
        app_url = os.environ.get('WEBSITE_HOSTNAME', '6e-devtest-databricks-appservice.azurewebsites.net')
        while True:
            try:
                # Ping the health check endpoint every 5 minutes
                requests.get(f"https://{app_url}/api/healthcheck", timeout=10)
                logger.info("Self-ping completed successfully")
            except Exception as e:
                logger.error(f"Self-ping failed: {str(e)}")
            
            # Sleep for 5 minutes
            time.sleep(120)
    
    # Start the thread
    thread = threading.Thread(target=ping_thread)
    thread.daemon = True
    thread.start()
    logger.info("Self-ping thread started")

# Run the Dash app - modified for Azure
if __name__ == '__main__':
    logger.info("Starting Dash application")
    start_self_ping()
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))
    #app.run(debug=False)