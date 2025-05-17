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
import functions as fun
import threading
import time
import requests

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

# Import your main module which processes data
import main as mn

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

# Your existing visualization function
def visualize_specific_containers_with_plotly(containers, placed_products, blocked_for_ULD, placed_ulds, container_number=None):
    colors = ['#A97835', '#C08F4F', '#CD9F61', '#C2A574', '#D6B88A','#E6D3B3','#CBB994']

    if container_number is not None:
        # No need to convert container_number to int, compare as string
        containers = [container for container in containers if container['id'] == container_number]
    
    fig = go.Figure()

    for container in containers:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.3, 0.7],  # Adjust column widths: 30% for table, 70% for plot
            specs=[[{"type": "table"}, {"type": "scene"}]]  # Left: Table, Right: 3D plot
        )

        destination_codes = set()
        awb_data = defaultdict(lambda: {'DestinationCode': None, 'Count': 0})

        product_found = False
        for p in placed_products:
            if p['container'] == container['id']:  # Compare string IDs
                product_found = True
                x, y, z, l, w, h = p['position']
                destination_codes.add(p['DestinationCode'])
                
                awb_data[p['awb_number']]['DestinationCode'] = p['DestinationCode']
                awb_data[p['awb_number']]['Count'] += 1

                fig.add_trace(go.Mesh3d(
                    x=[x, x + l, x + l, x, x, x + l, x + l, x],
                    y=[y, y, y + w, y + w, y, y, y + w, y + w],
                    z=[z, z, z, z, z + h, z + h, z + h, z + h],
                    alphahull=0,
                    color=colors[p['id'] % len(colors)],
                    opacity=1.0,
                    lightposition=dict(x=0, y=0, z=10),  # Disables shadowing by adjusting light source
                    lighting=dict(ambient=1, diffuse=0, specular=0),  # Controls the shading and shadowing
                    name=f"{p['awb_number']})"
                ), row=1, col=2)

        if not product_found:
            awb_table_data = pd.DataFrame(columns=['AWB Number', 'DestinationCode', 'Pieces'])
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],  # No products, so leave these empty
                mode='lines',
                line=dict(color='grey', width=4),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)
        else:
            awb_table_data = pd.DataFrame([{
                'AWB Number': awb, 
                'DestinationCode': data['DestinationCode'], 
                'Pieces': data['Count']
            } for awb, data in awb_data.items()])
            awb_table_data.sort_values(by='Pieces', inplace=True, ascending=False)
        
        # Add table to the left column
        table_trace = go.Table(
            header=dict(values=['AWB Number', 'Destination Code', 'Pieces'], fill_color='lightblue', align='left'),
            cells=dict(
                values=[ 
                    awb_table_data['AWB Number'],
                    awb_table_data['DestinationCode'],
                    awb_table_data['Pieces']
                ],
                fill_color='white',
                align='left'
            )
        )
        fig.add_trace(table_trace, row=1, col=1)

        # Container dimensions
        L, W, H = container['Length'], container['Width'], container['Height']
        HX, WX = container['Heightx'], container['Widthx']  # Example offsets
        W_offset = (W - WX) / 2

        if container['SD'] == 'S':
            if container['TB'] == 'T':
                # Define the vertices based on given parameters
                vertices = np.array([  # (same as the original code)
                    [0, 0, 0],          # 0
                    [0, W, 0],          # 1
                    [0, 0, H],          # 2
                    [0, WX, H],         # 3
                    [0, W, HX],         # 4
                    [L, 0, 0],          # 5
                    [L, W, 0],          # 6
                    [L, 0, H],          # 7
                    [L, WX, H],         # 8
                    [L, W, HX]          # 9
                ])
            elif container['TB'] == 'B':
                vertices = np.array([  # (same as the original code)
                    [0, 0, 0],          # 0
                    [0, WX, 0],         # 1
                    [0, 0, H],          # 2
                    [0, W, H],          # 3
                    [0, W, H - HX],     # 4
                    [L, 0, 0],          # 5
                    [L, WX, 0],         # 6
                    [L, 0, H],          # 7
                    [L, W, H],          # 8
                    [L, W, H - HX]      # 9
                ])
            edges = [  # (same as the original code)
                [0, 1], [1, 4], [4, 3], [0, 2], [2, 3],  
                [5, 6], [6, 9], [9, 8], [5, 7], [7, 8],  
                [3, 8], [4, 9], [1, 6],  
                [2, 7], [0, 5]  
            ]
            faces = [
                # Left Face [0, 1, 4, 3, 2]
                [0, 1, 4],
                [0, 4, 3],
                [0, 3, 2],

                # Right Face [5, 6, 9, 8, 7]
                [5, 6, 9],
                [5, 9, 8],
                [5, 8, 7],

                # Front Face [0, 1, 6, 5]
                [0, 1, 6],
                [0, 6, 5],

                # Back Face [2, 3, 8, 7]
                [2, 3, 8],
                [2, 8, 7],

                # Top Face [1, 4, 9, 6]
                [1, 4, 9],
                [1, 9, 6],

                # Bottom Face [0, 2, 7, 5]
                [0, 2, 7],
                [0, 7, 5]
            ]
        elif container['SD'] == 'D':
            if container['TB'] == 'T':
                vertices = np.array([
                    [0, 0, 0],             # 0
                    [0, W, 0],             # 1
                    [0, 0, HX],            # 2
                    [0, W_offset, H],      # 3
                    [0, W_offset + WX, H], # 4
                    [0, W, HX],            # 5
                    [L, 0, 0],             # 6
                    [L, W, 0],             # 7
                    [L, 0, HX],            # 8
                    [L, W_offset, H],      # 9
                    [L, W_offset + WX, H], # 10
                    [L, W, HX]             # 11
                ])
            elif container['TB'] == 'B':
                vertices = np.array([
                    [0, W_offset, 0],             # 0
                    [0, W_offset + WX, 0],             # 1
                    [0, 0, H-HX],            # 2
                    [0, 0, H],      # 3
                    [0, W, H], # 4
                    [0, W, H-HX],            # 5
                    [L, W_offset, 0],             # 0
                    [L, W_offset + WX, 0],             # 1
                    [L, 0, H-HX],            # 2
                    [L, 0, H],      # 3
                    [L, W, H], # 4
                    [L, W, H-HX]             # 11
                ])
            edges = [
                [0, 1], [1, 5], [5, 2], [2, 0], # Left base
                [6, 7], [7, 11], [11, 8], [8, 6], # Right base
                [2, 3], [3, 4], [4, 5], # Left top
                [8, 9], [9, 10], [10, 11], # Right top
                [3, 9], [4, 10], # Connecting edges
                [0, 6], [1, 7], [2, 8], [5, 11] # Vertical edges
            ]
            faces = [
                # Left Base [0, 1, 5, 2]
                [0, 1, 5],
                [0, 5, 2],

                # Right Base [6, 7, 11, 8]
                [6, 7, 11],
                [6, 11, 8],

                # Left Top [2, 3, 4, 5]
                [2, 3, 4],
                [2, 4, 5],

                # Right Top [8, 9, 10, 11]
                [8, 9, 10],
                [8, 10, 11],

                # Front Vertical [0, 2, 8, 6]
                [0, 2, 8],
                [0, 8, 6],

                # Back Vertical [1, 7, 11, 5]
                [1, 7, 11],
                [1, 11, 5],

                # Top Connector 1 [3, 9, 8, 2]
                [3, 9, 8],
                [3, 8, 2],

                # Top Connector 2 [4, 10, 9, 3]
                [4, 10, 9],
                [4, 9, 3],

                # Top Connector 3 [5, 11, 10, 4]
                [5, 11, 10],
                [5, 10, 4]
            ]

        # Determine if this container is blocked
        is_blocked = any(b['id'] == container['id'] for b in blocked_for_ULD)  # Compare string IDs

        # Set styling based on blockage
        container_opacity = 0.0 if not is_blocked else 1.0  # Opacity lower if blocked

        if is_blocked:
            if faces:
                i, j, k = zip(*faces)
                x, y, z = [], [], []
                for vertex in vertices:
                    x.append(vertex[0])
                    y.append(vertex[1])
                    z.append(vertex[2])

                # Now, add the Mesh3d trace
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i, j=j, k=k,
                    color='grey',
                    opacity=container_opacity,
                    showscale=False,
                    lightposition=dict(x=0, y=0, z=10),  # Disables shadowing by adjusting light source
                    lighting=dict(ambient=1, diffuse=0, specular=0),  # Controls the shading and shadowing
                ), row=1, col=2)

        # Extract edge coordinates
        edge_x, edge_y, edge_z = [], [], []
        for start, end in edges:
            edge_x += [vertices[start][0], vertices[end][0], None]
            edge_y += [vertices[start][1], vertices[end][1], None]
            edge_z += [vertices[start][2], vertices[end][2], None]

        if container['Type'] == 'Container':
            # Add wireframe container
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='black', width=4),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)

        elif container['Type'] == 'Pallet':
            # Add wireframe container
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='black', width=4, dash='dot'),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)

        # Calculate aspect ratio
        max_dim = max(L, W, H)
        aspect_ratio = {'x': L / max_dim, 'y': W / max_dim, 'z': H / max_dim}

        if not product_found:
            if not is_blocked:
                # Case where no products are placed in the container
                title_text = f"Container {container['ULDCategory']} - {container['id']}"
            else:
                title_text = f"Container {container['ULDCategory']} - {container['id']}<br>Blocked for BUP"
        else:
            # Case where products are placed in the container
            destination_codes_text = ', '.join(destination_codes)  # Convert the set to a comma-separated string
            title_text = f"Container {container['ULDCategory']} - {container['id']} and Placed Products<br>Destinations: {destination_codes_text}"

        # Update layout with the title - NO GRID, NO AXES, NO BACKGROUND
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    visible=False,  # Hide x-axis completely
                    showgrid=False,
                    showbackground=False,
                    showaxeslabels=False,
                    showticklabels=False,
                    showspikes=False,
                    showline=False,
                    zeroline=False
                ),
                yaxis=dict(
                    visible=False,  # Hide y-axis completely
                    showgrid=False,
                    showbackground=False,
                    showaxeslabels=False,
                    showticklabels=False,
                    showspikes=False,
                    showline=False,
                    zeroline=False
                ),
                zaxis=dict(
                    visible=False,  # Hide z-axis completely
                    showgrid=False,
                    showbackground=False,
                    showaxeslabels=False,
                    showticklabels=False,
                    showspikes=False,
                    showline=False,
                    zeroline=False
                ),
                aspectratio=aspect_ratio,
                bgcolor='rgba(0,0,0,0)'  # Transparent scene background
            ),
            title=title_text,
            title_x=0.5,
            margin=dict(l=0, r=0, t=40, b=0),  # Keep some top margin for the title
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper background
            plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot background
        )

    return fig

# Flask route to receive data from ASP.NET - MODIFIED WITH LOGGING
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
        container_summary = fun.create_container_product_summary(global_placed_products)
        
        # Create the table data
        table_df = fun.create_container_summary_table(container_summary)
        
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
            global_placed_products, global_containers, global_blocked_for_ULD, global_placed_ulds = mn.main(
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

      # NEW: Add simplified container list with essential info
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
    
    # NEW: Add simplified list of blocked containers (if any)
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

# Define the app layout - SIMPLIFIED, removed title and message box
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
    
    # Use your existing visualization function with the global data
    fig = visualize_specific_containers_with_plotly(
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
        app_url = os.environ.get('WEBSITE_HOSTNAME', 'your-app-name.azurewebsites.net')
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