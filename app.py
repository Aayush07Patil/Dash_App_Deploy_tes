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
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'lime', 'magenta']

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

        elif container['Type'] == 'Palette':
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

        # Title text based on whether products are placed
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

        # Update layout with the title
        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=10, title='Length'),
                yaxis=dict(nticks=10, title='Width'),
                zaxis=dict(nticks=10, title='Height'),
                aspectratio=aspect_ratio
            ),
            title=title_text,
            title_x=0.5
        )

    return fig

# Flask route to receive data from ASP.NET
@app.server.route('/update_data', methods=['POST'])
def update_data():
    global global_containers_df, global_products_df, global_processed
    global global_placed_products, global_containers, global_blocked_for_ULD, global_placed_ulds
    
    try:
        # Get data from POST request
        data = request.get_json()
        
        # Convert to dataframes
        global_containers_df = pd.DataFrame(data['containers'])
        global_products_df = pd.DataFrame(data['products'])
        
        # Reset the processed flag so we'll reprocess the data
        global_processed = False
        
        # Process the data right away
        ensure_data_processed()
        
        return jsonify({'status': 'success'})
    except Exception as e:
        print(f"Error processing data: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# Function to process data if it hasn't been processed yet
def ensure_data_processed():
    global global_containers_df, global_products_df, global_processed
    global global_placed_products, global_containers, global_blocked_for_ULD, global_placed_ulds
    
    # If we have data but haven't processed it yet
    if not global_processed and global_containers_df is not None and global_products_df is not None:
        try:
            # Run your existing processing function
            global_placed_products, global_containers, global_blocked_for_ULD, global_placed_ulds = mn.main(
                global_containers_df, global_products_df
            )
            global_processed = True
        except Exception as e:
            print(f"Error in data processing: {e}")
            # Initialize with empty results if processing fails
            global_placed_products, global_containers = [], []
            global_blocked_for_ULD, global_placed_ulds = [], []
            global_processed = True  # Mark as processed to avoid repeated errors

# Define the app layout - with message to send data first
app.layout = html.Div([
    #html.H1("Container Visualization Dashboard", style={'textAlign': 'center'}),
    
    # Message area for instructions or status
    html.Div([
        html.Div(id='message-area', children=[
            html.H3("Welcome to the Container Visualization Dashboard"),
            html.P("Please send data from the ASP.NET application first by clicking 'Send Data to Dash'."),
            html.P("Then select a specific container to visualize by clicking its button.")
        ], style={
            'textAlign': 'center',
            'padding': '20px',
            'backgroundColor': '#f9f9f9',
            'borderRadius': '8px',
            'marginBottom': '20px',
            'border': '1px solid #ddd'
        })
    ]),
    
    # Graph for displaying the 3D container visualization
    dcc.Graph(id='container-graph'),
    
    # Store the URL params in the dcc.Location component
    dcc.Location(id='url', refresh=False),
    
    # Interval component for periodic refreshes to check for new data
    dcc.Interval(
        id='interval-component',
        interval=180*1000,  # in milliseconds (2 seconds)
        n_intervals=0
    )
])

# Callback to update the visualization based on URL parameters
@app.callback(
    [Output('container-graph', 'figure'),
     Output('message-area', 'children')],
    [Input('url', 'search'),
     Input('interval-component', 'n_intervals')]
)
def update_container_visualization(search, n_intervals):
    # Check if data has been received
    if not global_processed:
        # No data yet, show welcome message
        message = [
            html.H3("Welcome to the Container Visualization Dashboard"),
            html.P("Please send data from the ASP.NET application first by clicking 'Send Data to Dash'."),
            html.P("Then select a specific container to visualize by clicking its button.")
        ]
        return go.Figure(), message
    
    # Parse URL parameters to get container ID
    query_params = parse_qs(search.lstrip('?'))
    selected_container = None
    
    if 'container' in query_params:
        try:
            # No need to convert to int, use as string
            selected_container = query_params['container'][0]
        except (IndexError):
            pass
    
    # Use your existing visualization function with the global data
    fig = visualize_specific_containers_with_plotly(
        global_containers, 
        global_placed_products, 
        global_blocked_for_ULD, 
        global_placed_ulds, 
        selected_container
    )
    
    # Update message based on selected container
    if selected_container:
        message = [
            #html.H3(f"Showing Container {selected_container}"),
        ]
    else:
        message = [
            html.H3("Data Loaded Successfully"),
            html.P("Use the container buttons to select a specific container to visualize.")
        ]
    
    return fig, message

# Run the Dash app - modified for Azure
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))