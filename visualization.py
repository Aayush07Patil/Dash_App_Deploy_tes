import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from collections import defaultdict

def visualize_specific_containers_with_plotly(containers, placed_products, blocked_for_ULD, placed_ulds, container_number=None):
    """
    Creates a visualiation of specific containers with their placed products.
    
    Args:
        containers (list): List of container objects
        placed_products (list): List of placed product objects
        blocked_for_ULD (list): List of containers blocked for ULD
        placed_ulds (list): List of placed ULD objects
        container_number (str, optional): Specific container ID to visualize
        
    Returns:
        go.Figure: Plotly figure object with container visualization
    """
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

        # Create vertices and edges based on container shape type
        if container['SD'] == 'S':
            if container['TB'] == 'T':
                # Define the vertices for a single top container
                vertices = np.array([
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
                # Define the vertices for a single bottom container
                vertices = np.array([
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
            edges = [
                [0, 1], [1, 4], [4, 3], [0, 2], [2, 3],  
                [5, 6], [6, 9], [9, 8], [5, 7], [7, 8],  
                [3, 8], [4, 9], [1, 6],  
                [2, 7], [0, 5]  
            ]
            faces = [
                # Left Face [0, 1, 4, 3, 2]
                [0, 1, 4], [0, 4, 3], [0, 3, 2],
                # Right Face [5, 6, 9, 8, 7]
                [5, 6, 9], [5, 9, 8], [5, 8, 7],
                # Front Face [0, 1, 6, 5]
                [0, 1, 6], [0, 6, 5],
                # Back Face [2, 3, 8, 7]
                [2, 3, 8], [2, 8, 7],
                # Top Face [1, 4, 9, 6]
                [1, 4, 9], [1, 9, 6],
                # Bottom Face [0, 2, 7, 5]
                [0, 2, 7], [0, 7, 5]
            ]
        elif container['SD'] == 'D':
            if container['TB'] == 'T':
                # Define the vertices for a double top container
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
                # Define the vertices for a double bottom container
                vertices = np.array([
                    [0, W_offset, 0],             # 0
                    [0, W_offset + WX, 0],        # 1
                    [0, 0, H-HX],                 # 2
                    [0, 0, H],                    # 3
                    [0, W, H],                    # 4
                    [0, W, H-HX],                 # 5
                    [L, W_offset, 0],             # 6
                    [L, W_offset + WX, 0],        # 7
                    [L, 0, H-HX],                 # 8
                    [L, 0, H],                    # 9
                    [L, W, H],                    # 10
                    [L, W, H-HX]                  # 11
                ])
            edges = [
                [0, 1], [1, 5], [5, 2], [2, 0],   # Left base
                [6, 7], [7, 11], [11, 8], [8, 6], # Right base
                [2, 3], [3, 4], [4, 5],           # Left top
                [8, 9], [9, 10], [10, 11],        # Right top
                [3, 9], [4, 10],                  # Connecting edges
                [0, 6], [1, 7], [2, 8], [5, 11]   # Vertical edges
            ]
            faces = [
                # Left Base [0, 1, 5, 2]
                [0, 1, 5], [0, 5, 2],
                # Right Base [6, 7, 11, 8]
                [6, 7, 11], [6, 11, 8],
                # Left Top [2, 3, 4, 5]
                [2, 3, 4], [2, 4, 5],
                # Right Top [8, 9, 10, 11]
                [8, 9, 10], [8, 10, 11],
                # Front Vertical [0, 2, 8, 6]
                [0, 2, 8], [0, 8, 6],
                # Back Vertical [1, 7, 11, 5]
                [1, 7, 11], [1, 11, 5],
                # Top Connector 1 [3, 9, 8, 2]
                [3, 9, 8], [3, 8, 2],
                # Top Connector 2 [4, 10, 9, 3]
                [4, 10, 9], [4, 9, 3],
                # Top Connector 3 [5, 11, 10, 4]
                [5, 11, 10], [5, 10, 4]
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

        # Add wireframe based on container type
        if container['Type'] == 'Container':
            # Add wireframe container
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='black', width=4),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)
        elif container['Type'] == 'Pallet':
            # Add wireframe container with dotted line
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='black', width=4, dash='dot'),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)

        # Calculate aspect ratio
        max_dim = max(L, W, H)
        aspect_ratio = {'x': L / max_dim, 'y': W / max_dim, 'z': H / max_dim}

        # Set the appropriate title based on container status
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

def visualize_all_containers_with_plotly(containers, placed_products, blocked_for_ULD, placed_ulds):
    """
    Creates a visualization of containers with placed products for testing purposes.
    Shows full axes and grid for better debugging.
    
    Args:
        containers (list): List of container objects
        placed_products (list): List of placed product objects
        blocked_for_ULD (list): List of containers blocked for ULD
        placed_ulds (list): List of placed ULD objects
        
    Returns:
        None: Displays figures directly
    """
    colors = ['#A97835', '#C08F4F', '#CD9F61', '#C2A574', '#D6B88A','#E6D3B3','#CBB994']

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

        # Create vertices and edges based on container shape type
        if container['SD'] == 'S':
            if container['TB'] == 'T':
                # Define the vertices for a single top container
                vertices = np.array([
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
                # Define the vertices for a single bottom container
                vertices = np.array([
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
            edges = [
                [0, 1], [1, 4], [4, 3], [0, 2], [2, 3],  
                [5, 6], [6, 9], [9, 8], [5, 7], [7, 8],  
                [3, 8], [4, 9], [1, 6],  
                [2, 7], [0, 5]  
            ]
            faces = [
                # Left Face [0, 1, 4, 3, 2]
                [0, 1, 4], [0, 4, 3], [0, 3, 2],
                # Right Face [5, 6, 9, 8, 7]
                [5, 6, 9], [5, 9, 8], [5, 8, 7],
                # Front Face [0, 1, 6, 5]
                [0, 1, 6], [0, 6, 5],
                # Back Face [2, 3, 8, 7]
                [2, 3, 8], [2, 8, 7],
                # Top Face [1, 4, 9, 6]
                [1, 4, 9], [1, 9, 6],
                # Bottom Face [0, 2, 7, 5]
                [0, 2, 7], [0, 7, 5]
            ]
        elif container['SD'] == 'D':
            if container['TB'] == 'T':
                # Define the vertices for a double top container
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
                # Define the vertices for a double bottom container
                vertices = np.array([
                    [0, W_offset, 0],             # 0
                    [0, W_offset + WX, 0],        # 1
                    [0, 0, H-HX],                 # 2
                    [0, 0, H],                    # 3
                    [0, W, H],                    # 4
                    [0, W, H-HX],                 # 5
                    [L, W_offset, 0],             # 6
                    [L, W_offset + WX, 0],        # 7
                    [L, 0, H-HX],                 # 8
                    [L, 0, H],                    # 9
                    [L, W, H],                    # 10
                    [L, W, H-HX]                  # 11
                ])
            edges = [
                [0, 1], [1, 5], [5, 2], [2, 0],   # Left base
                [6, 7], [7, 11], [11, 8], [8, 6], # Right base
                [2, 3], [3, 4], [4, 5],           # Left top
                [8, 9], [9, 10], [10, 11],        # Right top
                [3, 9], [4, 10],                  # Connecting edges
                [0, 6], [1, 7], [2, 8], [5, 11]   # Vertical edges
            ]
            faces = [
                # Left Base [0, 1, 5, 2]
                [0, 1, 5], [0, 5, 2],
                # Right Base [6, 7, 11, 8]
                [6, 7, 11], [6, 11, 8],
                # Left Top [2, 3, 4, 5]
                [2, 3, 4], [2, 4, 5],
                # Right Top [8, 9, 10, 11]
                [8, 9, 10], [8, 10, 11],
                # Front Vertical [0, 2, 8, 6]
                [0, 2, 8], [0, 8, 6],
                # Back Vertical [1, 7, 11, 5]
                [1, 7, 11], [1, 11, 5],
                # Top Connector 1 [3, 9, 8, 2]
                [3, 9, 8], [3, 8, 2],
                # Top Connector 2 [4, 10, 9, 3]
                [4, 10, 9], [4, 9, 3],
                # Top Connector 3 [5, 11, 10, 4]
                [5, 11, 10], [5, 10, 4]
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
                    lightposition=dict(x=0, y=0, z=10),  
                    lighting=dict(ambient=1, diffuse=0, specular=0),
                ), row=1, col=2)

        # Extract edge coordinates
        edge_x, edge_y, edge_z = [], [], []
        for start, end in edges:
            edge_x += [vertices[start][0], vertices[end][0], None]
            edge_y += [vertices[start][1], vertices[end][1], None]
            edge_z += [vertices[start][2], vertices[end][2], None]

        # Add wireframe based on container type
        if container['Type'] == 'Container':
            # Add wireframe container
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='black', width=4),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)
        elif container['Type'] == 'Pallet':
            # Add wireframe container with dotted line
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='black', width=4, dash='dot'),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)

        # Calculate aspect ratio
        max_dim = max(L, W, H)
        aspect_ratio = {'x': L / max_dim, 'y': W / max_dim, 'z': H / max_dim}

        # Set the appropriate title based on container status
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

        # Update layout with the title - WITH GRID AND AXES FOR TESTING
        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=10, title='Length'),
                yaxis=dict(nticks=10, title='Width'),
                zaxis=dict(nticks=10, title='Height'),
                aspectratio=aspect_ratio
            ),
            title= title_text,
            title_x=0.5,
            margin=dict(l=0, r=0, t=0, b=0)  # Remove margins
        )

        fig.show()