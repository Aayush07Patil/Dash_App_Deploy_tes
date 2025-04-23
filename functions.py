import pandas as pd
import numpy as np
#import pprint
from itertools import permutations
import math
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def convert_cms_to_inches(df, dimension_cols=['Length', 'Breadth', 'Height'], unit_col='MeasureUnit'):
    """
    Converts dimension columns from centimeters to inches for rows where the measurement unit is 'Cms'.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - dimension_cols (list): List of column names (dimensions) to convert from cm to inches.
    - unit_col (str): Name of the column that holds the measurement unit (e.g., 'MeasureUnit').

    Returns:
    - pd.DataFrame: The DataFrame with dimensions converted to inches and the unit updated.
    """
    # Convert only rows where MeasureUnit is 'Cms'
    mask = df[unit_col] == 'Cms'
    
    for col in dimension_cols:
        df.loc[mask, col] = df.loc[mask, col] / 2.54  # cm to inch

    # Update the unit
    df.loc[mask, unit_col] = 'Inches'
    
    return df

def round_columns(df, columns, n=2):
    """
    Rounds the specified columns of a DataFrame to n decimal places.

    Parameters:
    - df (pd.DataFrame): The DataFrame to modify.
    - columns (list): List of column names to round.
    - n (int): Number of decimal places to round to.

    Returns:
    - pd.DataFrame: The DataFrame with rounded values.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].round(n)
    return df

def organize_cargo(cargo):
    """
    Organizes cargo records into a structured list.

    Returns a list where:
    - The first element is a list of all cargo with PieceType 'ULD'
    - The remaining elements are lists of cargo for each DestinationCode (excluding 'ULD' types),
      sorted by increasing Volume.

    Parameters:
    - cargo (pd.DataFrame): The cargo DataFrame.

    Returns:
    - list: A list of lists as described above.
    """
    # Separate ULD and non-ULD cargo
    uld_cargo = cargo[cargo['PieceType'] == 'ULD']
    non_uld_cargo = cargo[cargo['PieceType'] != 'ULD']

    # Convert ULD cargo rows to dictionaries
    uld_list = uld_cargo.to_dict(orient='records')

    # Group non-ULD cargo by DestinationCode and sort each group by Volume
    grouped_by_destination = []
    for dest, group in non_uld_cargo.groupby('DestinationCode'):
        sorted_group = group.sort_values('Volume', ascending=True)
        grouped_by_destination.append(sorted_group.to_dict(orient='records'))

    # Final structured list
    return [uld_list] + grouped_by_destination

def palettes_to_list(palettes_df):
    """
    Converts a palettes DataFrame into a list of dictionaries.

    Parameters:
    - palettes_df (pd.DataFrame): The dataframe containing available pallets.

    Returns:
    - list: A list where each item is a dictionary representing a pallet.
    """
    return palettes_df.to_dict(orient='records')

def volumes_list_for_each_destination(cargo_list):
    """
    Calculates total volume per group in the structured cargo list.

    Parameters:
    - cargo_list (list): A structured list of cargo, where:
        - cargo_list[0] is a list of ULDs
        - cargo_list[1:] are lists grouped by DestinationCode

    Returns:
    - dict: A dictionary with total volume per group:
        { "ULDs": volume, "JFK": volume, "LHR": volume, ... }
    """
    result = {}

    # Sum for ULDs
    result["ULDs"] = sum(item["Volume"] for item in cargo_list[0])

    # Sum for each DestinationCode group
    for sublist in cargo_list[1:]:
        if not sublist:
            continue  # skip empty groups if any
        destination_code = sublist[0].get("DestinationCode", "Unknown")
        result[destination_code] = sum(item["Volume"] for item in sublist)

    return result

def write_cargo_list_to_text(structured_cargo, filepath='cargo_list.txt'):
    """
    Writes the structured cargo list to a text file, preserving the exact Python structure.

    Parameters:
    - structured_cargo (list): The output list from organize_cargo().
    - filepath (str): Path to the output text file.

    Returns:
    - None
    """
    #with open(filepath, 'w') as f:
        #pp = pprint.PrettyPrinter(indent=4, stream=f)
        #pp.pprint(structured_cargo)

def write_palettes_list_to_text(palettes_list, filepath='palettes_list.txt'):
    """
    Writes the list of palette dictionaries to a text file in exact Python format.

    Parameters:
    - palettes_list (list): List of palette dictionaries.
    - filepath (str): Output file path.

    Returns:
    - None
    """
    #with open(filepath, 'w') as f:
        #pp = pprint.PrettyPrinter(indent=4, stream=f)
        #pp.pprint(palettes_list)

def write_list_to_text(list, filepath):
    """
    Writes the list of palette dictionaries to a text file in exact Python format.

    Parameters:
    - palettes_list (list): List of palette dictionaries.
    - filepath (str): Output file path.

    Returns:
    - None
    """
    #with open(filepath, 'w') as f:
        #pp = pprint.PrettyPrinter(indent=4, stream=f)
        #pp.pprint(list)

def get_orientations(product):
    return set(permutations([product['Length'], product['Breadth'], product['Height']]))

def fits(container, placed_products, x, y, z, l, w, h):
    epsilon = 1e-6
    if container['SD'] == 'S':
        if container['TB'] == "B":
            limit = container['Height'] - container['Heightx']
            if z < limit:
                # Check container bounds
                if x + l > container['Length'] + epsilon or y + w > container['Widthx'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
            else:
                if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
        elif container['TB'] == "T":
            if z < container['Heightx']:
                # Check container bounds
                if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
            else:
                if x + l > container['Length'] + epsilon or y + w > container['Widthx'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
    
    elif container['SD'] == "D":
        if container['TB'] == "B":
            limit_height = container['Height'] - container['Heightx']
            width_small = (container['Width'] - container['Widthx'])/2
            if z < limit_height:
                # Check container bounds
                if x + l > container['Length'] + epsilon or y < width_small - epsilon or y + w > container['Width'] - width_small + epsilon or z + h > container['Height'] + epsilon:
                    return False
            else:
                if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
        elif container['TB'] == "T":
            width_small = (container['Width'] - container['Widthx'])/2
            if z < container['Heightx']:
                # Check container bounds
                if x + l > container['Length'] + epsilon or y + w > container['Width'] + epsilon or z + h > container['Height'] + epsilon:
                    return False
            else:
                if x + l > container['Length'] + epsilon or y < width_small - epsilon or y + w > container['Width'] - width_small + epsilon or z + h > container['Height'] + epsilon:
                    return False
        
        
        
    # Check for overlap with existing products
    for p in placed_products:
        px, py, pz, pl, pw, ph = p['position']
        if not (x + l <= px or px + pl <= x + epsilon or
                y + w <= py or py + pw <= y + epsilon or
                z + h <= pz or pz + ph <= z + epsilon):
            return False

    return True

def preprocess_containers_and_products(products, containers, blocked_for_ULD, placed_ulds, placed_products):
    # Filter products of type 'ULD'
    uld_products = [p for p in products if p['PieceType'] == 'ULD']
    blocked_containers = []

    # Check if containers with matching ULDCategory are available
    for product in uld_products:
        matching_container = next((c for c in containers if c['ULDCategory'] == product['ULDCategory']), None)
        if matching_container:
            print(f"Product {product['id']} (ULDCategory: {product['ULDCategory']}) blocks container {matching_container['id']}.")
            product_data = {
                            'id': product['id'],
                            'Length': product['Length'],
                            'Breadth': product['Breadth'],
                            'Height': product['Height'],
                            'position': (0,0,0,0,0,0),
                            'container': matching_container['id'],
                            'Volume': product['Volume'],
                            'DestinationCode': product['DestinationCode'],
                            'awb_number': product['awb_number']     
                        }
            placed_ulds.append(product_data)
            placed_products.append(product_data)
            blocked_containers.append(matching_container)
            blocked_for_ULD.append(matching_container)
            containers.remove(matching_container)
            products.remove(product)  # Exclude the product from packing

    # Remove blocked containers from the container list
    containers = [c for c in containers if c not in blocked_containers]
    return products, containers, blocked_containers, blocked_for_ULD, placed_ulds, placed_products

def pack_products_sequentially(containers, products, blocked_containers, DC_total_volumes, placed_products):
    """
    Pack products into containers sequentially based on volume constraints and dimensions.

    Parameters:
        containers (list): List of available containers with their dimensions and volumes.
        products (list): List of products to be placed with their dimensions and volumes.
        blocked_containers (list): List of containers that are blocked.
        DC_total_volumes (dict): Mapping of destination codes to total allowable volumes.

    Returns:
        tuple: (placed_products, remaining_products, blocked_containers, available_containers)
    """
    #placed_products = []
    remaining_products = products[:]  # Ensure we work with a copy of the products list
    used_containers = []
    missed_product_count = 0
    running_volume_sum = 0

    if products:
        destination_code = products[0]['DestinationCode']
        print(f"\nProcessing Destination Code: {destination_code}")
    else:
        destination_code = 'ULDs'

    for container in containers:
        print(f"Placing products in container {container['id']} ({container['ULDCategory']})")
        container_placed = []  # Products placed in the current container
        container_volume = container['Volume']
        occupied_volume = 0

        for product in remaining_products[:]:  # Iterate over a copy of remaining_products to avoid modification during iteration
            # Skip products that have already been placed
            if product in placed_products:
                continue

            dc_volume = DC_total_volumes.get(product['DestinationCode'], 0)

            # Check volume constraints
            if not (dc_volume - running_volume_sum) > 0.8 * container_volume:
                print("Volume constraint not satisfied, stopping process.")
                blocked_containers.extend(used_containers)
                remaining_containers = [c for c in containers if c not in blocked_containers]
                running_volume_sum = 0
                return placed_products, remaining_products, blocked_containers, remaining_containers

            if missed_product_count >= 3:
                print("Too many missed products. Blocking containers.")
                blocked_containers.extend(used_containers)
                break

            placed = try_place_product(product, container, container_placed, occupied_volume, placed_products)

            if placed:
                running_volume_sum += product['Volume']
                remaining_products.remove(product)  # Remove placed product from remaining products list
                if container not in used_containers:
                    used_containers.append(container)
            else:
                print(f"Product {product['id']} could not be placed in container {container['id']}.")
                missed_product_count += 1

        if not remaining_products:
            print(f"All products have been placed for {destination_code}")
            running_volume_sum = 0
            blocked_containers.extend(used_containers)
            break

        # Reverse remaining products for next iteration
        if missed_product_count >= 3:
            print("Reversing product list for retry.")
            remaining_products = remaining_products[::-1]
            missed_product_count = 0

    remaining_containers = [c for c in containers if c not in blocked_containers]
    return placed_products, remaining_products, blocked_containers, remaining_containers



def try_place_product(product, container, container_placed, occupied_volume, placed_products):
    """
    Attempt to place a product in the container.

    Parameters:
        product (dict): The product to be placed.
        container (dict): The container to place the product in.
        container_placed (list): List of already placed products in the container.
        occupied_volume (float): Current occupied volume of the container.
        placed_products (list): List of all placed products.

    Returns:
        bool: True if the product was successfully placed, False otherwise.
    """
    for orientation in get_orientations(product):
        l, w, h = orientation
        for x in range(0, math.floor(container['Length'] - l)):
            for y in range(0, math.floor(container['Width'] - w)):
                for z in range(0, math.floor(container['Height'] - h)):
                    if fits(container, container_placed, x, y, z, l, w, h):
                        product_data = {
                            'id': product['id'],
                            'Length': l,
                            'Breadth': w,
                            'Height': h,
                            'position': (x, y, z, l, w, h),
                            'container': container['id'],
                            'Volume': product['Volume'],
                            'DestinationCode': product['DestinationCode'],
                            'awb_number': product['awb_number']
                        }
                        container_placed.append(product_data)
                        placed_products.append(product_data)
                        occupied_volume += product['Volume']
                        remaining_volume_percentage = round(((container['Volume'] - occupied_volume) * 100 / container['Volume']), 2)
                        print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume: {remaining_volume_percentage}%")
                        return True
    return False

def process(products, containers, blocked_containers, DC_total_volumes):
    containers_tp = containers[:]
    blocked_for_ULD = []
    placed = []
    placed_products = []
    unplaced = []
    placed_ulds = []
    placements = {container['id']: [] for container in containers_tp}  # Tracks placements per container

    # First pass: Process each product
    for product in products:
        # Preprocess containers and products to block ULD-related containers
        products, containers_tp, blocked_containers, blocked_for_ULD, placed_ulds, placed_products = preprocess_containers_and_products(product, containers_tp, blocked_for_ULD, placed_ulds, placed_products)

        # Place products sequentially
        placed_products, remaining_products, blocked_containers, containers_tp = pack_products_sequentially(
            containers_tp, products, blocked_containers, DC_total_volumes, placed_products
        )

        # Record placements and update lists
        for p in placed_products:
            container_id = p['container']
            placements[container_id].append(p['position'])  # Store placement data
        placed.extend(placed_products)
        unplaced.extend(remaining_products)
        
    print("\nAttempting to place unplaced products in remaining spaces...")
    second_pass_placed = []
    placed = sorted(placed, key=lambda x:x['Volume'])
    used_container = []
    containers_tp = containers[:]
    containers_tp = [item for item in containers_tp if item not in blocked_for_ULD]
    for container in containers_tp:
        placed_products_in_container = [item for item in placed if item["container"] == container["id"]]
        total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
        if total_volume_of_placed_products >0.8*container['Volume']:
            containers_tp.remove(container)
    
    if containers_tp:
        unplaced = sorted(unplaced,key=lambda x:x["Volume"])
        missed_products_count = 0 
        for container in containers_tp:
            placed_products_in_container = [item for item in placed if item["container"] == container["id"]]
            total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
            container_volume = container['Volume']
            occupied_volume = total_volume_of_placed_products
            for product in unplaced:
                placed_ = False
                if missed_products_count < 3:
                    for orientation in get_orientations(product):
                        l, w, h = orientation
                        
                        for x in range(0,math.floor(container['Length'] - l)):
                            for y in range(0,math.floor(container['Width'] - w )):
                                for z in range(0,math.floor(container['Height'] - h)):
                                    if fits(container, placed_products_in_container, x, y, z, l, w, h):
                                        product_data = {
                                            'id': product['id'],
                                            'Length': l,
                                            'Breadth': w,
                                            'Height': h,
                                            'position': (x, y, z, l, w, h),
                                            'container': container['id'],
                                            'Volume': product['Volume'],
                                            'DestinationCode': product['DestinationCode'],
                                            'awb_number': product['awb_number']     
                                        }
                                        occupied_volume += product['Volume']
                                        remaining_volume_percentage =  round(((container['Volume'] - occupied_volume) * 100 / container['Volume']), 2)
                                        print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume in container = {remaining_volume_percentage}")
                                        placed.append(product_data)
                                        placed_products_in_container.append(product_data)
                                        unplaced.remove(product)
                                        placed_ = True
                                        if container not in used_container:
                                            used_container.append(container)
                                        break
                                if placed_:
                                    break
                            if placed_:
                                break
                        if placed_:
                            break
                    if not placed_:
                        print(f"Product {product['id']} could not be placed in container {container['id']}.")
                        missed_products_count += 1
                        if container not in used_container:
                            used_container.append(container)
            
            else:
                
                print("\nSwitching list around\n")
                unplaced = unplaced[::-1]
                missed_products_count = 0 
                
                for product in unplaced:
                    placed_ = False
                    if missed_products_count < 3:
                        for orientation in get_orientations(product):
                            l, w, h = orientation
                            
                            for x in range(0,math.floor(container['Length'] - l)):
                                for y in range(0,math.floor(container['Width'] - w )):
                                    for z in range(0,math.floor(container['Height'] - h)):
                                        if fits(container, placed_products_in_container, x, y, z, l, w, h):
                                            product_data = {
                                                'id': product['id'],
                                                'Length': l,
                                                'Breadth': w,
                                                'Height': h,
                                                'position': (x, y, z, l, w, h),
                                                'container': container['id'],
                                                'Volume': product['Volume'],
                                                'DestinationCode': product['DestinationCode'],
                                                'awb_number': product['awb_number']
                                            }
                                            occupied_volume += product['Volume']
                                            remaining_volume_percentage = (container_volume - occupied_volume)/container_volume
                                            print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume in container = {remaining_volume_percentage}")
                                            placed.append(product_data)
                                            placed_products_in_container.append(product_data)
                                            unplaced.remove(product)
                                            placed_ =True
                                            if container not in used_container:
                                                used_container.append(container)
                                            break
                                    if placed_:
                                        break
                                if placed_:
                                    break
                            if placed_:
                                break
                        if not placed_:
                            print(f"Product {product['id']} could not be placed in container {container['id']}.")
                            missed_products_count += 1
                            if container not in used_container:
                                used_container.append(container)
                            unplaced = unplaced[::-1]
                    
                    else:
                        break
            
            if not unplaced:
                print("All products have been placed.")
                break
                
    return placed, unplaced, blocked_for_ULD, placed_ulds

def visualize_separate_containers_with_plotly(containers, placed_products, blocked_for_ULD):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'lime', 'magenta']

    for container in containers:
        # Initialize subplot with two columns
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.3, 0.7],  # Adjust column widths: 30% for table, 70% for plot
            specs=[[{"type": "table"}, {"type": "scene"}]]  # Left: Table, Right: 3D plot
        )
        
        destination_codes = set()
        awb_data = defaultdict(lambda: {'DestinationCode': None, 'Count': 0})  # Tracks AWBs with DestinationCode and count
        product_found = False  # Flag to check if products are placed in this container

        # Add products to container
        for p in placed_products:
            if p['container'] == container['id']:
                product_found = True  # We found at least one product for this container
                x, y, z, l, w, h = p['position']
                destination_codes.add(p['DestinationCode'])
                
                # Update awb_data
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

        # Handle case where no products are placed in the container
        if not product_found:
            # Create an empty table
            awb_table_data = pd.DataFrame(columns=['AWB Number', 'DestinationCode', 'Pieces'])

            # Add empty wireframe container
            fig.add_trace(go.Scatter3d(
                x=[], y=[], z=[],  # No products, so leave these empty
                mode='lines',
                line=dict(color='grey', width=4),
                name=f"Container {container['ULDCategory']} - {container['id']}"
            ), row=1, col=2)

        else:
            # Convert awb_data to a DataFrame
            awb_table_data = pd.DataFrame([{
                'AWB Number': awb, 
                'DestinationCode': data['DestinationCode'], 
                'Pieces': data['Count']
            } for awb, data in awb_data.items()])
            awb_table_data.sort_values(by='Pieces', inplace=True, ascending=False)  # Optional: Sort by AWB Number

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
                # Define the vertices based on given parameters
                vertices = np.array([
                    [0, 0, 0],          # 0
                    [0, WX, 0],          # 1
                    [0, 0, H],          # 2
                    [0, W, H],         # 3
                    [0, W, H- HX],         # 4
                    [L, 0, 0],          # 5
                    [L, WX, 0],          # 6
                    [L, 0, H],          # 7
                    [L, W, H],         # 8
                    [L, W, H-HX]          # 9
                ])
            edges = [
                [0, 1], [1, 4], [4, 3], [0, 2], [2, 3],  # Left side edges
                [5, 6], [6, 9], [9, 8], [5, 7], [7, 8],  # Right side edges
                [3, 8], [4, 9], [1, 6],  # Connecting edges
                [2, 7], [0, 5]  # Connecting edges
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
        is_blocked = any(b['id'] == container['id'] for b in blocked_for_ULD)

        # Set styling based on blockage
        container_opacity = 0.0 if not is_blocked else 1.0  # Opacity lower if blocked

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

        if not product_found:
            if not is_blocked:
                # Case where no products are placed in the container
                title_text = f"Container {container['ULDCategory']} - {container['id']}"
            else:
                title_text = f"Container {container['ULDCategory']} - {container['id']}<br>Blocked for ULD"
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
            title=title_text,  # Set the dynamic title here
            title_x=0.5
        )
        
        # Show the figure
        fig.show()


def create_container_product_summary(placed_products):
    """
    Creates a structured summary of products placed in each container.
    
    Args:
        placed_products (list): List of dictionaries with placed product data
        
    Returns:
        dict: Nested dictionary with container -> awb_number -> dimensions -> count
              in a format ready for JSON serialization
    """
    # Initialize the container report dictionary
    container_summary = {}
    
    # Process each placed product
    for product in placed_products:
        container_id = product['container']
        awb_number = product['awb_number']
        position = product['position']
        destination_code = product['DestinationCode']
        
        # Extract dimensions from position (x, y, z, length, width, height)
        _, _, _, length, width, height = position
        
        # Create a dimension key that can be used in JSON
        dimensions = f"{length:.2f}x{width:.2f}x{height:.2f}"
        
        # Initialize container entry if not exists
        if container_id not in container_summary:
            container_summary[container_id] = {
                "awb_data": {},
                "total_products": 0
            }
        
        # Initialize AWB entry if not exists
        if awb_number not in container_summary[container_id]["awb_data"]:
            container_summary[container_id]["awb_data"][awb_number] = {
                "destination_code": destination_code,
                "dimensions": {},
                "total_count": 0
            }
        
        # Initialize or increment the count for this dimension
        if dimensions not in container_summary[container_id]["awb_data"][awb_number]["dimensions"]:
            container_summary[container_id]["awb_data"][awb_number]["dimensions"][dimensions] = 1
        else:
            container_summary[container_id]["awb_data"][awb_number]["dimensions"][dimensions] += 1
        
        # Update the total counts
        container_summary[container_id]["awb_data"][awb_number]["total_count"] += 1
        container_summary[container_id]["total_products"] += 1
    
    return container_summary

def create_container_summary_table(container_summary):
    """
    Converts the container_summary dictionary into a pandas DataFrame 
    with columns: ContainerID, AWBnumber, Dimensions, Pieces
    
    Args:
        container_summary (dict): The nested dictionary output from create_container_product_summary
        
    Returns:
        pd.DataFrame: A DataFrame with the requested columns
    """
    # Initialize empty lists for each column
    container_ids = []
    awb_numbers = []
    dimensions_list = []
    pieces_list = []
    
    # Iterate through the container summary dictionary
    for container_id, container_data in container_summary.items():
        for awb_number, awb_data in container_data["awb_data"].items():
            for dimensions, count in awb_data["dimensions"].items():
                # Append values to the respective lists
                container_ids.append(container_id)
                awb_numbers.append(awb_number)
                dimensions_list.append(dimensions)
                pieces_list.append(count)
    
    # Create a DataFrame with the collected data
    df = pd.DataFrame({
        'ContainerID': container_ids,
        'AWBnumber': awb_numbers,
        'Dimensions': dimensions_list,
        'Pieces': pieces_list
    })
    
    # Sort the DataFrame by ContainerID and AWBnumber for better readability
    df = df.sort_values(['ContainerID', 'AWBnumber', 'Dimensions'])
    
    return df