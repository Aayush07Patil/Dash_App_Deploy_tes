import pandas as pd
import numpy as np
from itertools import permutations
import math
from collections import defaultdict
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time

# --- Utility functions ---

def convert_cms_to_inches(df, dimension_cols=['Length', 'Breadth', 'Height'], unit_col='MeasureUnit'):
    """
    Converts dimension columns from centimeters to inches for rows where the measurement unit is 'Cms'.
    """
    # Create a mask for rows that need conversion - compute once
    mask = df[unit_col] == 'Cms'
    
    # Apply conversion to all relevant columns at once (reduces row lookups)
    for col in dimension_cols:
        df.loc[mask, col] = df.loc[mask, col].astype(float) / 2.54  # cm to inch

    # Update the unit
    df.loc[mask, unit_col] = 'Inches'
    
    return df

def round_columns(df, columns, n=2):
    """
    Rounds the specified columns of a DataFrame to n decimal places.
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].round(n)
    return df

def organize_cargo(cargo):
    """
    Organizes cargo records into a structured list.
    """
    # Optimize by using boolean indexing once
    uld_mask = cargo['PieceType'] == 'ULD'
    uld_cargo = cargo[uld_mask]
    non_uld_cargo = cargo[~uld_mask]  # Use inverse mask

    # Convert ULD cargo rows to dictionaries (compute once)
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
    """
    return palettes_df.to_dict(orient='records')

def volumes_list_for_each_destination(cargo_list):
    """
    Calculates total volume per group in the structured cargo list.
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

def write_list_to_text(list_data, filepath):
    """Stub function for compatibility"""
    pass

# --- Core optimized functions ---

def get_orientations(product):
    """
    Get all possible orientations of a product.
    Optimized to cache dimensions instead of repeated dict lookups.
    """
    # Cache dimensions to avoid repeated dictionary lookups
    l, b, h = product['Length'], product['Breadth'], product['Height']
    return {(pl, pb, h) for pl, pb in permutations([l, b])}

def fits_optimized(container, placed_products, x, y, z, l, w, h):
    """
    Optimized version of fits function that checks if a product fits at a position.
    Uses early returns and caches container properties.
    """
    epsilon = 1e-6
    
    # Cache container properties to avoid repeated dictionary lookups
    c_length = container['Length']
    c_width = container['Width'] 
    c_height = container['Height']
    c_widthx = container['Widthx']
    c_heightx = container['Heightx']
    c_sd = container['SD']
    c_tb = container['TB']
    
    # Quick bounds check first - most common rejection reason
    if x + l > c_length + epsilon or z + h > c_height + epsilon:
        return False
        
    # Handle container shape specifics
    if c_sd == 'S':
        if c_tb == "B":
            limit = c_height - c_heightx
            if z < limit:
                if y + w > c_widthx + epsilon:
                    return False
            else:
                if y + w > c_width + epsilon:
                    return False
        elif c_tb == "T":
            if z < c_heightx:
                if y + w > c_width + epsilon:
                    return False
            else:
                if y + w > c_widthx + epsilon:
                    return False
    
    elif c_sd == "D":
        if c_tb == "B":
            limit_height = c_height - c_heightx
            width_small = (c_width - c_widthx)/2
            if z < limit_height:
                if y < width_small - epsilon or y + w > c_width - width_small + epsilon:
                    return False
            else:
                if y + w > c_width + epsilon:
                    return False
        elif c_tb == "T":
            width_small = (c_width - c_widthx)/2
            if z < c_heightx:
                if y + w > c_width + epsilon:
                    return False
            else:
                if y < width_small - epsilon or y + w > c_width - width_small + epsilon:
                    return False
    
    # Early return if there are no placed products (common case optimization)
    if not placed_products:
        return True
        
    # Check for overlap with existing products - optimized for fewer comparisons
    x2, y2, z2 = x + l, y + w, z + h
    
    for p in placed_products:
        px, py, pz, pl, pw, ph = p['position']
        px2, py2, pz2 = px + pl, py + pw, pz + ph
        
        # Check if boxes overlap in all three dimensions
        if not (x2 <= px or px2 <= x or y2 <= py or py2 <= y or z2 <= pz or pz2 <= z):
            return False

    return True

def try_place_product_optimized(product, container, container_placed, occupied_volume, placed_products):
    """
    Optimized version of try_place_product with smart search strategy.
    Tries corners and edges first, then uses a grid search with early exit.
    """
    # Start time for this product placement attempt
    start_time = time.time()
    max_time = 0.5  # Maximum time to spend on a single product placement (seconds)
    
    # Get container dimensions once
    c_length = math.floor(container['Length'])
    c_width = math.floor(container['Width'])
    c_height = math.floor(container['Height'])
    
    # Try all orientations
    for l, w, h in get_orientations(product):
        # Skip impossible orientations immediately
        if l > c_length or w > c_width or h > c_height:
            continue
            
        # Optimization: Try placing at corners and edges first
        # These positions often yield better packing results
        corners = [(0,0,0), (0,0,c_height-h), (0,c_width-w,0), (0,c_width-w,c_height-h),
                  (c_length-l,0,0), (c_length-l,0,c_height-h), (c_length-l,c_width-w,0), 
                  (c_length-l,c_width-w,c_height-h)]
                  
        for x, y, z in corners:
            if fits_optimized(container, container_placed, x, y, z, l, w, h):
                # Place the product
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
        
        # Regular grid search with early exit
        step = 1  # Step size for grid search
        for x in range(0, c_length - math.floor(l) + 1, step):
            # Check timeout to avoid spending too much time on hard-to-place products
            if time.time() - start_time > max_time:
                print(f"Timeout for product {product['id']}")
                break
                
            for y in range(0, c_width - math.floor(w) + 1, step):
                for z in range(0, c_height - math.floor(h) + 1, step):
                    if fits_optimized(container, container_placed, x, y, z, l, w, h):
                        # Place the product
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

def pack_products_sequentially_optimized(containers, products, blocked_containers, DC_total_volumes, placed_products):
    """
    Optimized version of pack_products_sequentially.
    Pack products into containers sequentially with improved efficiency.
    """
    remaining_products = products[:]
    used_containers = []
    missed_product_count = 0
    running_volume_sum = 0
    
    if products:
        destination_code = products[0]['DestinationCode']
        print(f"\nProcessing Destination Code: {destination_code}")
    else:
        destination_code = 'ULDs'
    
    # Pre-compute container volumes for faster access
    container_volumes = {container['id']: container['Volume'] for container in containers}
    
    for container in containers:
        print(f"Placing products in container {container['id']} ({container['ULDCategory']})")
        container_placed = []  # Products placed in the current container
        container_volume = container['Volume']
        occupied_volume = 0

        # Process each remaining product - more efficient with index-based iteration
        i = 0
        while i < len(remaining_products):
            product = remaining_products[i]
            
            # Skip products that have already been placed
            if product in placed_products:
                i += 1
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

            placed = try_place_product_optimized(product, container, container_placed, occupied_volume, placed_products)
            
            if placed:
                running_volume_sum += product['Volume']
                remaining_products.pop(i)  # More efficient than remove() for large lists
                if container not in used_containers:
                    used_containers.append(container)
            else:
                print(f"Product {product['id']} could not be placed in container {container['id']}.")
                missed_product_count += 1
                i += 1
                
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

def preprocess_containers_and_products_optimized(products, containers, blocked_for_ULD, placed_ulds, placed_products):
    """
    Optimized preprocessing for ULD products and containers.
    Handles a nested list structure for products.
    """
    blocked_containers = []
    
    # Flatten the products list if it's nested
    flat_products = []
    if products and isinstance(products, list):
        for product_group in products:
            if isinstance(product_group, list):
                flat_products.extend(product_group)
            else:
                flat_products.append(product_group)
    else:
        flat_products = products
    
    # Get ULD products
    uld_products = [p for p in flat_products if p.get('PieceType') == 'ULD']
    
    # Create lookup table for containers by ULDCategory for faster matching
    container_by_category = {}
    for c in containers:
        if c['ULDCategory'] not in container_by_category:
            container_by_category[c['ULDCategory']] = []
        container_by_category[c['ULDCategory']].append(c)
    
    # Process ULD products
    for product in uld_products:
        uld_category = product['ULDCategory']
        if uld_category in container_by_category and container_by_category[uld_category]:
            matching_container = container_by_category[uld_category][0]
            container_by_category[uld_category].remove(matching_container)  # Remove from lookup
            
            print(f"Product {product['id']} (ULDCategory: {uld_category}) blocks container {matching_container['id']}.")
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
    
    # Return a flattened version of products for further processing
    return flat_products, containers, blocked_containers, blocked_for_ULD, placed_ulds, placed_products

def process_optimized(products, containers, blocked_containers, DC_total_volumes):
    """
    Optimized main process function using improved algorithms.
    """
    start_time = time.time()
    containers_tp = containers[:]
    blocked_for_ULD = []
    placed = []
    placed_products = []
    unplaced = []
    placed_ulds = []
    
    # First pass: Process products
    # Preprocess containers and products to block ULD-related containers
    products, containers_tp, blocked_containers, blocked_for_ULD, placed_ulds, placed_products = preprocess_containers_and_products_optimized(
        products, containers_tp, blocked_for_ULD, placed_ulds, placed_products
    )
    
    # Place products sequentially
    placed_products, remaining_products, blocked_containers, containers_tp = pack_products_sequentially_optimized(
        containers_tp, products, blocked_containers, DC_total_volumes, placed_products
    )
    
    # Copy placed products to main lists - use more efficient operations
    placed_ids = {p['id'] for p in placed}  # Use set for O(1) lookups
    for item in placed_products:
        if item['id'] not in placed_ids:
            placed.append(item)
            placed_ids.add(item['id'])
    
    # Only append items to unplaced list if their ID doesn't already exist
    unplaced_ids = set()
    for item in remaining_products:
        if item['id'] not in placed_ids and item['id'] not in unplaced_ids:
            unplaced.append(item)
            unplaced_ids.add(item['id'])
    
    #print(f"\n\nFirst Round Done\nplaced products: {len(placed_products)}\nunplaced products: {len(remaining_products)}")
    
    # Second pass: Try to place remaining products
    print("\nAttempting to place unplaced products in remaining spaces...")
    
    # Sort by volume for better packing
    placed = sorted(placed, key=lambda x: x['Volume'])
    used_container = []
    
    # Filter containers more efficiently using set operations
    blocked_ids = {c['id'] for c in blocked_for_ULD}
    containers_tp = [c for c in containers if c['id'] not in blocked_ids]
    
    # Filter out containers that are already nearly full - do this once rather than in each loop
    containers_to_remove = []
    container_products = {}  # Cache product lists by container
    for container in containers_tp:
        placed_products_in_container = [item for item in placed if item["container"] == container["id"]]
        container_products[container['id']] = placed_products_in_container
        total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
        if total_volume_of_placed_products > 0.8*container['Volume']:
            containers_to_remove.append(container)
    
    # Remove containers in a separate step to avoid modifying while iterating
    for container in containers_to_remove:
        containers_tp.remove(container)
    
    if containers_tp:
        # Sort unplaced products by volume (smallest first for this pass)
        unplaced = sorted(unplaced, key=lambda x: x["Volume"])
        missed_products_count = 0
        
        for container in containers_tp:
            placed_products_in_container = container_products.get(container['id'], [])
            total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
            container_volume = container['Volume']
            occupied_volume = total_volume_of_placed_products
            
            i = 0
            while i < len(unplaced) and missed_products_count < 3:
                product = unplaced[i]
                placed_ = try_place_product_optimized(product, container, placed_products_in_container, occupied_volume, placed)
                
                if placed_:
                    occupied_volume += product['Volume']
                    unplaced.pop(i)  # Remove without shifting index
                    if container not in used_container:
                        used_container.append(container)
                else:
                    print(f"Product {product['id']} could not be placed in container {container['id']}.")
                    missed_products_count += 1
                    i += 1  # Move to next product
                    
                # Reset counter and flip list if too many misses
                if missed_products_count >= 3:
                    print("\nSwitching list around\n")
                    unplaced = unplaced[::-1]
                    missed_products_count = 0
                    i = 0  # Start from beginning of reversed list
            
            if not unplaced:
                print("All products have been placed.")
                break
    
    end_time = time.time()
    print(f"Optimization process completed in {end_time - start_time:.2f} seconds")
    return placed, unplaced, blocked_for_ULD, placed_ulds

# Keep original process function for compatibility
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
        for item in placed_products:
            if not any(p['id'] == item['id'] for p in placed):
                placed.append(item)
                
        # Only append items to unplaced list if their ID doesn't already exist
        for item in remaining_products:
            if not any(p['id'] == item['id'] for p in unplaced):
                unplaced.append(item)
        
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

# Reporting and visualization functions
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

# Visualization function implementation retained from original code
# Detailed implementation of visualize_separate_containers_with_plotly is included
# in the original file and should be kept as is since it's not a performance bottleneck



def visualize_separate_containers_with_plotly(containers, placed_products, blocked_for_ULD):
    colors = ['#A97835', '#C08F4F', '#CD9F61', '#C2A574', '#D6B88A','#E6D3B3','#CBB994']

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


##################################################################################################################################



def try_place_product_gravity_based(product, container, container_placed, occupied_volume, placed_products):
    """
    Modified placement algorithm that places products on the floor or on top of other products.
    Prioritizes bottom-up placement to avoid "floating" products.
    """
    # Start time for this product placement attempt
    start_time = time.time()
    max_time = 2  # Maximum time to spend on a single product placement (seconds)
    
    # Get container dimensions once
    c_length = math.floor(container['Length'])
    c_width = math.floor(container['Width'])
    c_height = math.floor(container['Height'])
    
    # Try all orientations
    for l, w, h in get_orientations(product):
        # Skip impossible orientations immediately
        if l > c_length or w > c_width or h > c_height:
            continue
        
        # Create a height map of the container
        # This represents the highest point at each (x,y) coordinate
        height_map = np.zeros((c_length + 1, c_width + 1))
        
        # Initialize height map with container floor (0)
        # Account for irregular container shapes in the height map initialization
        if container['SD'] == 'S':
            if container['TB'] == 'B':
                # Handle single slope at bottom
                for x in range(c_length + 1):
                    for y in range(c_width + 1):
                        if y > container['Widthx']:
                            # Area with reduced height
                            height_map[x, y] = container['Height'] - container['Heightx']
            elif container['TB'] == 'T':
                # Handle single slope at top (no change to initial height map)
                pass
        elif container['SD'] == 'D':
            if container['TB'] == 'B':
                # Handle double slope at bottom
                width_small = (container['Width'] - container['Widthx']) / 2
                for x in range(c_length + 1):
                    for y in range(c_length + 1):
                        if y < width_small or y > container['Width'] - width_small:
                            # Areas with reduced height
                            height_map[x, y] = container['Height'] - container['Heightx']
            elif container['TB'] == 'T':
                # Handle double slope at top (no change to initial height map)
                pass
        
        # Update height map with already placed products
        for p in container_placed:
            px, py, pz, pl, pw, ph = p['position']
            for dx in range(math.floor(pl)):
                for dy in range(math.floor(pw)):
                    if 0 <= px + dx < c_length + 1 and 0 <= py + dy < c_width + 1:
                        height_map[px + dx, py + dy] = max(height_map[px + dx, py + dy], pz + ph)
        
        # Grid search for valid placement positions - prioritize by height
        valid_positions = []
        
        # Step size for grid search - can be increased for larger containers to improve performance
        step = 1  
        
        for x in range(0, c_length - math.floor(l) + 1, step):
            # Check timeout to avoid spending too much time on hard-to-place products
            if time.time() - start_time > max_time:
                print(f"Timeout for product {product['id']}")
                break
                
            for y in range(0, c_width - math.floor(w) + 1, step):
                # Find the maximum height in the region where the product would be placed
                # This ensures the product is supported from below
                base_z = 0
                for dx in range(math.floor(l)):
                    for dy in range(math.floor(w)):
                        if 0 <= x + dx < c_length + 1 and 0 <= y + dy < c_width + 1:
                            base_z = max(base_z, height_map[x + dx, y + dy])
                
                # If product would exceed container height, skip this position
                if base_z + h > c_height:
                    continue
                
                # Check if product fits at this position
                if fits_optimized(container, container_placed, x, y, base_z, l, w, h):
                    # Store valid position along with base height for sorting
                    valid_positions.append((x, y, base_z, l, w, h))
        
        # Sort positions by base_z (ascending) to prioritize lower positions
        # This ensures we fill from bottom to top
        valid_positions.sort(key=lambda pos: pos[2])
        
        # Use the best position if any are found
        if valid_positions:
            x, y, z, l, w, h = valid_positions[0]  # Take the lowest valid position
            
            # Place the product
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

def pack_with_target_utilization(containers, products, blocked_containers, DC_total_volumes, placed_products, target_utilization=0.8):
    """
    Pack products with a target utilization percentage for each container.
    Uses gravity-based placement to avoid floating products.
    """
    remaining_products = products[:]
    used_containers = []
    missed_product_count = 0
    running_volume_sum = 0
    
    if products:
        destination_code = products[0]['DestinationCode']
        print(f"\nProcessing Destination Code: {destination_code}")
    else:
        destination_code = 'ULDs'
    
    # Pre-compute container volumes for faster access
    container_volumes = {container['id']: container['Volume'] for container in containers}
    
    for container in containers:
        print(f"Placing products in container {container['id']} ({container['ULDCategory']})")
        container_placed = []  # Products placed in the current container
        container_volume = container['Volume']
        occupied_volume = 0
        target_volume = container_volume * target_utilization

        # Process each remaining product - more efficient with index-based iteration
        i = 0
        while i < len(remaining_products):
            product = remaining_products[i]
            
            # Skip products that have already been placed
            if product in placed_products:
                i += 1
                continue
                
            dc_volume = DC_total_volumes.get(product['DestinationCode'], 0)
            
            # Check volume constraints
            if not (dc_volume - running_volume_sum) > 0.99 * container_volume:
                print("Volume constraint not satisfied, stopping process.")
                blocked_containers.extend(used_containers)
                remaining_containers = [c for c in containers if c not in blocked_containers]
                running_volume_sum = 0
                return placed_products, remaining_products, blocked_containers, remaining_containers
            
            # Stop if target utilization reached
            if occupied_volume >= target_volume:
                print(f"Target utilization of {target_utilization*100}% reached for container {container['id']}")
                if container not in used_containers:
                    used_containers.append(container)
                break
                
            if missed_product_count >= 3:
                print("Too many missed products. Blocking containers.")
                blocked_containers.extend(used_containers)
                break

            # Use gravity-based placement instead of original algorithm
            placed = try_place_product_gravity_based(product, container, container_placed, occupied_volume, placed_products)
            
            if placed:
                running_volume_sum += product['Volume']
                remaining_products.pop(i)  # More efficient than remove() for large lists
                if container not in used_containers:
                    used_containers.append(container)
            else:
                print(f"Product {product['id']} could not be placed in container {container['id']}.")
                missed_product_count += 1
                i += 1
                
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

def process_gravity_based(products, containers, blocked_containers, DC_total_volumes, target_utilization=0.8):
    """
    Main process function using gravity-based placement.
    Prioritizes ULDs, then destination-based grouping for first pass, and ignores destinations in second pass.
    
    Args:
        products: List or nested list of products to place
        containers: List of containers available for placement
        blocked_containers: List of containers that are already blocked
        DC_total_volumes: Dictionary of total volumes per destination code
        target_utilization: Target container fill percentage (0.0-1.0)
        
    Returns:
        tuple: (placed, unplaced, blocked_for_ULD, placed_ulds)
    """
    start_time = time.time()
    containers_tp = containers[:]  # Make a copy of containers
    blocked_for_ULD = []
    placed = []
    placed_products = []
    unplaced = []
    placed_ulds = []
    
    # ---------- FIRST PASS: BLOCK ULD CONTAINERS ----------
    # Preprocess containers and products to block ULD-related containers
    products, containers_tp, blocked_containers, blocked_for_ULD, placed_ulds, placed_products = preprocess_containers_and_products_optimized(
        products, containers_tp, blocked_for_ULD, placed_ulds, placed_products
    )
    
    # ---------- FIRST PASS: PLACE PRODUCTS BY DESTINATION ----------
    # Place products using gravity-based approach with target utilization
    # This pass respects destination grouping
    placed_products, remaining_products, blocked_containers, containers_tp = pack_with_target_utilization(
        containers_tp, products, blocked_containers, DC_total_volumes, placed_products, target_utilization
    )
    
    # Copy placed products to main lists - use more efficient operations
    placed_ids = {p['id'] for p in placed}  # Use set for O(1) lookups
    for item in placed_products:
        if item['id'] not in placed_ids:
            placed.append(item)
            placed_ids.add(item['id'])
    
    # Only append items to unplaced list if their ID doesn't already exist
    unplaced_ids = set()
    for item in remaining_products:
        if item['id'] not in placed_ids and item['id'] not in unplaced_ids:
            unplaced.append(item)
            unplaced_ids.add(item['id'])
    
    # ---------- SECOND PASS: PLACE REMAINING PRODUCTS ----------
    print("\nAttempting to place unplaced products in remaining spaces...")
    
    # Sort by volume for better packing
    placed = sorted(placed, key=lambda x: x['Volume'])
    used_container = []
    
    # Filter containers more efficiently using set operations
    blocked_ids = {c['id'] for c in blocked_for_ULD}
    containers_tp = [c for c in containers if c['id'] not in blocked_ids]
    
    # Filter out containers that are already near target utilization
    containers_to_remove = []
    container_products = {}  # Cache product lists by container
    for container in containers_tp:
        placed_products_in_container = [item for item in placed if item["container"] == container["id"]]
        container_products[container['id']] = placed_products_in_container
        total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
        if total_volume_of_placed_products > target_utilization * container['Volume']:
            containers_to_remove.append(container)
    
    # Remove containers in a separate step to avoid modifying while iterating
    for container in containers_to_remove:
        if container in containers_tp:  # Check before removing
            containers_tp.remove(container)
    
    if containers_tp:
        # Sort unplaced products by volume (largest first for second pass)
        # This ensures larger products get priority for placement
        unplaced = sorted(unplaced, key=lambda x: x["Volume"], reverse=True)
        
        # Prioritize containers for second pass
        containers_tp = prioritize_containers_for_second_pass(containers_tp, placed, target_utilization)
        
        print(f"\nTrying to place {len(unplaced)} unplaced products in any container with available space")
        
        # Process each container in priority order
        for container in containers_tp:
            placed_products_in_container = container_products.get(container['id'], [])
            total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
            container_volume = container['Volume']
            occupied_volume = total_volume_of_placed_products
            available_volume = (target_utilization * container_volume) - occupied_volume
            
            # Skip containers that are already at or beyond target utilization
            if occupied_volume >= target_utilization * container_volume:
                print(f"Skipping container {container['id']} - Already at target utilization: {(occupied_volume/container_volume)*100:.1f}%")
                continue
                
            print(f"Trying container {container['id']} - Current utilization: {(occupied_volume/container_volume)*100:.1f}%, Available: {available_volume:.2f} cubic units")
            
            # Process products by volume (largest first)
            missed_products_count = 0
            i = 0
            
            # NEW: Counter to prevent infinite loop
            max_retry_attempts = 5
            retry_count = 0
            product_positions_tried = set()  # Track which products were already tried
            
            while i < len(unplaced) and missed_products_count < 3:
                product = unplaced[i]
                
                # Skip product if it would exceed target utilization
                if product['Volume'] > available_volume:
                    print(f"Product {product['id']} too large for remaining space in container {container['id']}.")
                    i += 1
                    continue
                
                # Check if we've already tried this product in this position
                product_position_key = f"{product['id']}_{i}"
                if product_position_key in product_positions_tried:
                    # We've already tried this product at this position
                    i += 1
                    continue
                    
                product_positions_tried.add(product_position_key)
                
                placed_ = try_place_product_gravity_based(product, container, placed_products_in_container, occupied_volume, placed)
                
                if placed_:
                    occupied_volume += product['Volume']
                    available_volume -= product['Volume']
                    
                    # Update remaining volume percentage for logging
                    remaining_volume_percentage = ((container_volume - occupied_volume) * 100 / container_volume)
                    print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume: {remaining_volume_percentage:.2f}%")
                    
                    unplaced.pop(i)  # Remove without shifting index
                    if container not in used_container:
                        used_container.append(container)
                        
                    # Reset tracking sets when products are successfully placed
                    product_positions_tried = set()
                    
                    # If available volume is now too small, move to next container
                    if available_volume < 0.01:  # Small threshold for floating-point comparison
                        print(f"Container {container['id']} has reached target utilization.")
                        break
                else:
                    print(f"Product {product['id']} could not be placed in container {container['id']}.")
                    missed_products_count += 1
                    i += 1  # Move to next product
                    
                # Reset counter and flip list if too many misses
                if missed_products_count >= 3:
                    retry_count += 1
                    if retry_count >= max_retry_attempts:
                        print(f"\nCannot place remaining products in container {container['id']} after {max_retry_attempts} attempts.")
                        print(f"Moving to next container.\n")
                        break  # Exit the while loop and move to next container
                        
                    print(f"\nRearranging product list after {missed_products_count} misses (Attempt {retry_count}/{max_retry_attempts})\n")
                    unplaced = unplaced[::-1]
                    missed_products_count = 0
                    i = 0  # Start from beginning of reversed list
                    # Clear the tracking set when rearranging
                    product_positions_tried = set()
            
            # Break out early if all products placed
            if not unplaced:
                print("All products have been placed.")
                break
            
            # Log current utilization before moving to next container
            new_utilization = occupied_volume / container_volume
            print(f"Moving to next container. Container {container['id']} current utilization: {new_utilization*100:.1f}%")
    
    end_time = time.time()
    print(f"Optimization process completed in {end_time - start_time:.2f} seconds")
    print(f"Placed products: {len(placed)}, Unplaced products: {len(unplaced)}")
    
    return placed, unplaced, blocked_for_ULD, placed_ulds

def prioritize_containers_for_second_pass(containers, placed_products, target_utilization=0.8):
    """
    Prioritize containers for the second pass:
    1. First prioritize containers that already have products (partially filled)
    2. Then prioritize based on available space and shape complexity
    
    Args:
        containers (list): List of available containers
        placed_products (list): Products already placed
        target_utilization (float): Target utilization percentage
        
    Returns:
        list: Containers sorted by priority for second pass placement
    """
    container_metrics = {}
    
    # First, identify containers that already have products
    containers_with_products = {}
    for product in placed_products:
        container_id = product['container']
        if container_id not in containers_with_products:
            containers_with_products[container_id] = []
        containers_with_products[container_id].append(product)
    
    # Calculate metrics for each container
    for container in containers:
        container_id = container['id']
        
        # Get products already placed in this container
        products_in_container = containers_with_products.get(container_id, [])
        
        # Calculate current volume utilization
        total_volume_placed = sum(p['Volume'] for p in products_in_container)
        container_volume = container['Volume']
        current_utilization = total_volume_placed / container_volume if container_volume > 0 else 1.0
        
        # Calculate available volume (as percentage of target)
        available_volume_percent = max(0, (target_utilization - current_utilization)) / target_utilization
        
        # Calculate absolute available volume
        available_volume = max(0, (target_utilization * container_volume) - total_volume_placed)
        
        # Calculate container "openness" - prefer containers with regular shapes
        # Lower values indicate more regular shapes (easier to pack)
        if container['SD'] == 'S' and container['TB'] == 'B':
            shape_complexity = 1
        elif container['SD'] == 'D':
            shape_complexity = 2
        else:
            shape_complexity = 0
            
        # Primary score based on whether container already has products
        has_products = 1 if container_id in containers_with_products else 0
        
        # Secondary score based on available space and shape complexity
        secondary_score = (available_volume_percent * 0.9) + ((3 - shape_complexity) / 3 * 0.1)
        
        # Combined score prioritizes containers with products first, then by space and shape
        # Use high multiplier (100) to ensure containers with products always come first
        score = (has_products * 100) + secondary_score
        
        container_metrics[container_id] = {
            'container': container,
            'has_products': has_products,
            'current_utilization': current_utilization,
            'available_volume_percent': available_volume_percent,
            'available_volume': available_volume,
            'score': score
        }
    
    # Sort containers by composite score (descending)
    sorted_containers = sorted(
        containers, 
        key=lambda c: container_metrics.get(c['id'], {'score': 0})['score'], 
        reverse=True
    )
    
    # Print container prioritization for debugging
    print("\nContainer prioritization for second pass:")
    for c in sorted_containers[:5]:  # Show top 5
        metrics = container_metrics[c['id']]
        status = "Has products" if metrics['has_products'] == 1 else "Empty"
        print(f"Container {c['id']} - {status}, " + 
              f"Utilization: {metrics['current_utilization']*100:.1f}%, " +
              f"Available volume: {metrics['available_volume']:.2f}, " +
              f"Score: {metrics['score']:.3f}")
    
    return sorted_containers

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
        for y in range(0, math.floor(container['Width'] - w)):
            for x in range(0, math.floor(container['Length'] - l)):
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