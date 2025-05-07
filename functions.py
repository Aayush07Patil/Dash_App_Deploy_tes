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
        df.loc[mask, col] = df.loc[mask, col] / 2.54  # cm to inch

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
    return set(permutations([l, b, h]))

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
    """
    Original process function to maintain compatibility.
    Consider replacing calls to this with process_optimized.
    """
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
        products, containers_tp, blocked_containers, blocked_for_ULD, placed_ulds, placed_products = preprocess_containers_and_products_optimized(product, containers_tp, blocked_for_ULD, placed_ulds, placed_products)

        # Place products sequentially
        placed_products, remaining_products, blocked_containers, containers_tp = pack_products_sequentially_optimized(
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

    #print(f"\n\nFirst Round Done\nplaced products: {placed_products}\nunplaced products: {remaining_products}")
        
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
                                    if fits_optimized(container, placed_products_in_container, x, y, z, l, w, h):
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
                                        if fits_optimized(container, placed_products_in_container, x, y, z, l, w, h):
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
                                            remaining_volume_percentage = round(((container['Volume'] - occupied_volume) * 100 / container['Volume']), 2)
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