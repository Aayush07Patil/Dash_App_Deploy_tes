import math
import logging
from container_utils import get_orientations, fits, preprocess_containers_and_products

logger = logging.getLogger(__name__)

def process(products, containers, blocked_containers, DC_total_volumes):
    """
    Process products and containers to create an optimal packing solution.
    
    Args:
        products (list): List of products to pack
        containers (list): List of available containers
        blocked_containers (list): List of containers that are blocked
        DC_total_volumes (dict): Mapping of destination codes to total volumes
        
    Returns:
        tuple: (placed_products, unplaced_products, blocked_for_ULD, placed_ulds)
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
    
    # Only run second pass if there are unplaced products
    if unplaced:
        print("\nAttempting to place unplaced products in remaining spaces...")
        second_pass_placed = []
        placed = sorted(placed, key=lambda x:x['Volume'])
        used_container = []
        containers_tp = containers[:]
        containers_tp = [item for item in containers_tp if item not in blocked_for_ULD]
        
        # Filter out overloaded containers
        for container in containers_tp[:]:
            placed_products_in_container = [item for item in placed if item["container"] == container["id"]]
            total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
            if total_volume_of_placed_products > 1 * container['Volume']:
                containers_tp.remove(container)
        
        # Try to place unplaced products in the second pass
        if containers_tp:
            unplaced = sorted(unplaced, key=lambda x:x["Volume"])
            second_pass_packing(containers_tp, unplaced, placed, used_container)
    else:
        print("\nAll products have been placed in the first pass. No need for second pass.")
                
    return placed, unplaced, blocked_for_ULD, placed_ulds

def second_pass_packing(containers_tp, unplaced, placed, used_container):
    """
    Second pass packing algorithm to place remaining products in available containers.
    
    Args:
        containers_tp (list): Available containers
        unplaced (list): Products that haven't been placed yet
        placed (list): Products that have already been placed
        used_container (list): Containers that have been used
        
    Returns:
        None: Updates the placed and unplaced lists in-place
    """
    for container in containers_tp:
        placed_products_in_container = [item for item in placed if item["container"] == container["id"]]
        total_volume_of_placed_products = sum(item["Volume"] for item in placed_products_in_container)
        container_volume = container['Volume']
        occupied_volume = total_volume_of_placed_products
        
        # First attempt at placement
        missed_products_count = 0
        for product in unplaced[:]:
            placed_ = False
            if missed_products_count < 3:
                for orientation in get_orientations(product):
                    l, w, h = orientation
                    
                    for x in range(0, math.floor(container['Length'] - l)):
                        for y in range(0, math.floor(container['Width'] - w)):
                            for z in range(0, math.floor(container['Height'] - h)):
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
                                    remaining_volume_percentage = round(((container['Volume'] - occupied_volume) * 100 / container['Volume']), 2)
                                    print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume in container: {remaining_volume_percentage}%")
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
        
        # Second attempt with reversed list if first attempt reached the missed limit
        if missed_products_count >= 3:
            print("\nSwitching list around\n")
            unplaced.reverse()
            missed_products_count = 0 
            
            for product in unplaced[:]:
                placed_ = False
                if missed_products_count < 3:
                    for orientation in get_orientations(product):
                        l, w, h = orientation
                        
                        for x in range(0, math.floor(container['Length'] - l)):
                            for y in range(0, math.floor(container['Width'] - w)):
                                for z in range(0, math.floor(container['Height'] - h)):
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
                                        remaining_volume_percentage = round(((container_volume - occupied_volume) * 100 / container_volume), 2)
                                        print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume in container: {remaining_volume_percentage}%")
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
                        unplaced.reverse()  # Switch back for the next container
                else:
                    break
        
        if not unplaced:
            print("All products have been placed.")
            break

def pack_products_sequentially(containers, products, blocked_containers, DC_total_volumes, placed_products):
    """
    Pack products into containers sequentially based on volume constraints and dimensions.

    Args:
        containers (list): List of available containers with their dimensions and volumes.
        products (list): List of products to be placed with their dimensions and volumes.
        blocked_containers (list): List of containers that are blocked.
        DC_total_volumes (dict): Mapping of destination codes to total allowable volumes.
        placed_products (list): List of products that have already been placed.

    Returns:
        tuple: (placed_products, remaining_products, blocked_containers, available_containers)
    """
    remaining_products = products[:]  # Ensure we work with a copy of the products list
    used_containers = []
    running_volume_sum = 0

    if products:
        destination_code = products[0]['DestinationCode']
        print(f"\nProcessing Destination Code: {destination_code}")
    else:
        destination_code = 'ULDs'

    container_index = 0
    missed_product_count = 0
    first_reversal_done = False
    
    while container_index < len(containers) and remaining_products:
        container = containers[container_index]
        print(f"Placing products in container {container['id']} ({container['ULDCategory']})")
        container_placed = []  # Products placed in the current container
        container_volume = container['Volume']
        occupied_volume = 0
        
        # Process current container until we decide to move on
        product_index = 0
        while product_index < len(remaining_products) and remaining_products:
            # Get current product
            product = remaining_products[product_index]
            
            # Skip products that have already been placed
            if product in placed_products:
                product_index += 1
                continue
            
            # Try to place the product
            placed, new_occupied_volume = try_place_product(product, container, container_placed, 
                                                         occupied_volume, placed_products)
            
            if placed:
                print(f"Product {product['id']} placed in container {container['id']}")
                running_volume_sum += product['Volume']
                occupied_volume = new_occupied_volume
                remaining_products.remove(product)  # Remove product so product_index still points to next item
                if container not in used_containers:
                    used_containers.append(container)
                missed_product_count = 0  # Reset counter on successful placement
            else:
                print(f"Product {product['id']} could not be placed in container {container['id']}.")
                missed_product_count += 1
                product_index += 1  # Move to next product
            
            # Check if we've hit the missed threshold
            if missed_product_count >= 3:
                if not first_reversal_done:
                    # First time hitting 3 misses - reverse the list and try again
                    print("First 3 misses reached. Reversing product list for retry.")
                    remaining_products = remaining_products[::-1]
                    missed_product_count = 0
                    product_index = 0  # Start from beginning of reversed list
                    first_reversal_done = True
                else:
                    # Second time hitting 3 misses - block container and move on
                    print("Second 3 misses reached. Blocking container and moving to next.")
                    if container not in blocked_containers:
                        blocked_containers.append(container)
                    remaining_products = remaining_products[::-1]  # Reverse back to original order
                    missed_product_count = 0
                    first_reversal_done = False  # Reset for next container
                    container_index += 1  # Move to next container
                    break  # Break out of product processing loop
        
        # If we've processed all products or none fit, move to next container
        if product_index >= len(remaining_products):
            if first_reversal_done:
                # We've already tried with reversed list, move to next container
                container_index += 1
                first_reversal_done = False
            else:
                # We haven't tried reversed list yet, let's try that
                print("Processed all products. Reversing list for retry.")
                remaining_products = remaining_products[::-1]
                missed_product_count = 0
                first_reversal_done = True
        
        # If all products are placed, we're done
        if not remaining_products:
            print(f"All products have been placed for {destination_code}")
            running_volume_sum = 0
            blocked_containers.extend([c for c in used_containers if c not in blocked_containers])
            break

    # Add used containers to blocked_containers if not already there
    for container in used_containers:
        if container not in blocked_containers:
            blocked_containers.append(container)
    
    remaining_containers = [c for c in containers if c not in blocked_containers]
    return placed_products, remaining_products, blocked_containers, remaining_containers

def try_place_product(product, container, container_placed, occupied_volume, placed_products):
    """
    Attempt to place a product in the container.

    Args:
        product (dict): The product to be placed.
        container (dict): The container to place the product in.
        container_placed (list): List of already placed products in the container.
        occupied_volume (float): Current occupied volume of the container.
        placed_products (list): List of all placed products.

    Returns:
        tuple: (bool, float) - (True if placed, updated occupied volume)
    """
    print(f"Before placing product {product['id']}: occupied_volume = {occupied_volume}")
    for orientation in get_orientations(product):
        l, w, h = orientation
        for y in range(0, math.floor(container['Width'] - l)):
            for x in range(0, math.floor(container['Length'] - w)):
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
                        print(f"Before placing product {product['id']}: occupied_volume = {occupied_volume}")
                        print(f"Product {product['id']} placed in container {container['id']}\n Remaining volume: {remaining_volume_percentage}%\n\n")
                        return True, occupied_volume
    return False, occupied_volume