import math
import numpy as np

def get_orientations(product):
    """
    Get all possible orientations of a product.
    
    Args:
        product (dict): Product data with Length, Breadth, and Height
        
    Returns:
        set: Set of tuples representing different orientations (l, w, h)
    """
    # Cache dimensions to avoid repeated dictionary lookups
    l, b, h = product['Length'], product['Breadth'], product['Height']
    # All possible orientations with length, width, and height
    return {(l,b,h), (b,l,h), (l,h,b), (b,h,l), (h,l,b), (h,b,l)}

def fits(container, placed_products, x, y, z, l, w, h):
    """
    Check if a product with dimensions l,w,h fits at position x,y,z in the container
    without overlapping with existing products.
    
    Args:
        container (dict): Container data with shape and dimensions
        placed_products (list): List of products already placed in the container
        x, y, z (float): Coordinates for placement
        l, w, h (float): Product dimensions
        
    Returns:
        bool: True if the product fits, False otherwise
    """
    epsilon = 1e-6
    
    # Check bounds based on container shape (Single/Double, Top/Bottom)
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
    """
    Preprocess containers and products to handle ULD-type products.
    
    Args:
        products (list): List of products
        containers (list): List of available containers
        blocked_for_ULD (list): List of containers blocked for ULD
        placed_ulds (list): List of placed ULDs
        placed_products (list): List of placed products
        
    Returns:
        tuple: Updated lists (products, containers, blocked_containers, blocked_for_ULD, placed_ulds, placed_products)
    """
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