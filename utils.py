import pandas as pd
from collections import defaultdict

def convert_cms_to_inches(df, dimension_cols=['Length', 'Breadth', 'Height'], unit_col='MeasureUnit'):
    """
    Converts dimension columns from centimeters to inches for rows where the measurement unit is 'Cms'.
    
    Args:
        df (DataFrame): DataFrame containing product or container data
        dimension_cols (list): List of column names to convert
        unit_col (str): Name of the column containing unit information
        
    Returns:
        DataFrame: The DataFrame with converted measurements
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
    
    Args:
        df (DataFrame): DataFrame containing the columns to round
        columns (list): List of column names to round
        n (int): Number of decimal places
        
    Returns:
        DataFrame: The DataFrame with rounded values
    """
    for col in columns:
        if col in df.columns:
            df[col] = df[col].round(n)
    return df

def organize_cargo(cargo):
    """
    Organizes cargo records into a structured list.
    
    Args:
        cargo (DataFrame): DataFrame containing cargo information
        
    Returns:
        list: Structured list with ULD items first, followed by groups by destination
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
        sorted_group = group.sort_values('Volume', ascending=False)
        grouped_by_destination.append(sorted_group.to_dict(orient='records'))

    # Final structured list
    return [uld_list] + grouped_by_destination

def pallets_to_list(pallets_df):
    """
    Converts a pallets DataFrame into a list of dictionaries sorted by volume.
    
    Args:
        pallets_df (DataFrame): DataFrame containing palette/container information
        
    Returns:
        list: List of dictionaries with palette/container data sorted by volume (descending)
    """
    # First check if 'Volume' column exists
    if 'Volume' not in pallets_df.columns:
        # If not, calculate volume from dimensions
        pallets_df['Volume'] = pallets_df['Length'] * pallets_df['Width'] * pallets_df['Height']
    
    # Sort by volume in descending order (largest first)
    sorted_df = pallets_df.sort_values(by='Volume', ascending=False)
    
    # Convert to list of dictionaries
    return sorted_df.to_dict(orient='records')

def volumes_list_for_each_destination(cargo_list):
    """
    Calculates total volume per group in the structured cargo list.
    
    Args:
        cargo_list (list): Structured list of cargo items
        
    Returns:
        dict: Dictionary mapping destination codes to total volumes
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
    """
    Writes a list to a text file for debugging purposes.
    
    Args:
        list_data (list): List to write to file
        filepath (str): Path to output file
    """
    # Implementation can be added if needed
    pass

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