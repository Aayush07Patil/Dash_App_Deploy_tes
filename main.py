import pandas as pd
import utils
import packing_algorithm
import time
import logging

logger = logging.getLogger(__name__)

def main(pallets, cargo):
    """
    Main function that processes the input data and coordinates the packing algorithm.
    
    Args:
        pallets (DataFrame): DataFrame containing container/palette information
        cargo (DataFrame): DataFrame containing cargo/products information
        
    Returns:
        tuple: (placed_products, containers, blocked_for_ULD, placed_ulds)
    """
    # Convert measurements from centimeters to inches
    pallets = utils.convert_cms_to_inches(pallets, ['Length', 'Width', 'Height', 'Widthx', 'Heightx'])
    cargo = utils.convert_cms_to_inches(cargo, ['Length', 'Breadth', 'Height'])

    # Round dimensions to 2 decimal places
    cargo = utils.round_columns(cargo, ['Length', 'Breadth', 'Height'], n=2)
    pallets = utils.round_columns(pallets, ['Length', 'Width', 'Height', 'Widthx', 'Heightx'], n=2)

    # Organize cargo by destination code
    cargo_list = utils.organize_cargo(cargo)
    utils.write_list_to_text(cargo_list, 'cargo_list.txt')

    # Convert pallets DataFrame to list of dictionaries
    pallets_list = utils.pallets_to_list(pallets)
    utils.write_list_to_text(pallets_list, 'pallets_list.txt')

    # Calculate total volumes for each destination code
    volumes = utils.volumes_list_for_each_destination(cargo_list)

    # Initialize variables for the packing algorithm
    containers = pallets_list
    blocked_containers = []
    
    # Track execution time
    start_time = time.time()
    
    # Process the cargo and containers using the packing algorithm
    placed_products, unplaced_products, blocked_for_ULD, placed_ulds = packing_algorithm.process(
        cargo_list, containers, blocked_containers, volumes)
    
    logger.info("placed_ulds: %s", len(placed_ulds))

    # Sort the results
    placed_products = sorted(placed_products, key=lambda x: x['container'])
    unplaced_products = sorted(unplaced_products, key=lambda x: x['id'])
    
    logger.info("placed_products: %s", len(placed_products))
    logger.info("unplaced_products: %s", len(unplaced_products))

    # Calculate and log execution time
    end_time = time.time()
    time_elapsed = end_time - start_time
    logger.info(f"Time taken for execution: {time_elapsed:.2f} seconds")
    
    return placed_products, containers, blocked_for_ULD, placed_ulds

if __name__ == "__main__":
    # Test with local data if running this file directly
    pallets = pd.read_excel("data/Containers.xlsx")
    cargo = pd.read_excel("data/Products_List.xlsx")
    
    placed_products, containers, blocked_for_ULD, placed_ulds = main(pallets, cargo)
    
    # Create visualization for testing
    from visualization import visualize_all_containers_with_plotly
    visualize_all_containers_with_plotly(containers, placed_products, blocked_for_ULD, placed_ulds)
    
    # Generate summary information
    container_summary = utils.create_container_product_summary(placed_products)
    table_df = utils.create_container_summary_table(container_summary)
    print("\nContainer Summary Table:")
    print(table_df)