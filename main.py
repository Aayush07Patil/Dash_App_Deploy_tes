import pandas as pd
import functions as fun
import pprint
import time

def main(palettes, cargo):

    palettes = fun.convert_cms_to_inches(palettes, ['Length', 'Width', 'Height', 'Widthx', 'Heightx'])
    cargo = fun.convert_cms_to_inches(cargo, ['Length', 'Breadth', 'Height'])

    cargo = fun.round_columns(cargo, ['Length', 'Breadth', 'Height'], n=2)
    palettes = fun.round_columns(palettes, ['Length', 'Width', 'Height', 'Widthx', 'Heightx'], n=2)

    cargo_list = fun.organize_cargo(cargo)
    fun.write_list_to_text(cargo_list, 'cargo_list.txt')

    palettes_list = fun.palettes_to_list(palettes)
    fun.write_list_to_text(palettes_list, 'palettes_list.txt')

    volumes = fun.volumes_list_for_each_destination(cargo_list)

    containers = palettes_list
    blocked_containers = []
    start_time = time.time()
    
    # Flatten the cargo_list before processing, or let the function handle it
    placed_products, unplaced_products, blocked_for_ULD, placed_ulds = fun.process_optimized(
        cargo_list, containers, blocked_containers, volumes)
    
    print("placed_ulds")
    print(placed_ulds)

    placed_products = sorted(placed_products, key=lambda x: x['container'])
    unplaced_products = sorted(unplaced_products, key=lambda x: x['id'])
    print("placed_products")
    print(len(placed_products))
    
    print("unplaced_products")
    print(len(unplaced_products))

    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Time taken for execution {time_elapsed}")
    
    return placed_products, containers, blocked_for_ULD, placed_ulds

if __name__ == "__main__":

    palettes = pd.read_excel("data/Containers.xlsx")
    cargo = pd.read_excel("data/Products_List.xlsx")
    placed_products, containers, blocked_for_ULD, placed_ulds= main(palettes,cargo)
    fun.visualize_separate_containers_with_plotly(containers, placed_products,blocked_for_ULD)
    #print(placed_products)
    container_summary=fun.create_container_product_summary(placed_products)
    #print(container_summary)
    table_df = fun.create_container_summary_table(container_summary)
    print("\nContainer Summary Table:")
    print(table_df)