import pandas as pd

# Load data
warehouses_df = pd.read_csv("/mnt/data/Warehouses_with_Vendor_Locations.csv")
orders_df = pd.read_csv("/mnt/data/Updated_Orders_with_Unique_Vendors.csv")

# Preprocess warehouse data for lookup
warehouses = warehouses_df.rename(columns={
    "Warehouse": "Warehouse",
    "Assigned Vendor": "Vendor",
    "Volume Capacity (cm³)": "Capacity",
    "State": "State",
    "City": "City"
}).copy()
warehouses["Capacity"] = warehouses["Capacity"].astype(float)

# Main Program
def place_order():
    while True:
        print("\n--- Welcome to the Warehouse Order System ---")

        # Step 1: User enters product name
        product_name = input("Enter the product name (or type 'exit' to quit): ").strip()
        if product_name.lower() == 'exit':
            print("Thank you for using the system! Goodbye!")
            break

        # Step 2: Search for product in orders dataset
        product_matches = orders_df[orders_df["Product Name"].str.contains(product_name, case=False, na=False)]
        if product_matches.empty:
            print("No matching product found. Please try again.")
            continue

        print("\nMatching Products:")
        print(product_matches[["Product Name", "Assigned Vendor", "Volume (cm^3)"]].drop_duplicates().to_string(index=False))

        # Step 3: Select vendor by name
        selected_vendor = input("Enter the vendor name for the product: ").strip()
        vendor_matches = product_matches[product_matches["Assigned Vendor"] == selected_vendor]

        if vendor_matches.empty():
            print("Invalid vendor selection. Please try again.")
            continue

        volume = vendor_matches["Volume (cm^3)"].values[0]

        # Step 4: User chooses a warehouse for the selected vendor, filtered by city
        vendor_warehouses = warehouses[warehouses["Vendor"] == selected_vendor]

        if vendor_warehouses.empty():
            print(f"No warehouses available for vendor {selected_vendor}. Order cannot be placed.")
            continue

        print("\nAvailable Warehouses for the Vendor:")
        print(vendor_warehouses[["Warehouse", "City", "State", "Capacity"]].to_string(index=False))

        selected_city = input("Enter the city where you want to place the order: ").strip()
        city_warehouses = vendor_warehouses[vendor_warehouses["City"].str.contains(selected_city, case=False, na=False)]

        if city_warehouses.empty():
            print(f"No warehouses available in {selected_city} for vendor {selected_vendor}. Please try again.")
            continue

        print("\nWarehouses in the selected city:")
        print(city_warehouses[["Warehouse", "Capacity"]].to_string(index=False))

        selected_warehouse = input("Enter the name of the warehouse to use: ").strip()
        warehouse_data = city_warehouses[city_warehouses["Warehouse"] == selected_warehouse]

        if warehouse_data.empty():
            print("Invalid warehouse selection. Please try again.")
            continue

        # Step 5: User specifies the quantity of the product
        while True:
            try:
                quantity = int(input("Enter the quantity of the product you want to order: ").strip())
                if quantity <= 0:
                    raise ValueError("Quantity must be a positive integer.")
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter a valid number.")

        total_volume = volume * quantity
        available_capacity = warehouse_data["Capacity"].values[0]

        if available_capacity < total_volume:
            print(f"\nInsufficient capacity in the selected warehouse. Available capacity: {available_capacity} cm^3. Please choose another warehouse or reduce the quantity.")
            continue

        # Step 6: Deduct total volume from warehouse capacity
        warehouses.loc[warehouses["Warehouse"] == selected_warehouse, "Capacity"] -= total_volume

        print(f"\nOrder placed successfully! {quantity} units ordered. Remaining capacity in {selected_warehouse}: {available_capacity - total_volume} cm^3")

        # Step 7: Ask user if they want to continue
        next_action = input("\nDo you want to place another order? (yes/no): ").strip().lower()
        if next_action != 'yes':
            print("Thank you for using the system! Goodbye!")
            break

# Run the program
place_order()
