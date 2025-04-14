import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Load data
warehouses_df = pd.read_csv("D:/GAIP SG/PROJECT/Warehouses_with_Vendor_Locations.csv")
orders_df = pd.read_csv("D:/GAIP SG/PROJECT/Updated_Orders_with_Unique_Vendors.csv")

# Preprocess warehouse data for lookup
warehouses = warehouses_df.rename(columns={
    "Warehouse": "Warehouse",
    "Assigned Vendor": "Vendor",
    "Volume Capacity (cm³)": "Capacity",
    "State": "State",
    "City": "City"
}).copy()
warehouses["Capacity"] = warehouses["Capacity"].astype(float)

# Assume vendor ratings are included in the orders DataFrame
# If not, you can manually add them for demonstration purposes
if 'Vendor Rating' not in orders_df.columns:
    import numpy as np

    orders_df['Vendor Rating'] = np.random.uniform(1, 5, size=len(orders_df))  # Adding random ratings for demonstration

# Define categories and sub-categories
categories = {
    "Furniture": ["Bookcases", "Chairs", "Tables", "Storage", "Furnishings"],
    "Technology": ["Phones", "Machines", "Copiers"],
    "Office Supplies": ["Labels", "Binders", "Appliances", "Paper", "Accessories", "Envelopes", "Fasteners", "Supplies"]
}


def predict_view(category, sub_category):
    prediction_result = None
    best_time = None

    if category and sub_category:
        model_filename = f"D:/GAIP SG/PROJECT/model_pkl/{category}_{sub_category}_sarima_model.pkl"
        model_path = os.path.join('models', model_filename)  # Assuming models folder exists

        if os.path.exists(model_path):
            with open(model_path, 'rb') as file:
                model_fit = pickle.load(file)

                # Assuming you have a method to get the product data for prediction
                product_data = get_product_data(category, sub_category)  # Replace with your actual data retrieval logic

                if product_data is not None and not product_data.empty:
                    # Forecasting logic
                    forecast_steps = 100
                    forecast = model_fit.forecast(steps=forecast_steps)

                    # Generate forecast dates
                    forecast_dates = pd.date_range(product_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]

                    # Combine forecast dates and values into a DataFrame
                    forecast_df = pd.DataFrame({
                        'Date': forecast_dates,
                        'Forecasted Sales': forecast
                    })

                    # Find the best time to sell (date with the highest forecasted sales)
                    best_time = forecast_df.loc[forecast_df['Forecasted Sales'].idxmax()]

                    # Plot the forecasted values
                    plt.figure(figsize=(10, 6))
                    plt.plot(product_data.index, product_data, label='Historical Data')
                    plt.plot(forecast_dates, forecast, label='Forecast', color='blue')
                    plt.title(f"Forecast for {category} - {sub_category}")
                    plt.xlabel('Date')
                    plt.ylabel('Sales')
                    plt.legend()
                    st.pyplot(plt)

                    # Prepare data to display
                    prediction_result = forecast_df
                    best_time = best_time['Date']

                else:
                    st.write("Product data not found or empty.")
        else:
            st.write("Model file not found.")

    return prediction_result, best_time


def get_product_data(category, sub_category, file_path='D:/GAIP SG/PROJECT/MRP_data.csv'):
    """
    Retrieve and process product sales data based on the specified category and sub-category.

    Parameters:
        category (str): The product category to filter by.
        sub_category (str): The product sub-category to filter by.
        file_path (str): Path to the CSV file containing the data (default: 'D:/GAIP SG/PROJECT/MRP_data.csv').

    Returns:
        pd.Series: Monthly average sales data with 'order_date' as the index.
    """
    # Load the CSV file and filter rows based on category and sub-category
    df = pd.read_csv(file_path, usecols=['category', 'sub_category', 'order_date', 'sales'])
    filtered_data = df.query("category == @category and sub_category == @sub_category")

    # Convert 'order_date' to datetime and aggregate sales data
    filtered_data.loc[:, 'order_date'] = pd.to_datetime(filtered_data['order_date'])
    monthly_sales = (
        filtered_data
        .groupby(pd.Grouper(key='order_date', freq='M'))['sales']
        .mean()
    )

    return monthly_sales


def get_top_5_products(category, sub_category, orders_df):
    filtered_data = orders_df[(orders_df['Category'] == category) &
                              (orders_df['Sub-Category'] == sub_category)]

    # Aggregate sales by product_name
    product_sales = filtered_data.groupby('Product Name')['Sales'].sum().reset_index()

    # Sort by sales in descending order and get top 5
    top_5_products = product_sales.sort_values(by='Sales', ascending=False).head(5)

    return top_5_products


# Main app
st.title("Sales Forecasting Application")

# Session state to manage multi-page workflow
if "page" not in st.session_state:
    st.session_state.page = 1

if st.session_state.page == 1:
    # Dropdown for categories
    category = st.selectbox("Select the category:", list(categories.keys()))

    # Dropdown for sub-categories based on the selected category
    sub_category = st.selectbox("Select the sub-category:", categories[category])

    if st.button("Predict"):
        # Predict and display the results
        prediction_result, best_time = predict_view(category, sub_category)

        if prediction_result is not None:
            st.write("\nPrediction Results:")
            st.dataframe(prediction_result)
            st.write(f"\nBest time to sell: {best_time}")

            # Get and display top 5 products
            top_5_products = get_top_5_products(category, sub_category, orders_df)
            st.write("\nTop 5 Trending Products:")
            st.dataframe(top_5_products)

            st.session_state.page = 2
        else:
            st.write("No prediction results found.")

if st.session_state.page == 2:
    st.header("Select Vendors")
    product_name = st.text_input("Enter the product name:")
    product_matches = orders_df[orders_df["Product Name"].str.contains(product_name, case=False, na=False)]

    if not product_matches.empty:
        st.write("\nMatching Products:")
        st.dataframe(
            product_matches[["Product Name", "Assigned Vendor", "Volume (cm^3)", "Vendor Rating"]].drop_duplicates())

        # Display vendor names along with ratings
        vendor_options = product_matches[["Assigned Vendor", "Vendor Rating"]].drop_duplicates().sort_values(
            by="Vendor Rating", ascending=False)
        vendor_options['Vendor Option'] = vendor_options.apply(
            lambda x: f"{x['Assigned Vendor']} (Rating: {x['Vendor Rating']:.1f})", axis=1)
        selected_vendor_option = st.selectbox("Choose a vendor for the product:", vendor_options['Vendor Option'])

        selected_vendor = \
        vendor_options[vendor_options['Vendor Option'] == selected_vendor_option]['Assigned Vendor'].values[0]
        st.session_state.selected_vendor = selected_vendor
        st.session_state.product_matches = product_matches
    else:
        st.write("No matching product found. Please try again.")

    if st.button("Next"):
        st.session_state.page = 3

if st.session_state.page == 3:
    st.header("Select Warehouse")
    selected_vendor = st.session_state.selected_vendor
    product_matches = st.session_state.product_matches
    vendor_matches = product_matches[product_matches["Assigned Vendor"].str.lower() == selected_vendor.lower()]
    volume = vendor_matches["Volume (cm^3)"].iloc[0]

    vendor_warehouses = warehouses[warehouses["Vendor"].str.lower() == selected_vendor.lower()]

    if not vendor_warehouses.empty:
        st.write("\nAvailable Warehouses for the Vendor:")
        st.dataframe(vendor_warehouses[["Warehouse", "City", "State", "Capacity"]])

        selected_city = st.text_input("Enter the city where you want to place the order:")
        city_warehouses = vendor_warehouses[vendor_warehouses["City"].str.contains(selected_city, case=False, na=False)]

        if not city_warehouses.empty:
            st.write("\nWarehouses in the selected city:")
            st.dataframe(city_warehouses[["Warehouse", "Capacity"]])

            selected_warehouse = st.selectbox("Choose a warehouse:", list(city_warehouses["Warehouse"]))
            st.session_state.selected_warehouse = selected_warehouse
            st.session_state.city_warehouses = city_warehouses
        else:
            st.write(f"No warehouses available in {selected_city} for vendor {selected_vendor}. Please try again.")

    if st.button("Place Order"):
        selected_warehouse = st.session_state.selected_warehouse
        warehouse_data = st.session_state.city_warehouses[
            st.session_state.city_warehouses["Warehouse"].str.lower() == selected_warehouse.lower()]

        # Step 5: User specifies the quantity of the product
        quantity = st.number_input("Enter the quantity of the product you want to order:", min_value=1, step=1)
        total_volume = volume * quantity
        available_capacity = warehouse_data["Capacity"].iloc[0]

        if available_capacity < total_volume:
            st.write(
                f"\nInsufficient capacity in the selected warehouse. Available capacity: {available_capacity} cm³. Please choose another warehouse or reduce the quantity.")
        else:
            # Deduct total volume from warehouse capacity
            warehouses.loc[
                warehouses["Warehouse"].str.lower() == selected_warehouse.lower(), "Capacity"] -= total_volume
            st.write(
                f"\nOrder placed successfully! {quantity} units ordered. Remaining capacity in {selected_warehouse}: {available_capacity - total_volume} cm³")

    if st.button("Finish"):
        st.session_state.page = 1
