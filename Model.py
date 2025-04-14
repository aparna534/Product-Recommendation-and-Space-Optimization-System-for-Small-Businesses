
MODEL OF EVERY COMBINATION:
  
categories = {
    "Furniture": ["Bookcases", "Chairs", "Tables", "Storage", "Furnishings"],
    "Technology": ["Phones", "Machines", "Copiers"],
    "Office Supplies": ["Labels", "Binders", "Appliances", "Paper", "Accessories", "Envelopes", "Fasteners", "Supplies"]
}

# Function to train SARIMA for each product and forecast next 100 steps
def train_and_forecast_sarima(data, category, product_name):
    # Filter the data for the specific product
    product_data = data.loc[(data['category'] == category) & (data['sub_category'] == product_name)]

    # Check if data is available for the product
    if product_data.empty:
        print(f"No data available for {category} - {product_name}. Skipping...")
        return None

    # Sort the data by date (ensure proper chronological order)
    product_data['order_date'] = pd.to_datetime(product_data['order_date'])
    product_data = product_data.sort_values('order_date')

    # Set the date as the index
    product_data.set_index('order_date', inplace=True)

    # Create the SARIMA model (you can adjust the order and seasonal_order based on your requirements)
    model = sm.tsa.statespace.SARIMAX(product_data['sales'], order=(0,0,0), seasonal_order=(1, 1, 0, 12))
    
    # Fit the model
    model_fit = model.fit(disp=False)

    # Forecast the next 100 steps
    forecast_steps = 100
    forecast = model_fit.forecast(steps=forecast_steps)

    # Plot the forecasted values
    plt.figure(figsize=(10, 6))
    plt.plot(product_data.index, product_data['sales'], label='Historical Data')
    plt.plot(pd.date_range(product_data.index[-1], periods=forecast_steps + 1, freq='M')[1:], forecast, label='Forecast', color='red')
    plt.title(f"Forecast for {category} - {product_name}")
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    # Print the forecasted values
    print(f"\nForecast for the next {forecast_steps} steps for {category} - {product_name}:")
    print(forecast)

# Loop through the categories and products to train SARIMA models and show forecasts
for category, products in categories.items():
    for product_name in products:
        train_and_forecast_sarima(data, category, product_name)
