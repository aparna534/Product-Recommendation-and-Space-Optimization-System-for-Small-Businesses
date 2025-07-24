#  Product Recommendation and Space Optimization System for Small Businesses

##  Overview

This project is designed to support small businesses in two major areas:

1.  **Product Recommendation** – Suggesting products likely to perform well based on historical sales data.
2.  **Space Optimization** – Improving shelf and store layout based on item dimensions and layout constraints.

By combining intelligent algorithms and a user-friendly interface, the system helps shopkeepers improve sales and space utilization without needing deep technical knowledge.

---

##  Features

- **Product Recommendation Engine**  
  Uses past sales trends to recommend the most effective product inventory for different store types.

- **Space Optimization Module**  
  Suggests how to arrange products efficiently within the constraints of your physical store layout.

- **User Dashboard**  
  Web-based interface for uploading data, viewing reports, and adjusting parameters.

- **Visual Insights**  
  Displays optimization suggestions and recommendations clearly using charts and tables.

---

##  Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/aparna534/Product-Recommendation-and-Space-Optimization-System-for-Small-Businesses.git
cd Product-Recommendation-and-Space-Optimization-System-for-Small-Businesses

***Set Up Virtual Environment***

Set up a Python virtual environment to isolate dependencies:
```
### Step 2: Set Up Virtual Environment

```bash
python -m venv env
source env/bin/activate
 Windows: env\Scripts\activate

```
### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```
# USAGE


**Prepare Your Data**
**Ensure the following files are placed in the **data/** folder**:

sales_data.csv – Historical sales records (SKU, sales count, category, etc.)

product_dimensions.csv – Dimensions of each product (width, height, depth)

store_layout.json – Store layout metadata (aisle sizes, shelf structure)


**Run the Application**
```bash
python main.py
```


