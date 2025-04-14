# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'MRP_data.csv'  # Update with the correct file path in Colab
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Info:")
print(data.info())

print("\nFirst Few Rows of the Dataset:")
print(data.head())

# Categorical Analysis
categorical_columns = ['Category', 'Sub-Category', 'Ship Mode', 'Segment', 'Region']

# Bar Plots for Categorical Columns
for col in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data, x=col, palette='viridis', order=data[col].value_counts().index)
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

 Correlation Analysis
numerical_columns = ['Sales', 'Quantity', 'Discount', 'Profit']
correlation_matrix = data[numerical_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14)
plt.show()

# Distribution of Numerical Features
numerical_columns = ['Sales', 'Quantity', 'Discount', 'Profit']

for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

# Boxplot for Numerical Features
for col in numerical_columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data, x=col, color='skyblue')
    plt.title(f'Boxplot of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.show()

# Scatter Plot: Relationships Between Features
scatter_pairs = [('Sales', 'Profit'), ('Discount', 'Profit'), ('Quantity', 'Sales')]

for x, y in scatter_pairs:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=data, x=x, y=y, hue='Category', palette='viridis')
    plt.title(f'Relationship Between {x} and {y}', fontsize=14)
    plt.xlabel(x, fontsize=12)
    plt.ylabel(y, fontsize=12)
    plt.legend(title='Category', loc='best')
    plt.show()
