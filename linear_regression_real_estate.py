# Step 1: Import Modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 2: Read the CSV File
file_path = "real_estate_data.csv"
df = pd.read_csv(file_path)
print(f"df ::: {df}")

# Step 3: Validate CSV Data
print("Dataset Information:")
print(df.info())  # Check for missing values and column types
print("\nMissing Values:\n", df.isnull().sum())  # Show missing values per column

# Handle Missing Values
df['sqft'].fillna(df['sqft'].median(), inplace=True)  # Fill sqft with median
df.dropna(subset=['price'], inplace=True)  # Drop rows where price is missing

# Step 4: Fit in Linear Regression Model
# 🎯 Goal: Predict house price based on square footage using the formula:
#     y = m * x + b
#     where:
#         y = predicted price
#         x = square footage
#         m = slope (rate of price change per sqft)
#         b = intercept (price when sqft = 0)

X = df[['sqft']].values  # Independent Variable (x)
y = df['price'].values   # Dependent Variable (y)

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Internally, model.fit() calculates:
#     m = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)^2)
#     b = ȳ - m * x̄
# Where:
#     x̄ = mean of sqft
#     ȳ = mean of price

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Print Results
print(f"Slope of Regression Line (m): {model.coef_[0]:.2f}")
print(f"Intercept (b): {model.intercept_:.2f}")

# Now the equation becomes:
#     price = {model.coef_[0]:.2f} * sqft + {model.intercept_:.2f}

# Plot Actual vs Predicted Prices
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')  # Real prices
plt.scatter(X_test, y_pred, color='red', label='Predicted Prices', alpha=0.5)  # Predicted prices
plt.plot(X_test, model.predict(X_test), color='green', linewidth=2, label='Regression Line')  # Regression line
plt.xlabel("Square Footage (sqft)")
plt.ylabel("Price ($)")
plt.legend()
plt.title("Linear Regression: Price vs. Square Footage")
plt.savefig("price_vs_sqft_regression.png")
print("Plot saved as 'price_vs_sqft_regression.png'")
