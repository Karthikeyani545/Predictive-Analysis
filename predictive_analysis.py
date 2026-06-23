import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. DATA LOADING
# ==========================================
try:
    df = pd.read_csv('historical_sales.csv')
    print("--- Data Loaded Successfully! ---")
    print(df.head()) 
except FileNotFoundError:
    print("Error: 'historical_sales.csv' file not found. Please check the folder.")
    exit()

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Features: 'Marketing_Budget_USD' (X), Target: 'Actual_Sales_Units' (y)
X = df[['Marketing_Budget_USD']]
y = df['Actual_Sales_Units']

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================
# 80% Data for Training, 20% Data for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. MODEL TRAINING
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)
print("\n--- Model Training Completed! ---")

# ==========================================
# 5. PREDICTIONS & EVALUATION
# ==========================================
y_pred = model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Model Accuracy (R2 Score): {r2 * 100:.2f}%")

# ==========================================
# 6. FUTURE FORECASTING (Fixed to prevent UserWarning)
# ==========================================
# Creating a DataFrame with the correct column name to avoid feature name warning
future_budgets = pd.DataFrame([[5500], [6000]], columns=['Marketing_Budget_USD'])
future_predictions = model.predict(future_budgets)

print("\n--- Future Trend Forecasting ---")
print(f"Predicted Sales for $5500 Budget: {int(future_predictions[0])} Units")
print(f"Predicted Sales for $6000 Budget: {int(future_predictions[1])} Units")

# ==========================================
# 7. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 6))

# Plotting historical actual data (Blue color)
plt.scatter(X, y, color='blue', label='Historical Actual Data')

# Plotting predictive trend line (Red color)
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Predictive Trend Line')

plt.title('Predictive Analytics: Marketing Budget vs Sales Units')
plt.xlabel('Marketing Budget (USD)')
plt.ylabel('Actual Sales (Units)')
plt.legend()
plt.grid(True)
plt.show()