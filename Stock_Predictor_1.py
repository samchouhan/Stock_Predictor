import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Ask user
TIME = input("Enter time period (5y / 1y / 1m): ")

# Select dataset based on input
if TIME == "5y":
    stock = yf.download("ADANIPOWER.NS", start="2021-01-01", end="2026-04-18")
elif TIME == "1y":
    stock = yf.download("ADANIPOWER.NS", start="2025-01-01", end="2026-04-18")
elif TIME == "1m":
    stock = yf.download("ADANIPOWER.NS", start="2026-03-01", end="2026-04-01")
else:
    print("Invalid input, defaulting to 1 year")
    stock = yf.download("ADANIPOWER.NS", start="2025-01-01", end="2026-04-01")

# Step 2: Keep only closing price
data = stock[['Close']]

# Step 3: Create future prediction column
future_days = 30
data['Prediction'] = data[['Close']].shift(-future_days)

# Step 4: Prepare data
X = np.array(data.drop(['Prediction'], axis=1))[:-future_days]
y = np.array(data['Prediction'])[:-future_days]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 6: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict future prices
x_future = data.drop(['Prediction'], axis=1).tail(future_days)
x_future = np.array(x_future)

predictions = model.predict(x_future)

# Step 8: Plot
plt.figure(figsize=(10,5))
plt.plot(data['Close'], label="Actual Price")
plt.plot(range(len(data), len(data)+future_days), predictions, label="Predicted Price")
plt.legend()
plt.show()

# Step 9: Output
print("Model Accuracy:", model.score(X_test, y_test))
print("Next 30 days prediction:")
print(predictions)
