import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Get stock data
stock = yf.download("ADANIPOWER.NS", start="2020-01-01", end="2024-01-01")

# Step 2: Keep only closing price
data = stock[['Close']]

# Step 3: Create future prediction column
future_days = 30
data['Prediction'] = data[['Close']].shift(-future_days)
