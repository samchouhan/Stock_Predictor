import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

#Download stock data
ticker = "JISLJALEQS.NS"
TIME = input("Enter time period (5y / 1y / 1m): ").strip().lower()

period_map = {
    "5y": ("2021-01-01", "2026-04-18"),