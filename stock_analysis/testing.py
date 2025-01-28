from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
import io
import urllib, base64
import requests
API_KEY = "KS0ODGM3ZS4R9I98"
SYMBOL = "AAPL"

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={API_KEY}"
response = requests.get(url)
data = response.json()
df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')

if "Meta Data" in data:
    # Convert the time series data into a DataFrame
    #time_series_key = "Time Series (Daily)" if time_frame == "TIME_SERIES_DAILY" else (
    #    "Weekly Time Series" if time_frame == "TIME_SERIES_WEEKLY" else "Monthly Time Series")
    time_series_key = "Time Series (Daily)"
    df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index)
    df = df.astype(float)

    # Parameters to adjust sensitivity of extrema detection
    window = 10
    df['min'] = df.iloc[argrelextrema(df['close'].values, np.less_equal, order=window)[0]]['close']
    df['max'] = df.iloc[argrelextrema(df['close'].values, np.greater_equal, order=window)[0]]['close']

    # Drop NaN values
    support = df.dropna(subset=['min'])
    resistance = df.dropna(subset=['max'])

    # Fit linear regression lines
    def fit_linear_regression(x, y):
        x = np.array(x).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        return model.coef_[0], model.intercept_

    support_coef, support_intercept = fit_linear_regression(support.index.map(pd.Timestamp.toordinal), support['min'])
    resistance_coef, resistance_intercept = fit_linear_regression(resistance.index.map(pd.Timestamp.toordinal), resistance['max'])
    parallel = np.isclose(support_coef, resistance_coef, atol=0.0001)
