# stock_analysis/views.py

from django.shortcuts import render
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import argrelextrema
import io
import urllib, base64

def stock_analysis_view(request):
    if request.method == 'POST' and request.FILES.get('stock_file'):
        stock_file = request.FILES['stock_file']
        timeframe = request.POST.get('timeframe')  # Get selected timeframe
        df = pd.read_csv(stock_file, parse_dates=True, index_col='time')

        # Resample data according to the selected timeframe
        if timeframe == 'W':
            df = df.resample('W').mean()  # Weekly
        elif timeframe == 'M':
            df = df.resample('M').mean()  # Monthly
        elif timeframe == 'Y':
            df = df.resample('Y').mean()  # Yearly
        # 'D' or default case is daily, no resampling needed

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
        
        
        check = ''

        parallel = np.isclose(support_coef, resistance_coef, atol=0.0001)
        check = check.join("1") if parallel else check

        # Slope calculation in degrees
        slope_in_radians = np.arctan(support_coef)
        slope_in_degrees = np.degrees(slope_in_radians)

        
        # Check if the slope is greater than 10 degrees
        well_performing = 'Well Performing' if slope_in_degrees > 3 else 'Not Well Performing'
        check = check.join("1") if slope_in_degrees > 3 else check

        # Check if the latest stock price is within 5% of the support line (Buy signal)
        latest_stock_price = df['close'].iloc[-1]
        latest_support_line_value = support_coef * df.index[-1].toordinal() + support_intercept
        buy_signal = 'Good Time' if latest_stock_price <= 1.05 * latest_support_line_value else 'Bad Time'
        check = check.join("1") if latest_stock_price <= 1.05 * latest_support_line_value else check

        # Calculate distance from the latest stock price to the support regression line
        latest_stock_price = df['close'].iloc[-1]
        latest_time = pd.Timestamp.toordinal(df.index[-1])

        # Using the line equation: y = mx + b
        A = -support_coef  # m
        B = 1  # y coefficient
        C = -support_intercept  # b

        # Distance formula
        distance = abs(A * latest_time + B * latest_stock_price + C) / np.sqrt(A**2 + B**2)

        if check == "111":
            final_check = "BUY"
        else:
            final_check = "DO NOT BUY"
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(df['close'], label='Stock Price', color='black')
        plt.scatter(support.index, support['min'], color='green', label='Support')
        plt.scatter(resistance.index, resistance['max'], color='red', label='Resistance')
        x_values = np.array(df.index.map(pd.Timestamp.toordinal)).reshape(-1, 1)
        plt.plot(df.index, support_coef * x_values + support_intercept, label='Support Line', linestyle='--', color='green')
        plt.plot(df.index, resistance_coef * x_values + resistance_intercept, label='Resistance Line', linestyle='--', color='red')
        plt.title(f"Regression Model")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # Save plot to PNG image in memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        string = base64.b64encode(buf.read())
        uri = urllib.parse.quote(string)
        
        
            

        return render(request, 'stock_analysis/analysis.html', {
            'data': uri,
            'parallel_channel': 'Detected' if parallel else 'Not Detected',
            'buy_signal': buy_signal,
            'well_performing': well_performing,
            'slope_in_degrees': slope_in_degrees,
            'latest_stock_price': latest_stock_price,
            'latest_support_line_value': latest_support_line_value,
            'final_check':final_check,
            'distance':distance
        })

    return render(request, 'stock_analysis/upload.html')
