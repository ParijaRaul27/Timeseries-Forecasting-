import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load data
data = pd.read_csv('sales_data.csv', parse_dates=['date'], index_col='date')

# Explore data
print(data.head())
print(data.describe())

# Plot the sales data
data['sales'].plot(title='Sales Over Time', figsize=(12, 6))
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(data['sales'], model='multiplicative', period=12)
decomposition.plot()
plt.show()

# Split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:]

# Plot train and test sets
train['sales'].plot(figsize=(12, 6), title='Train and Test Sets')
test['sales'].plot()
plt.legend(['Train', 'Test'])
plt.show()

# Fit model
model = ExponentialSmoothing(train['sales'], seasonal='multiplicative', seasonal_periods=12).fit()

# Make predictions
predictions = model.forecast(len(test))

# Plot predictions
train['sales'].plot(figsize=(12, 6), title='Train, Test, and Predicted Sales')
test['sales'].plot()
predictions.plot()
plt.legend(['Train', 'Test', 'Predictions'])
plt.show()

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test['sales'], predictions))
print(f'Root Mean Squared Error: {rmse}')

# Forecast future demand
future_forecast = model.forecast(12)  # Forecast for the next 12 months

# Plot future forecast
data['sales'].plot(figsize=(12, 6), title='Sales and Future Forecast')
future_forecast.plot()
plt.legend(['Sales', 'Future Forecast'])
plt.show()
