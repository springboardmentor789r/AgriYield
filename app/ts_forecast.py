import numpy as np
from sklearn.linear_model import LinearRegression

# Simple baseline model: repeat last value
def simple_last_value_forecast(series, horizon):
    if not series:
        return [0.0] * horizon
    return [float(series[-1])] * horizon

# Simple AR model
def simple_ar_forecast(series, horizon, lags=3):

    if len(series) <= lags:
        return simple_last_value_forecast(series, horizon)

    X, y = [], []
    for i in range(lags, len(series)):
        X.append(series[i-lags:i])
        y.append(series[i])

    model = LinearRegression()
    model.fit(X, y)

    window = series[-lags:]
    forecast = []

    for _ in range(horizon):
        pred = model.predict([window])[0]
        forecast.append(float(pred))
        window = window[1:] + [pred]

    return forecast
