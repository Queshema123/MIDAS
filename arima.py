import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calc_metrics(y_true, y_pred):
    # 1. Выравниваем по индексам (оставляем только общие метки)
    y_true_aligned, y_pred_aligned = y_true.align(y_pred, join='inner')
    # 2. Отбрасываем любые NaN
    mask = y_true_aligned.notna() & y_pred_aligned.notna()
    y_true_clean = y_true_aligned[mask]
    y_pred_clean = y_pred_aligned[mask]

    abs_err = np.abs(y_true_clean - y_pred_clean)
    rel_err = abs_err / y_true_clean
    print(abs_err)
    print(rel_err)
    
    # 3. Считаем метрики
    mae  = mean_absolute_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mape = (np.abs((y_true_clean - y_pred_clean) / y_true_clean).mean()) * 100

    return mae, rmse, mape

df = pd.read_csv('data\\low_freq_data\\GDP.csv', parse_dates=['Date'], index_col='Date')
df['Value'] = df['Value'].astype(float)
df_full = df.copy()
df = df.loc[pd.Timestamp('2019-03-31'):pd.Timestamp('2023-06-30')]

model = ARIMA(df['Value'], order=(4, 1, 1))
result = model.fit()
print(result.summary())

fitted = result.predict(start=4, end=len(df_full))
series = pd.Series(fitted, index=df_full.index)

sarima = SARIMAX(df['Value'], order=(1,1,1),seasonal_order=(1,1,1,4), enforce_stationarity=False, enforce_invertibility=False)
result2 = sarima.fit(disp=False)

fitted2 = result2.predict(start=1, end=len(df_full))
series2 = pd.Series(fitted2, index=df_full.index)

# y_true — ваш реальный ряд значений
y_true = df_full['Value']

# y_pred для модели с lags=1
y_pred1 = series  # предсказания result.predict(start=1, end=len(df))

# y_pred для модели с lags=4
y_pred2 = series2  # предсказания result4.predict(start=4, end=len(df))

# Рассчитываем
mae1, rmse1, mape1 = calc_metrics(y_true.tail(3), y_pred1.tail(3))
mae2, rmse2, mape2 = calc_metrics(y_true.tail(3), y_pred2.tail(3))

print("ARIMA:")
print(f" MAE  = {mae1:.4f}")
print(f" RMSE = {rmse1:.4f}")
print(f" MAPE = {mape1:.2f}%\n")

print("SARIMA:")
print(f" MAE  = {mae2:.4f}")
print(f" RMSE = {rmse2:.4f}")
print(f" MAPE = {mape2:.2f}%")

plt.grid(True)
plt.title('Результаты')
plt.plot(df.index,      df['Value'],    marker='o', label='ВВП')
plt.plot(series.index,  series.values,  marker='o', label='ARIMA')
plt.plot(series2.index,  series2.values,  marker='o', label='SARIMA')
plt.legend()
plt.show()