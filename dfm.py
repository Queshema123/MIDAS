import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from load_data import load_data_to_dict

data_dict = load_data_to_dict("data\\filtered_data\\stl")

# Список преобразованных рядов
hf_series = []
# Возможно для индексов стоит брать средний показатель за период, а не суммировать его
for name, meta in data_dict.items():
    df = meta['data'].copy()
    df = df.set_index('Date')
    df = df.resample('QE').sum()
    df = df.rename(columns={"Value": name})
    hf_series.append(df)

# Объединение по дате
df_all = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'), hf_series)

# (необязательно) Заполним пропуски — например, интерполяцией
df_all = df_all.interpolate(method='linear')

df_const = pd.DataFrame({"const": 1}, index=df_all.index)

endog_train = df_all.iloc[:-3]
exog_train  = df_const.iloc[:-3]
exog_future = df_const.loc['2024-03-31':'2024-09-30']

model = sm.tsa.DynamicFactor(endog=endog_train, exog=exog_train, k_factors=2, factor_order=3)
result = model.fit(method="lbfgs", disp=True)

actual = df_all["GDP"]
pred = result.get_prediction(end='2024-09-30', exog=exog_future)  
fitted = pred.predicted_mean["GDP"]

y_true = actual.tail(3)
y_pred = fitted.tail(3)

# 2. Расчет метрик «вручную» через numpy/pandas
errors        = y_pred - y_true
abs_errors    = errors.abs()
rel_errors    = abs_errors / y_true
squared_errors = errors.pow(2)

mae_manual  = abs_errors.mean()
rmse_manual = np.sqrt(squared_errors.mean())
mape_manual = (abs_errors / y_true).mean() * 100

print(f"MAE (manual)  = {mae_manual:.4f}")
print(f"RMSE (manual) = {rmse_manual:.4f}")
print(f"MAPE (manual) = {mape_manual:.2f}%")
print(f"abs = {abs_errors}, rel = {rel_errors}%")

plt.figure(figsize=(10, 5))
plt.grid(True)
plt.plot(actual.index, actual.values, label="ВВП",     marker='o')
plt.plot(fitted.index, fitted.values, label="DFM", marker='o')
plt.title("Результаты")
plt.xlabel("Дата")
plt.ylabel("ВВП")
plt.legend()
plt.grid(True)
plt.show()