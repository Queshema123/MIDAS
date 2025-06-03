import statsmodels.api as sm
import pandas as pd
from functools import reduce
from load_data import load_data_to_dict

data_dict = load_data_to_dict("data\\filtered_data\\stl")

# Список преобразованных рядов
hf_series = []
# Возможно для индексов стоит брать средний показатель за период, а не суммировать его
for name, meta in data_dict.items():
    '''
    if meta['frequency'] == 'quarterly':
        continue
    '''

    df = meta['data'].copy()
    df = df.set_index('Date')
    df = df.resample('QE').sum()
    df = df.rename(columns={"Value": name})
    hf_series.append(df)

# Объединение по дате
df_hf = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'), hf_series)

# (необязательно) Заполним пропуски — например, интерполяцией
df_hf = df_hf.interpolate(method='linear')

df_const = pd.DataFrame({"const": 1}, index=df_hf.index)
model = sm.tsa.DynamicFactor(endog=df_hf, exog=df_const, k_factors=2, factor_order=3)

result_ex = model.fit(method="lbfgs", maxiter=200, disp=True)

# Проверка: intercept (“gamma.GDP”) должен быть ≈ среднему GDP
print("Mean GDP history:", df_hf["GDP"].mean())
print("gamma.GDP:", result_ex.params.get("gamma.GDP"))

# --- 4. Строим exog_future на 4 квартала вперёд ---
last_date   = df_hf.index[-1]  # например, 2024-09-30
future_index = pd.date_range(
    start=last_date + pd.offsets.QuarterEnd(), 
    periods=4, 
    freq="QE-DEC"
)
exog_future = pd.DataFrame({"const": 1}, index=future_index)

# --- 5. Прогнозируем с передачей exog_future ---
forecast_ex = result_ex.get_forecast(steps=4, exog=exog_future)

print("\nПрогноз GDP (сырые единицы):")
print(forecast_ex.predicted_mean["GDP"])

print("\n95%-й доверительный интервал для GDP:")
print(forecast_ex.conf_int()[["lower GDP", "upper GDP"]])