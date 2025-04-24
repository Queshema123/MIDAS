import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Загрузка ряда
df = pd.read_csv("data\\low_freq_data\\GDP.csv", parse_dates=["Date"], index_col="Date")
ts = df["Value"]

# 1) Ручная проверка
for order in [(1,1,1), (2,1,0), (1,2,1), (2,1,2)]:
    try:
        m = ARIMA(ts, order=order,
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit()
        print(f"order={order} → AIC={m.aic:.2f}")
    except Exception as e:
        print(f"order={order} → error: {e}")

# 2) Grid search по p,d,q (например, p=0..3, d=0..2, q=0..3)
best_aic = float("inf")
best_cfg = None

warnings.filterwarnings("ignore")  # подавим предупреждения старта
for p in range(4):
    for d in range(3):
        for q in range(4):
            order = (p, d, q)
            try:
                m = ARIMA(ts, order=order,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit()
                if m.aic < best_aic:
                    best_aic, best_cfg = m.aic, order
            except:
                continue

print(f"\nЛучший order={best_cfg} с AIC={best_aic:.2f}")
