import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.ar_model import AutoReg

# Загрузка
df = pd.read_csv("data\\low_freq_data\\GDP.csv", parse_dates=["Date"], index_col="Date")
ts = df["Value"].astype(float)

adf_stat, adf_p, _, _, adf_cr, _ = adfuller(ts, autolag="AIC")
print(f"ADF: stat={adf_stat:.4f}, p={adf_p:.4f}")
for k, v in adf_cr.items():
    print(f"  crit[{k}]={v:.4f}")

kpss_stat, kpss_p, kpss_lags, kpss_cr = kpss(ts, regression="c", nlags="auto")
print(f"KPSS: stat={kpss_stat:.4f}, p={kpss_p:.4f}")
for k, v in kpss_cr.items():
    print(f"  crit[{k}]={v:.4f}")

# Подгонка AR(1) и остатки
model = AutoReg(ts, lags=1, trend="c").fit()
resid = model.resid

# Одна фигура — три оси
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

# 1) ACF
plot_acf(ts, lags=40, alpha=0.05, ax=axes[0])
axes[0].set_title("ACF")

# 2) Исходный ряд
axes[1].plot(ts.index, ts.values)
axes[1].set_title("Original Series")

# 3) Остатки AR(1)
axes[2].plot(resid.index, resid.values)
axes[2].axhline(0, ls="--", lw=1)
axes[2].set_title("AR(1) Residuals")

plt.tight_layout()
plt.show(block=True)