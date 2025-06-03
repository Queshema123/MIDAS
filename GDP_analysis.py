import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.seasonal import seasonal_decompose
from prepare_data import clip_with_iqr, clip_with_zscore
# Загрузка
df = pd.read_csv("data\\low_freq_data\\GDP.csv", parse_dates=["Date"], index_col="Date")
df['Value'] = df['Value'].astype(float)
ts = df['Value']

df['diff'] = df['Value'].diff(4).diff()
df['log'] = np.log(df['Value'])
fig, axes = plt.subplots(1, 2, figsize=(8, 8))
axes[0].plot(df['diff'], label='Diff')
axes[0].set_title('Дифференцирование')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df['log'], label='Log')
axes[1].set_title('Логарифмирование')
axes[1].legend()
axes[1].grid(True)

plt.show()

df['iqr'] = clip_with_iqr(df['Value'])
df['z']   = clip_with_zscore(df['Value'])
print('____________________________________________________________________________')
print(df['Value'] - df['iqr'])
print('____________________________________________________________________________')
print(df['Value'] - df['z'])
print('____________________________________________________________________________')

fig, axes = plt.subplots(1, 1, figsize=(8, 8))
axes[0].plot(df['iqr'], label='IQR')
axes[0].plot(df['Value'], label='ВВП')
axes[0].set_title('IQR')
axes[0].legend()

axes[1].plot(df['z'], label='Z-оценка')
axes[1].plot(df['Value'], label='ВВП')
axes[1].set_title('Z-оценка')
axes[1].legend()

plt.show()

'''
adf_stat, adf_p, _, _, adf_cr, _ = adfuller(ts, autolag="AIC")
print(f"ADF: stat={adf_stat:.4f}, p={adf_p:.4f}")
for k, v in adf_cr.items():
    print(f"  crit[{k}]={v:.4f}")

kpss_stat, kpss_p, kpss_lags, kpss_cr = kpss(ts)
print(f"KPSS: stat={kpss_stat:.4f}, p={kpss_p:.4f}")
for k, v in kpss_cr.items():
    print(f"  crit[{k}]={v:.4f}")
'''

kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], initial_state_mean=df['Value'].iloc[0])
stl = STL(df['Value'], period=4).fit()
stl.plot()
plt.show()

print('____________________________________________________________________________')
adf_result = adfuller(stl.resid)
print(f'ADF: {adf_result[0]}, p-value: {adf_result[1]}')
kpss_result = kpss(stl.resid)
print(f'KPSS: {kpss_result[0]}, p-value: {kpss_result[3]}')

print('____________________________________________________________________________')

hp = hpfilter(df['Value'], 1600)
df['STL'] = ts - res.resid
df['HP'] = hp[1]
df['Kalman'] = kf.filter(df['Value'])[0].flatten()

fig, axes = plt.subplots(1, 3, figsize=(8, 8))
axes[0].plot(df['STL'], label='STL')
axes[0].plot(df['Value'], label='ВВП')
axes[0].set_title('STL')
axes[0].legend()

axes[1].plot(df['HP'], label='HP')
axes[1].plot(df['Value'], label='ВВП')
axes[1].set_title('HP')
axes[1].legend()

axes[2].plot(df['Kalman'], label='Kalman')
axes[2].plot(df['Value'], label='ВВП')
axes[2].set_title('Kalman')
axes[2].legend()
plt.show()

'''
plt.plot(df['STL'], label='STL')
plt.plot(df['HP'], label='HP')
plt.plot(df['Kalman'], label='Kalman')
plt.legend()
plt.show()
'''

print('STL')
adf_result = adfuller(df['STL'].dropna())
print(f'ADF: {adf_result[0]}, p-value: {adf_result[1]}')
kpss_result = kpss(df['STL'].dropna())
print(f'KPSS: {kpss_result[0]}, p-value: {kpss_result[1]}')

print('HP')
adf_result = adfuller(df['HP'].dropna())
print(f'ADF: {adf_result[0]}, p-value: {adf_result[1]}')
kpss_result = kpss(df['HP'].dropna())
print(f'KPSS: {kpss_result[0]}, p-value: {kpss_result[1]}')

print('Kalman')
adf_result = adfuller(df['Kalman'].dropna())
print(f'ADF: {adf_result[0]}, p-value: {adf_result[1]}')
kpss_result = kpss(df['Kalman'].dropna())
print(f'KPSS: {kpss_result[0]}, p-value: {kpss_result[1]}')