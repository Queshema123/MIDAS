import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from pykalman import KalmanFilter
from statsmodels.tsa.filters.hp_filter import hpfilter
import os

l_vals = { 'daily' : 129600, 'monthly' : 14400}
f_vals = { 'daily' : 270,    'monthly' : 90   }

freqs_path = ["data\\daily", "data\\monthly"]

for path in freqs_path:
    freq = path[path.rfind('\\') + 1:]
    l = l_vals.get(freq)
    f = f_vals.get(freq)
    for file in os.listdir(path):
        df = pd.read_csv(f"{path}\\{file}", parse_dates=["Date"], index_col='Date')
        df['Value'] = df['Value'].astype(float)
        ts = df['Value']
        
        stl = STL(df['Value'], period = f).fit()
        adf = adfuller(stl.resid)
        kp  = kpss(stl.resid)

        print(f"ADF: {adf[0]}, p: {adf[1]}")
        print(f"KPSS: {kp[0]}, p: {adf[3]}")

        df['Value'] = ts - stl.resid
        df[['Value']].to_csv(f"data\\filtered_data\\stl\\{freq}\\{file}", index=True, index_label='Date')

        kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1], initial_state_mean=df['Value'].iloc[0])
        df['Value'] = kf.filter(ts)[0].flatten()
        df[['Value']].to_csv(f"data\\filtered_data\\kalman\\{freq}\\{file}", index=True, index_label='Date')

        hp = hpfilter(ts, l)
        df['Value'] = hp[1]
        df[['Value']].to_csv(f"data\\filtered_data\\hp\\{freq}\\{file}", index=True, index_label='Date')