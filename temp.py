import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.filters.hp_filter import hpfilter
from pykalman import KalmanFilter
# from statsmodels.tsa.x13 import x13_arima_analysis  # требует установленный X-13 ARIMA-SEATS

# 1) Загрузка ряда (предполагаем квартальные данные GDP)
df = pd.read_csv("data\\low_freq_data\\GDP.csv", parse_dates=["Date"], index_col="Date")
ts = df["Value"].asfreq("Q").dropna()
