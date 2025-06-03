import pandas as pd
import matplotlib.pyplot as plt

stl_df = pd.read_csv("data\\filtered_data\\stl\\daily\\Курс_Рубля_к_Доллару.csv", parse_dates=["Date"], index_col='Date')
nf_df = pd.read_csv("data\\daily\\Курс_Рубля_к_Доллару.csv", parse_dates=["Date"], index_col='Date')
kalman_df = pd.read_csv("data\\filtered_data\\kalman\\daily\\Курс_Рубля_к_Доллару.csv", parse_dates=["Date"], index_col='Date')
hp_df = pd.read_csv("data\\filtered_data\\hp\\daily\\Курс_Рубля_к_Доллару.csv", parse_dates=["Date"], index_col='Date')

stl_df['Value'] = stl_df['Value'].astype(float)
nf_df['Value'] = nf_df['Value'].astype(float)
kalman_df['Value'] = kalman_df['Value'].astype(float)
hp_df['Value'] = hp_df['Value'].astype(float)

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(nf_df['Value'], label='Исходные данные')
ax[0, 0].plot(stl_df['Value'], label='STL разложение')
ax[0, 0].legend()
ax[0, 0].grid(True)

ax[0, 1].plot(nf_df['Value'], label='Исходные данные')
ax[0, 1].plot(kalman_df['Value'], label='Фильтр Калмана')
ax[0, 1].legend()
ax[0, 1].grid(True)

ax[1, 0].plot(nf_df['Value'], label='Исходные данные')
ax[1, 0].plot(hp_df['Value'], label='Фильтр Ходрика-Прескотта')
ax[1, 0].legend()
ax[1, 0].grid(True)

fig.delaxes(ax[1, 1])
input()