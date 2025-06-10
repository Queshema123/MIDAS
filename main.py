from midas import MIDAS
from weights import BetaWeights, AlmonExpPolyWeights, AlmonPolynomialWeights, Weights as UMIDAS
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

def calc_errs(df):
    err = df['Прогноз'] - df["Факт"]
    abs_err = err.abs()
    rel_err = abs_err / df['Факт']
    print(f'abs_err={abs_err}, rel = {rel_err}')

def show_forecast_stat(df):
    errors = df['Прогноз'] - df["Факт"]
    abs_errs = errors.abs()
    squared_errs = errors.pow(2)
    mae = abs_errs.mean()
    rmse = np.sqrt(squared_errs.mean())
    mape = (abs_errs / df['Факт']).mean() * 100
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAPE = {mape:.2f}%")

if __name__ == "__main__":
    folder_path = "data\\filtered_data\\stl" 
    model = MIDAS(folder_path, AlmonExpPolyWeights, lambda_l1=0.2, lambda_l2=0.1)
    model.train('2019-03-31', 'GDP', 6)
    #model.window_train('2019-03-31', 'GDP', 1, 4, 1, '2023-12-31') # 6 1 3; 1 4 1
    forecastes = model.forecast('GDP', '2019-03-31', '2024-09-30')

    df = pd.read_csv("data\\low_freq_data\\GDP.csv")
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.rename(columns={'Value':'Факт'})

    forecastes['Date'] = pd.to_datetime(forecastes['Date']).dt.date
    forecastes = forecastes.rename(columns={'Value':'Прогноз'})

    merged_df = pd.merge(df, forecastes, on='Date', how='inner')
    #merged_df = merged_df.tail(3)

    plt.title('Результаты')
    plt.grid(True)
    plt.plot(merged_df['Date'], merged_df['Факт'],    label='ВВП',    marker='o')
    plt.plot(merged_df['Date'], merged_df['Прогноз'], label='MIDAS Exp', marker='o')
    plt.legend()
    plt.show()
    show_forecast_stat(merged_df)
    calc_errs(merged_df)
    input()