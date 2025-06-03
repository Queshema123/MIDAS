import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from pykalman import KalmanFilter
from scipy.stats.mstats import winsorize

# Define input and output base paths
INPUT_BASE = "data"
OUTPUT_BASE_HP = "data\\filtered_data\\hp"
OUTPUT_BASE_KF = "data\\filtered_data\\kalman"

# Create necessary output directories
for base in [OUTPUT_BASE_HP, OUTPUT_BASE_KF]:
    for freq in ['daily', 'monthly']:
        path = os.path.join(base, freq)
        os.makedirs(path, exist_ok=True)


def extend_and_ffill(df, freq):
    """
    Extend the DataFrame index to cover the full date range at the given frequency
    and forward-fill missing values.
    """
    if freq == 'daily':
        rng = pd.date_range(df.index.min(), df.index.max(), freq='D')
    elif freq == 'monthly':
        rng = pd.date_range(df.index.min(), df.index.max(), freq='MS')
    else:
        raise ValueError(f"Unknown frequency: {freq}")

    df = df.reindex(rng)
    df['Value'] = df['Value'].ffill()
    return df


def replace_outliers_iqr_interpolate(series: pd.Series) -> pd.Series:
    """
    Вариант 1: помечаем выбросы NaN и заполняем линейной интерполяцией
    (между соседями выходит среднее ровно из двух точек).
    """
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    # заменяем выбросы на NaN
    s = series.mask(series.lt(lower) | series.gt(upper))
    # интерполируем NaN (линейно между соседями)
    return s.interpolate(method='linear', limit_direction='both')


def replace_outliers_iqr_winsorize(series: pd.Series, limits=(0.1, 0.1)) -> pd.Series:
    """
    Вариант 3: винзоризация — обрезка крайних долей.
    По умолчанию обрезает нижние 5% значений до 5-го перцентиля
    и верхние 5% до 95-го.
    """
    # winsorize возвращает masked array, приводим обратно в Series
    arr = winsorize(series, limits=limits)
    return pd.Series(arr, index=series.index)


def clip_with_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return np.clip(data, lower, upper)

def clip_with_zscore(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    clipped = np.clip(data, mean - threshold*std, mean + threshold*std)
    return clipped

def apply_hp_filter(series, freq:str):
    """
    Apply Hodrick-Prescott filter and return the trend (smoothed) component.
    """
    lamb = 0
    if freq == 'monthly':
        lamb = 14400
    elif freq == 'daily':
        lamb = 129600
    else:
        raise ValueError(f"Unknown frequency - {freq}")
    cycle, trend = hpfilter(series, lamb=lamb)
    return trend


def apply_kalman_filter(series):
    """
    Apply a simple Kalman filter to the series and return the smoothed estimates.
    """
    # Initialize a one-dimensional Kalman filter
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
    kf = kf.em(series.values, n_iter=5)
    state_means, _ = kf.filter(series.values)
    return pd.Series(state_means.flatten(), index=series.index)


def process_frequency(freq):
    input_dir = os.path.join(INPUT_BASE, freq)
    file_path = os.listdir(input_dir)

    for fn in file_path:
        df = pd.read_csv(f"{input_dir}\\{fn}", parse_dates=['Date'], index_col='Date')
        df = extend_and_ffill(df, freq)
        #df['Value'] = replace_outliers_iqr_interpolate(df['Value'])

        # HP Filter
        trend = apply_hp_filter(df['Value'], freq)
        hp_df = pd.DataFrame({'Date': trend.index, 'Value': trend.values})
        hp_file = f"{OUTPUT_BASE_HP}\\{freq}\\{fn}"
        hp_df.to_csv(hp_file, index=False)

        # Kalman Filter
        kf_series = apply_kalman_filter(df['Value'])
        kf_df = pd.DataFrame({'Date': kf_series.index, 'Value': kf_series.values})
        kf_file = f"{OUTPUT_BASE_KF}\\{freq}\\{fn}"
        kf_df.to_csv(kf_file, index=False)

        print(f"Processed {file_path} -> HP: {hp_file}, KF: {kf_file}")


if __name__ == '__main__':
    for freq in ['daily', 'monthly']:
        process_frequency(freq)
    print('All files processed.')
