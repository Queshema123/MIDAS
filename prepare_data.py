import os
import glob
import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter
from pykalman import KalmanFilter
from scipy.stats.mstats import winsorize

# Define input and output base paths
INPUT_BASE = "data\\high_frequency_data"
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


def replace_outliers_iqr_winsorize(series: pd.Series, limits=(0.05, 0.05)) -> pd.Series:
    """
    Вариант 3: винзоризация — обрезка крайних долей.
    По умолчанию обрезает нижние 5% значений до 5-го перцентиля
    и верхние 5% до 95-го.
    """
    # winsorize возвращает masked array, приводим обратно в Series
    arr = winsorize(series, limits=limits)
    return pd.Series(arr, index=series.index)

def replace_outliers_iqr(series):
    """
    Identify outliers based on IQR and replace each outlier
    with the median of its nearest non-outlier neighbors.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outliers = (series < lower) | (series > upper)
    clean = series[~outliers]
    for idx in series[outliers].index:
        # find neighbors
        prev_vals = clean[:idx]
        next_vals = clean[idx:]
        if not prev_vals.empty and not next_vals.empty:
            rep = np.median([prev_vals.iloc[-1], next_vals.iloc[0]])
        elif not prev_vals.empty:
            rep = prev_vals.iloc[-1]
        elif not next_vals.empty:
            rep = next_vals.iloc[0]
        else:
            rep = series[idx]
        series.at[idx] = rep
    return series


def apply_hp_filter(series, lamb=1600):
    """
    Apply Hodrick-Prescott filter and return the trend (smoothed) component.
    """
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
    files = glob.glob(os.path.join(input_dir, '*.csv'))

    for file_path in files:
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        df = extend_and_ffill(df, freq)
        df['Value'] = replace_outliers_iqr_interpolate(df['Value'])

        # HP Filter
        trend = apply_hp_filter(df['Value'])
        hp_df = pd.DataFrame({'Date': trend.index, 'HP_Value': trend.values})
        hp_file = os.path.join(OUTPUT_BASE_HP, freq,
                               os.path.basename(file_path).replace('.csv', '_hp.csv'))
        hp_df.to_csv(hp_file, index=False)

        # Kalman Filter
        kf_series = apply_kalman_filter(df['Value'])
        kf_df = pd.DataFrame({'Date': kf_series.index, 'KF_Value': kf_series.values})
        kf_file = os.path.join(OUTPUT_BASE_KF, freq,
                               os.path.basename(file_path).replace('.csv', '_kf.csv'))
        kf_df.to_csv(kf_file, index=False)

        print(f"Processed {file_path} -> HP: {hp_file}, KF: {kf_file}")


if __name__ == '__main__':
    for freq in ['daily', 'monthly']:
        process_frequency(freq)
    print('All files processed.')
