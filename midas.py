import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from bw_funcs import optimize_theta_metric, beta_weights, get_lags_from_frequency
import warnings


def train_midas_model_tssplit(data_dict, target_variable, n_splits=5):
    # Фильтрация данных: отделяем целевую переменную и высокочастотные данные
    lf_df = data_dict[target_variable]['data'].reset_index(drop=True)
    hf_data = {k: v['data'].reset_index(drop=True) for k, v in data_dict.items() if v['frequency'] != 'quarterly' }

    # Настройка кросс-валидации
    T = len(lf_df)
    window_size = 4
    n_splits = max(1, T - window_size)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1, max_train_size=window_size)
    
    true_vals, preds = [], []
    model_params = []

    for train_idx, test_idx in tscv.split(lf_df):
        lf_train = lf_df.iloc[train_idx]
        lf_test = lf_df.iloc[test_idx]

        hf_matrix = {}
        for metric, df_hf in hf_data.items():
            X_list = []
            for dt in pd.to_datetime(lf_train['Date']):
                end = dt.to_period('Q').end_time
                n_lags = get_lags_from_frequency(data_dict[metric]['frequency'])
                window = df_hf[df_hf['Date'] <= end]['Value'].values[-n_lags:]
                if len(window) < n_lags:
                    warnings.warn(f"Недостаточно данных для {metric} заполнение пропусков значением NAN", UserWarning)
                    pad = np.full(n_lags - len(window), np.nan)
                    window = np.concatenate([pad, window])
                X_list.append(window)
            hf_matrix[metric] = np.vstack(X_list)

        y_train = lf_train['Value'].values

        # Оптимизация параметров theta для каждой HF-метрики
        theta_params = {
            metric: optimize_theta_metric(X, y_train)
            for metric, X in hf_matrix.items()
        }

        # Прогноз для тестовой точки
        forecast = 0.0
        for metric, df_hf in hf_data.items():
            dt = pd.to_datetime(lf_test['Date'].iloc[0])
            end = dt.to_period('Q').end_time
            n_lags = get_lags_from_frequency(data_dict[metric]['frequency'])
            window = df_hf[df_hf['Date'] <= end]['Value'].values[-n_lags:]
            if len(window) < n_lags:
                pad = np.full(n_lags - len(window), np.nan)
                window = np.concatenate([pad, window])
            X_test = window.reshape(1, -1)

            j = np.arange(1, n_lags + 1)
            t1 = theta_params[metric]['theta1']
            t2 = theta_params[metric]['theta2']
            w = beta_weights(j, t1, t2, n_lags)
            forecast += (X_test * w).sum()

        true_vals.append(lf_test['Value'].iloc[0])
        preds.append(forecast)

        # Сохраняем параметры модели и веса
        weights = {
            m: beta_weights(np.arange(1, n_lags + 1), p['theta1'], p['theta2'], n_lags)
            for m, p in theta_params.items()
        }
        model_params.append({
            'train_end': pd.to_datetime(lf_train['Date'].iloc[-1]).to_period('Q').end_time,
            'theta': theta_params,
            'weights': weights
        })

    metrics = {
        'MAE': mean_absolute_error(true_vals, preds),
        'RMSE': np.sqrt(mean_squared_error(true_vals, preds)),
        'true': true_vals,
        'predicted': preds
    }
    return model_params, metrics


def forecast_next_quarter_from_model(data_dict: dict, trained_model, target_var: str = 'GDP') -> pd.DataFrame:
    """
    Выполняет прогноз на следующий квартал целевой переменной,
    используя переданный словарь данных и уже обученную модель MIDAS.

    :param data_dict: словарь серий {'daily': ..., 'monthly': ..., 'quarterly': ..., target_var: Series/DF}
    :param trained_model: обученный объект модели MIDAS с методом predict(horizon)
    :param target_var: имя целевой переменной, например 'GDP'
    :return: DataFrame с одним значением прогноза по дате следующего квартала
    """
    # Получаем серию квартальных наблюдений целевой переменной
    series = data_dict[target_var]

    # Проверим, что серия индексирована датами
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Индекс серии должен быть DatetimeIndex с квартальными датами")

    # Выполняем прогноз на следующий квартал
    forecast_values = trained_model.predict(horizon=1)

    # Определяем дату конца следующего квартала
    last_date = series.index[-1]
    next_quarter_date = last_date + pd.offsets.QuarterEnd()

    # Формируем DataFrame с прогнозом
    df_forecast = pd.DataFrame({
        'Date': [next_quarter_date],
        f'{target_var}_forecast': forecast_values
    }).set_index('Date')

    return df_forecast
