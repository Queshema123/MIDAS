import os
import pandas as pd

def load_data_to_dict(folder_path):
    """
    Загружает данные из папок low_freq_data и high_freq_data и формирует словарь
    в формате {'Metric_name': {'frequency': data}}.

    Параметры:
    ----------
    folder_path : str
        Путь к папке проекта (например, 'folder').
    forecast_horizon : str, optional
        Горизонт прогноза. Поддерживается 'quarterly' (квартал).
        По умолчанию 'quarterly'.

    Возвращает:
    -----------
    dict
        Словарь в формате {'Metric_name': {'frequency': data}}.
    """
    data_dict = {}

    # Путь к папке с низкочастотными данными
    low_freq_folder = f"data\\low_freq_data"
    for file_name in os.listdir(low_freq_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(low_freq_folder, file_name)
            metric_name = file_name.split('.')[0]  # Имя метрики из названия файла
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            data_dict[metric_name] = {'frequency': 'quarterly', 'data': df}
    
    # Пути к папкам с высокочастотными данными
    daily_folder = os.path.join(folder_path, 'daily')
    monthly_folder = os.path.join(folder_path, 'monthly')

    # Загрузка ежедневных данных
    for file_name in os.listdir(daily_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(daily_folder, file_name)
            metric_name = file_name.split('.')[0]  # Имя метрики из названия файла
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            data_dict[metric_name] = {'frequency': 'daily', 'data': df}

    # Загрузка ежемесячных данных
    for file_name in os.listdir(monthly_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(monthly_folder, file_name)
            metric_name = file_name.split('.')[0]  # Имя метрики из названия файла
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            data_dict[metric_name] = {'frequency': 'monthly', 'data': df}

    return data_dict

def extract_records_from_date(data_dict, start_date, target_metric_name):
    """
    Извлекает данные, автоматически определяя start_date для высокочастотных данных 
    как начало квартала указанной даты конца квартала для ВВП.
    """
    extracted_data = {}
    start_date = pd.to_datetime(start_date)

    # Вычисляем начало квартала для стартовой даты
    start_of_quarter = start_date.to_period("Q").start_time

    for metric, info in data_dict.items():
        df = info['data'].copy(deep=True)
        df['Date'] = pd.to_datetime(df['Date'])

        # Преобразование к концу квартала ТОЛЬКО для квартальных данных
        if info['frequency'] == 'quarterly':
            df.loc[:, 'Date'] = (
                df['Date']
                .dt.to_period('Q')
                .dt.to_timestamp('Q')
            )

        # Для ВВП используем исходный start_date, для остальных - начало квартала
        filter_date = start_date if metric == target_metric_name else start_of_quarter
        mask = (df['Date'] >= filter_date)
        extracted_df = df.loc[mask]

        if extracted_df.empty:
            raise ValueError(f"Нет данных для {metric} от {filter_date}")

        extracted_data[metric] = {
            'frequency': info['frequency'],
            'data': extracted_df[['Date', 'Value']].reset_index(drop=True)
        }

        #print(f"[DEBUG] {metric}: {len(extracted_df)} samples, {extracted_df['Date'].min().date()} - {extracted_df['Date'].max().date()}")

    return extracted_data