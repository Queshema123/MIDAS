import os
import pandas as pd
from pandas.tseries.offsets import QuarterEnd

def quarter(path_to_file: str, path_to_save:str):
    # Загрузка данных из Excel
    df = pd.read_excel(path_to_file, header=None)

    # Извлечение года из первой строки
    year = df.iloc[0, 0]
    try:
        year = int(year)
    except ValueError:
        raise ValueError(f"Неверный формат года: {year}")

    # Извлечение кварталов и значений
    quarters = df.iloc[1, :].dropna().tolist()
    values = df.iloc[2, :].dropna().tolist()

    if len(quarters) != len(values):
        raise ValueError("Количество кварталов и значений не совпадает")

    # Сопоставление кварталов с последним днем квартала
    quarter_to_month = {
        'I квартал': 3,
        'II квартал': 6,
        'III квартал': 9,
        'IV квартал': 12
    }

    # Создание списка данных для нового DataFrame
    data = []
    for q, val in zip(quarters, values):
        q_clean = q.strip()
        if q_clean not in quarter_to_month:
            raise ValueError(f"Неизвестный квартал: {q_clean}")
        month = quarter_to_month[q_clean]
        date = pd.Timestamp(year=year, month=month, day=1) + QuarterEnd(0)
        if month == 12:
            year+=1
        # Обработка значения: удаление пробелов и замена точки на запятую
        value_clean = str(val).replace(' ', '')
        data.append({'Date': date.strftime('%d.%m.%Y'), 'Value': value_clean})

    # Создание и сохранение результата
    output_df = pd.DataFrame(data)
    output_df['Value'] = output_df['Value'].astype(float)
    fname = path_to_file[path_to_file.rfind('\\')+1:path_to_file.find('.')] + '.csv'
    output_df.to_csv(path_to_save + "\\" + fname, index=False)

def convert_to_bill_rub(path_to_file:str, path_to_exchange_file:str, path_to_save:str):
    df = pd.read_csv(path_to_file)
    ex_df = pd.read_csv(path_to_exchange_file)

    df['Date'] = pd.to_datetime(df['Date'], format="%d.%m.%Y")
    ex_df['Date'] = pd.to_datetime(ex_df['Date'], format="%d.%m.%Y")

    df = df.sort_values('Date')
    ex_df = ex_df.sort_values('Date')

    merged = pd.merge_asof(df, ex_df, on="Date", direction="backward", suffixes=('', '_rub'))
    merged['Value'] = (merged['Value']*merged['Value_rub']) / 1000 
    merged['Date'] = merged['Date'].dt.strftime('%d.%m.%Y')

    fname = path_to_file[path_to_file.rfind('\\')+1:path_to_file.find('.')] + '.csv'
    merged[['Date', 'Value']].to_csv(path_to_save + "\\" + fname, index=False)

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
    low_freq_folder = os.path.join(folder_path, 'data', 'low_freq_data')
    for file_name in os.listdir(low_freq_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(low_freq_folder, file_name)
            metric_name = file_name.split('.')[0]  # Имя метрики из названия файла
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            data_dict[metric_name] = {'frequency': 'quarterly', 'data': df}
    
    # Пути к папкам с высокочастотными данными
    high_freq_folder = os.path.join(folder_path, 'data', 'high_freq_data')
    daily_folder = os.path.join(high_freq_folder, 'daily')
    monthly_folder = os.path.join(high_freq_folder, 'monthly')

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

def fill_missing_dates(df, date_col='Date', value_col='Value', freq='D'):
    """
    Заполняет пропущенные даты в DataFrame и заполняет пропущенные значения
    с использованием forward fill.
    """
    df = df.copy()
    df.loc[:, date_col] = pd.to_datetime(df[date_col], errors='coerce')
    valid_dates_df = df.dropna(subset=[date_col]).copy()
    # Если DataFrame пустой, возвращаем его без изменений
    if df.empty:
        return df.copy()
    
    if freq == 'Q':
        df[date_col] = df[date_col].dt.to_period('Q').dt.end_time.dt.date

    # Конвертируем даты и проверяем на валидность
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Удаляем строки с NaT в датах
    valid_dates_df = df.dropna(subset=[date_col])
    
    # Если после фильтрации нет данных
    if valid_dates_df.empty:
        raise ValueError("Нет валидных дат в DataFrame")

    # Получаем границы диапазона дат
    start_date = valid_dates_df[date_col].min()
    end_date = valid_dates_df[date_col].max()

    # Создаём полный диапазон дат
    full_date_range = pd.date_range(
        start=start_date,
        end=end_date,
        freq=freq
    )

    # Создаём полный DataFrame и мерджим
    full_df = pd.DataFrame({date_col: full_date_range})
    merged_df = pd.merge(full_df, valid_dates_df, on=date_col, how='left')

    # Заполняем пропущенные значения
    merged_df[value_col] = merged_df[value_col].ffill()

    return merged_df

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

        print(f"[DEBUG] {metric}: {len(extracted_df)} samples, {extracted_df['Date'].min().date()} - {extracted_df['Date'].max().date()}")

    return extracted_data