import pandas as pd
import os
from statsmodels.tsa.filters.hp_filter import hpfilter

# Папка с исходными CSV-файлами
input_folder = 'data/high_freq_data/daily'
output_folder = 'data/high_freq_data/daily_filtered'
os.makedirs(output_folder, exist_ok=True)

# Параметр lambda для monthly данных
hp_lambda = 14400  # 1600 - MS 14400 -D

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_folder, filename)
        df = pd.read_csv(file_path)

        # Преобразование даты
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

        # Применение HP фильтра
        cycle, trend = hpfilter(df['Value'], lamb=hp_lambda)
        df['Value'] = trend
        df['HP_Cycle'] = cycle

        # Сохранение
        output_path = os.path.join(output_folder, filename)
        df.to_csv(output_path, index=False)

        print(f"✔ Файл обработан: {filename}")
