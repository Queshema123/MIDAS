from load_data import load_data_to_dict, extract_records_from_date
from midas import train_midas_model_tssplit, forecast_next_quarter_from_model
from matplotlib import pyplot as plt

# Основной скрипт
if __name__ == "__main__":

    folder_path = "" 
    data_dict = load_data_to_dict(folder_path)
        
    target_var = 'GDP'
    start_date = '2019-03-31'
    extracted_data = extract_records_from_date(data_dict, start_date, target_var)
    
    model_params, metrics = train_midas_model_tssplit(
        data_dict=extracted_data,
        target_variable=target_var
    )
    
    print("Результаты обучения:")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}\n")
    mae = metrics['MAE']
    rmse = metrics['RMSE']
    mean_val = sum(metrics['predicted']) / len(metrics['predicted'])
    mae_relative = (mae / mean_val) * 100
    rmse_rel = (rmse / mean_val) * 100
    print(f"Relative MAE: {mae_relative:.2f}")
    print(f"Relative RMSE: {rmse_rel:.2f}")
    plt.plot(metrics['true'], label='Actual')
    plt.plot(metrics['predicted'], label='Predicted')
    plt.legend()
    plt.show()
    input()

"""
# Пример использования
if __name__ == '__main__':
    from load_data import load_data_to_dict, extract_records_from_date
    from midas import train_midas_model_tssplit

    folder = 'data/high_frequency_data'
    # Загружаем и экстрагируем данные
    data = load_data_to_dict(folder)
    data = extract_records_from_date(data, '2019-03-31', 'GDP')

    # Убираем последний квартал из целевой переменной для обучения
    q = data['GDP']
    data['GDP'] = q.iloc[:-1]

    # Обучаем модель MIDAS
    model, metrics = train_midas_model_tssplit(data_dict=data, target_variable='GDP')

    # Делаем прогноз
    forecast = forecast_next_quarter_from_model(data, model, 'GDP')
    print("Forecast for next quarter:")
    print(forecast)
"""