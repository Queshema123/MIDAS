import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from load_data import load_data_to_dict, extract_records_from_date

class MIDAS():

    def __init__(self, data_folder:str, weights_type:type):
        self.data_dict = load_data_to_dict(data_folder)
        self.hf_data = {}
        self.w_type = weights_type

    def get_lags_from_frequency(frequency='monthly'):
        """
        Определяет количество лагов в зависимости от частоты данных.
        Пример: если данные ежемесячные, для прогноза на квартал (3 месяца).
    
        Параметры:
        frequency : str
            Частота данных. Возможные значения: 'monthly', 'daily'.
        
        Возвращает:
        lags : int
            Количество лагов для прогноза на квартал.
        """
        # Переделать по табличный метод
        if frequency == 'monthly':
            return 3  # 3 месяца для квартала
        elif frequency == 'daily':
            return 90  # 90 дней для квартала
        elif frequency == "quarterly":
            return 1
        else:
            raise ValueError("Неподдерживаемая частота данных. Используйте 'monthly' или 'daily'.")

    def prepare_hf_data(self, dict:dict):
        for name, info in dict.items():
            if info['frequency'] == 'quarterly':
                continue

            weights = self.w_type(
                np.array([1.0, 5.0]),
                info['data']
            )   
            self.hf_data[name] = {
                'frequency': info['frequency'],
                'weights_obj': weights
            }

    def calc_ar_val(self, Y:np.array, lags=1):
        model = AutoReg(Y, lags=lags, trend='n').fit()
        return sum(model.params * Y[-lags:])

    def calc_arima_val(self, Y:np.array, order:tuple = (1, 1, 1)):
        model = ARIMA(Y, order=order).fit()
        return sum(model.arparams * Y[-order[0]:][::-1]) # Берем последние значения и перечисляем их с конца

    def train(self, start_date:str, target_var:str, n_splits:int = 10, test_size:int = 1):
        extract_data_dict = extract_records_from_date(self.data_dict, start_date, target_var)
        lf_df = extract_data_dict[target_var]['data'].reset_index(drop=True)
        self.prepare_hf_data(extract_data_dict)

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        predicted, vals, dates = [], [], []

        for train_idx, test_idx in tscv.split(lf_df):
            lf_train = lf_df.iloc[train_idx]
            lf_test = lf_df.iloc[test_idx]
            ar_val = self.calc_ar_val(lf_train['Value'].values, 1)
            for opt_w_obj in (v['weights_obj'] for v in self.hf_data.values()):
                forecastes = []
                for i in train_idx:
                    end = pd.to_datetime(lf_train['Date'][i])
                    start = end.to_period('Q').start_time
                    forecast = ar_val
                    for w_obj in (v['weights_obj'] for v in self.hf_data.values() if v['weights_obj'] != opt_w_obj):
                        forecast += w_obj.calc_sum(start, end)
                    forecastes.append(forecast)

                opt_w_obj.optimize_theta(start, end, lf_train['Value'].values, forecastes)

            
            for i in test_idx:
                end = pd.to_datetime(lf_test['Date'][i])
                start = end.to_period('Q').start_time
                y = lf_test['Value'][i]
                forecast = ar_val
                for w_obj in (v['weights_obj'] for v in self.hf_data.values()):
                    forecast += w_obj.calc_sum(start, end)

                predicted.append(forecast)
                vals.append(y)
                dates.append(end)

        self.train_results =  {
            'MAE'      : mean_absolute_error( vals, predicted ),
            'RMSE'     : np.sqrt(mean_squared_error( vals, predicted )), 
            'dates'    : dates,
            'vals'     : vals,
            'forecast' : predicted
        }

    def forecast(self):
        pass