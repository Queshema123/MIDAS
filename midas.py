import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from load_data import load_data_to_dict, extract_records_from_date

class MIDAS():

    def __init__(self, data_folder:str, weights_type:type, seasons_coeff:dict = { 1 : 0.91, 2 : 0.96, 3 : 1.025, 4 : 1.1 }):
        self.data_dict = load_data_to_dict(data_folder)
        self.hf_data = {}
        self.w_type = weights_type
        self.lf_coeffs = []
        self.quarter_coeff = seasons_coeff

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

    def mse_loss(self, thetas, true_vals, data, summary):
        self.lf_coeffs = thetas
        predicted = summary + np.sum(data*self.lf_coeffs)
        return np.mean((true_vals-predicted)**2)

    def optimize_lf_coeff(self, true_vals, data, summary):
        self.lf_coeffs = np.zeros(len(data))
        res = minimize(
            fun = self.mse_loss,
            x0 = self.lf_coeffs,
            args = (true_vals, data, summary)
        )
        self.lf_coeffs = res.x

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

    def get_season_coeff(self, date:pd.Timestamp):
        return self.quarter_coeff.get(date.quarter, 1)
        
    def train(self, start_date:str, target_var:str, n_splits:int = 15, test_size:int = 1):
        extract_data_dict = extract_records_from_date(self.data_dict, start_date, target_var)
        lf_df = extract_data_dict[target_var]['data'].reset_index(drop=True)
        #lf_df = lf_df.drop(lf_df.index[-1]) # Удаление последнего значения 
        self.prepare_hf_data(extract_data_dict)

        # Разбиваем данные на отрезок от 1 квартала до t, тестируем на t+test_size кварталах
        # После подсчета ошибок добавляем в обучающую выборку тестовую. Отрезок - [1, t+test_size]
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, max_train_size=5)
        predicted, vals, dates = [], [], []

        for train_idx, test_idx in tscv.split(lf_df):
            lf_train = lf_df.iloc[train_idx]
            lf_test = lf_df.iloc[test_idx]
            # Подбираем параметры тета для высокочастотных переменных. Фиксируем одну метрику с коэффициентами для оптимизации
            for opt_w_obj in (v['weights_obj'] for v in self.hf_data.values()):
                forecastes = []
                # Проходимся по всему тренировочном кварталам
                for i in train_idx: 
                    end = pd.to_datetime(lf_train['Date'][i])
                    start = end.to_period('Q').start_time
                    coeff = self.get_season_coeff(start)
                    forecast = 0
                    # Вычисляем суммы всех метрик кроме оптимизируемой
                    for w_obj in (v['weights_obj'] for v in self.hf_data.values() if v['weights_obj'] != opt_w_obj):
                        forecast += w_obj.calc_sum(start, end) * coeff
                    forecastes.append(forecast)
                # Оптимизируем значения параметров под вычисленные прогнозы
                opt_w_obj.optimize_theta(start, end, lf_train['Value'].values, forecastes)

            # Вычисление тета для исторических значений target_var
            self.optimize_lf_coeff(lf_train['Value'].values, lf_train['Value'].values, forecastes) 

            # Пронозируем на обученных параметрах
            for i in test_idx:
                end = pd.to_datetime(lf_test['Date'][i])
                start = end.to_period('Q').start_time
                y = lf_test['Value'][i]
                forecast = 0
                coeff = self.get_season_coeff(start)
                for w_obj in (v['weights_obj'] for v in self.hf_data.values()):
                    forecast += w_obj.calc_sum(start, end) * coeff
                forecast += np.sum(lf_train['Value'].values * self.lf_coeffs) 

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

    def forecast(self, date:str, var:str) -> float:
        date = pd.to_datetime(date)
        q_start, q_end =  date.to_period('Q').start_time, date.to_period('Q').end_time
        extr_data = extract_records_from_date(self.data_dict, q_start, var)
        self.prepare_hf_data(extr_data)
        forecast = 0
        for w_obj in (v['weights_obj'] for v in self.hf_data.values()):
            forecast += w_obj.calc_sum(q_start, q_end)
        return {'forecast': forecast, 'value': extr_data[var]['data'].iloc[-1]}

