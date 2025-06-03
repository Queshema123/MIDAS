import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize

class Weights():
    
    def __init__(self, thetas, df, lambda_l1:float = 0.0, lambda_l2:float = 0.0):
        self.params = thetas
        self.data = df
        self.weights = []
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def calc_weights(self, J):
        self.weights = np.ones(J)
    
    def extract_data(self, start:pd.DatetimeIndex, end:pd.DatetimeIndex):
        return self.data.loc[ (self.data['Date']>= start) & (self.data['Date'] <= end), 'Value' ].to_numpy()

    def calc_sum(self, start:pd.DatetimeIndex, end:pd.DatetimeIndex):
        extr_data = self.extract_data(start, end)
        self.calc_weights( len(extr_data) )
        return (extr_data * self.weights).sum()

    def mse_loss(self, th:np.array, start:pd.DatetimeIndex, end:pd.DatetimeIndex, true_val:float, forecast_val:float):
        pass

    def optimize_theta(self, start:pd.DatetimeIndex, end:pd.DatetimeIndex, true_value:float, forecast_val:float):
        pass


class BetaWeights(Weights):

    def __init__(self, thetas:np.array, x:pd.DataFrame, lambda_l1:float = 0.0, lambda_l2:float = 0.0):
        super().__init__(thetas, x, lambda_l1, lambda_l2)
    
    def calc_weights(self, J):
        '''
            Вычисляет значения весов на основе бета-распределения(Beta lag / Beta polynom)
            Особенностью функции является логорифмирование значений для большого кол-ва лагов.
            Затем обратное преобразование и нахождение суммы.
            Это нужно при большом кол-ве лагов, например, обучение на 4 кварталах для ежедневых данных
            Из-за чего происходит переполнение и сумма будет равна 0
        '''
        j = np.arange(1, J+1)
        j = np.asarray(j)

        x = j/J
        eps = 1e-8
        x = np.clip(x, eps, 1-eps)
        # Логарифм числителя: (theta1-1)*ln(x) + (theta2-1)*ln(1-x)
        log_num = (self.params[0] - 1) * np.log(x) + (self.params[1] - 1) * np.log(1 - x)
        i = np.arange(1, J + 1)
        x_i = i / J
        x_i = np.clip(x_i, eps, 1 - eps)
        log_terms = (self.params[0] - 1) * np.log(x_i) + (self.params[1] - 1) * np.log(1 - x_i)
        log_denom = logsumexp(log_terms)
        weights = np.exp(log_num - log_denom)
        weights = np.clip(weights, 1e-8, None)
        weights /= weights.sum() 
        self.weights = weights
    
    def extract_data(self, start:pd.DatetimeIndex, end:pd.DatetimeIndex):
        return self.data.loc[ (self.data['Date']>= start) & (self.data['Date'] <= end), 'Value' ].to_numpy()

    def calc_sum(self, start:pd.DatetimeIndex, end:pd.DatetimeIndex):
        extr_data = self.extract_data(start, end)
        self.calc_weights( len(extr_data) )
        return (extr_data * self.weights).sum()
    
    def mse_loss(self, th:np.array, start:pd.DatetimeIndex, end:pd.DatetimeIndex, true_values:np.array, forecast_vals:np.array):
        self.params = th
        summary = forecast_vals + self.calc_sum(start, end)
        
        mse = np.mean((true_values - summary) ** 2)
        l1 = self.lambda_l1 * np.sum(np.abs(self.params))
        l2 = self.lambda_l2 * np.sum(self.params**2)

        return mse + l1 + l2

    def optimize_theta(self, start:pd.DatetimeIndex, end:pd.DatetimeIndex, true_values:np.array, forecast_vals:np.array):
        res = minimize(
            fun=self.mse_loss,
            x0=self.params,
            args=(start, end, true_values, forecast_vals),
            method='L-BFGS-B'
        )
        self.params = res.x
        return {f'theta{i+1}': float(th) for i, th in enumerate(self.params)}


class AlmonExpPolyWeights(Weights):
    def __init__(self, thetas: np.array, df: pd.DataFrame, lambda_l1:float = 0.0, lambda_l2:float = 0.0):
        super().__init__(thetas, df, lambda_l1, lambda_l2)

    def calc_weights(self, J: int):
        k = np.arange(1, J + 1)
        powers = np.array([k**(q+1) for q in range(len(self.params))])
        exponents = np.dot(self.params, powers)
        log_numerators = exponents
        log_denominator = logsumexp(exponents)
        weights = np.exp(log_numerators - log_denominator)
        self.weights = weights

    def extract_data(self, start: pd.DatetimeIndex, end: pd.DatetimeIndex):
        return self.data.loc[(self.data['Date'] >= start) & (self.data['Date'] <= end), 'Value'].to_numpy()

    def calc_sum(self, start: pd.DatetimeIndex, end: pd.DatetimeIndex):
        extr_data = self.extract_data(start, end)
        self.calc_weights(len(extr_data))
        return (extr_data * self.weights).sum()

    def mse_loss(self, th: np.array, start: pd.DatetimeIndex, end: pd.DatetimeIndex, true_values: np.array, forecast_vals: np.array):
        self.params = th
        summary = forecast_vals + self.calc_sum(start, end)
        
        mse = np.mean((true_values - summary) ** 2)
        l1 = self.lambda_l1 * np.sum(np.abs(self.params))
        l2 = self.lambda_l2 * np.sum(self.params**2)

        return mse + l1 + l2

    def optimize_theta(self, start: pd.DatetimeIndex, end: pd.DatetimeIndex, true_values: np.array, forecast_vals: np.array):
        res = minimize(
            fun=self.mse_loss,
            x0=self.params,
            args=(start, end, true_values, forecast_vals),
            method='L-BFGS-B'
        )
        self.params = res.x
        return {f'theta{i+1}': float(th) for i, th in enumerate(self.params)}    
    

class AlmonPolynomialWeights(Weights):
    def __init__(self, thetas: np.array, df: pd.DataFrame, lambda_l1:float = 0.0, lambda_l2:float = 0.0):
        super().__init__(thetas, df, lambda_l1, lambda_l2)

    def _poly_weights(self, J):
        j = np.arange(1, J + 1)
        poly_terms = np.vstack([j ** i for i in range(len(self.params))])
        weights = self.params @ poly_terms
        return weights

    def calc_weights(self, J):
        raw_weights = self._poly_weights(J)
        self.weights = raw_weights
        raw_weights = np.clip(raw_weights, 1e-8, None)  
        self.weights = raw_weights / np.sum(raw_weights)

    def extract_data(self, start: pd.DatetimeIndex, end: pd.DatetimeIndex):
        return self.data.loc[(self.data['Date'] >= start) & (self.data['Date'] <= end), 'Value'].to_numpy()

    def calc_sum(self, start: pd.DatetimeIndex, end: pd.DatetimeIndex):
        values = self.extract_data(start, end)
        self.calc_weights(len(values))
        return np.sum(values * self.weights)

    def mse_loss(self, th: np.array, start, end, true_vals, forecast_vals):
        self.params = th
        summary = forecast_vals + self.calc_sum(start, end)
        
        mse = np.mean((true_vals - summary) ** 2)
        l1 = self.lambda_l1 * np.sum(np.abs(self.params))
        l2 = self.lambda_l2 * np.sum(self.params**2)

        return mse + l1 + l2

    def optimize_theta(self, start, end, true_vals, forecast_vals):
        res = minimize(
            fun=self.mse_loss,
            x0=self.params,
            args=(start, end, true_vals, forecast_vals),
            method='L-BFGS-B'
        )
        self.params = res.x
        return {f"theta{i}": float(p) for i, p in enumerate(self.params)}
