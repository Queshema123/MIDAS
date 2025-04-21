"""
    Подбор оптимальных параметров бета-функции для вычисления весов высокочастотных данных 
"""

import numpy as np
from scipy.special import beta as beta_function
from scipy.optimize import minimize

def beta_weights(j, theta1, theta2, J):
    """
    Вычисляет веса лагов с использованием бета-функции.

    Параметры:
    j : int или массив
        Лаг или массив лагов.
    theta1 : float
        Первый параметр бета-функции.
    theta2 : float
        Второй параметр бета-функции.
    J : int
        Максимальный лаг.

    Возвращает:
    w_j : float или массив
        Веса для лагов.
    """
    # Нормализуем лаги в интервал [0, 1]
    x = j / J

    # Защита от деления на ноль для крайних значений
    epsilon = 1e-8  # Очень маленькое значение для замены
    x = np.clip(x, epsilon, 1 - epsilon)  # Ограничиваем x, чтобы избежать деления на 0 или 1

    # Бета-функция для вычисления весов
    w_j = (x ** (theta1 - 1)) * ((1 - x) ** (theta2 - 1))  # Бета-функция
    w_j /= beta_function(theta1, theta2)  # Нормализация

    return w_j

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
    if frequency == 'monthly':
        return 3  # 3 месяца для квартала
    elif frequency == 'daily':
        return 90  # 90 дней для квартала
    elif frequency == "quarterly":
        return 1
    else:
        raise ValueError("Неподдерживаемая частота данных. Используйте 'monthly' или 'daily'.")

def optimize_theta_metric(X, y):
    """
    Оптимизация параметров theta1 и theta2 для одной высокочастотной метрики
    с помощью scipy.optimize.minimize (метод L-BFGS-B).

    X: np.ndarray shape (n_samples, n_lags)
    y: np.ndarray shape (n_samples,)
    """
    def mse_loss(t):
        # t: [theta1, theta2]
        J = X.shape[1]
        j = np.arange(1, J+1)
        w = beta_weights(j, t[0], t[1], J)
        y_pred = (X * w).sum(axis=1)
        return np.mean((y - y_pred)**2)

    bounds = [(1e-6, 1.0), (1e-6, 1.0)]
    res = minimize(
        fun=mse_loss,
        x0=np.array([0.5, 0.5]),
        bounds=bounds,
        method='L-BFGS-B'
    )
    theta1_opt, theta2_opt = res.x
    return {'theta1': float(theta1_opt), 'theta2': float(theta2_opt)}
