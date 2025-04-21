"""
     Подбор оптимальных параметров для вычисления весов низкочастотных(исторических) прогнозируемой переменной 
"""
# Построение ACF/PACF графиков для обоснования выбора AR(1) модели для вычисления коэффициентов для исторических данных объясняющей переменной
''' 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
# Построение графика ACF
plt.figure(figsize=(10, 6))
plot_acf(Y, lags=5)  # Можно настроить количество лагов (lags)
plt.title("ACF (Автокорреляционная функция)")
plt.show()

# Построение графика PACF
plt.figure(figsize=(10, 6))
plot_pacf(Y, lags=5)  # Можно настроить количество лагов (lags)
plt.title("PACF (Частичная автокорреляционная функция)")
plt.show()
'''

# Вычисление коэффициентов для исторических данных объясняющей переменной
from statsmodels.tsa.ar_model import AutoReg
def calc_lw_data_params(Y, n_lags):
    ar_model = AutoReg(Y, lags=n_lags).fit()
    return  ar_model.params