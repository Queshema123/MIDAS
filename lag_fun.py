import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.special import expit  # Логистическая функция

plt.figure(figsize=(15, 10))

# 1. Бета-распределение
plt.subplot(2, 2, 1)
x = np.linspace(0, 1, 1000)
params = [(0.5, 0.5), (2, 5), (5, 2), (1, 1), (3, 3)]
for a, b in params:
    plt.plot(x, beta.pdf(x, a, b), label=f'α={a}, β={b}')
plt.title('Бета-распределение')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# 2. Экспоненциальные полиномы Алмона
plt.subplot(2, 2, 2)
m = 12  # Количество лагов
i = np.arange(1, m+1)

# Разные наборы параметров
theta_sets = [
    (-0.1, -0.01),  # Медленное затухание
    (-0.5, -0.05),   # Среднее затухание
    (-1.0, -0.1),    # Быстрое затухание
    (0.2, -0.02)     # Горб в начале
]

for θ1, θ2 in theta_sets:
    w = np.exp(θ1 * i + θ2 * i**2)
    w /= w.sum()  # Нормировка
    plt.plot(i, w, 'o-', label=f'θ1={θ1}, θ2={θ2}')

plt.title(f'Экспоненциальные полиномы Алмона (m={m})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.xticks(i)

# 3. Неэкспоненциальные полиномы Алмона
plt.subplot(2, 2, 3)
theta_sets_linear = [
    (1, 0.2, -0.02),   # Стандартный
    (0.5, 0.3, -0.03), # Более пологий
    (0.1, 0.4, -0.04)  # Сдвиг максимума
]

for θ0, θ1, θ2 in theta_sets_linear:
    w = θ0 + θ1 * i + θ2 * i**2
    w = np.maximum(w, 0)  # Обеспечение неотрицательности
    w /= w.sum()          # Нормировка
    plt.plot(i, w, 'o-', label=f'θ0={θ0}, θ1={θ1}, θ2={θ2}')

plt.title(f'Неэкспоненциальные полиномы Алмона (m={m})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.xticks(i)

# 4. Сравнение двух типов полиномов Алмона
plt.subplot(2, 2, 4)
θ_exp = (-0.3, -0.03)  # Параметры для экспоненциального
θ_lin = (0.8, 0.15, -0.015)  # Параметры для линейного

# Экспоненциальный
w_exp = np.exp(θ_exp[0] * i + θ_exp[1] * i**2)
w_exp /= w_exp.sum()

# Линейный
w_lin = θ_lin[0] + θ_lin[1] * i + θ_lin[2] * i**2
w_lin = np.maximum(w_lin, 0)
w_lin /= w_lin.sum()

plt.plot(i, w_exp, 'bo-', label='Экспоненциальный')
plt.plot(i, w_lin, 'ro-', label='Линейный')
plt.title('Сравнение полиномов Алмона')
plt.xlabel('Лаг')
plt.ylabel('Вес')
plt.legend()
plt.grid(True)
plt.xticks(i)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Параметры
d = 12  # Максимальный лаг
j = np.arange(1, d+1)  # Лаги от 1 до d (j=0 исключен из-за log(0))

# Наборы параметров (θ1, θ2)
theta_sets = [
    (0.0, 1.0),    # Базовый
    (1.0, 0.5),    # Сдвиг пика
    (2.0, 0.1),    # Узкий пик
    (0.5, 2.0),    # Плавное распределение
    (1.5, 0.01)    # Очень острый пик
]

plt.figure(figsize=(12, 8))

# Расчет и построение для каждого набора параметров
for θ1, θ2 in theta_sets:
    # Вычисление числителя
    log_j = np.log(j)
    numerator = (1/j) * (1/(θ2**2 + (log_j - θ1)**2))
    
    # Вычисление знаменателя (нормировочной константы)
    denom_sum = 0
    for s in range(1, d+1):
        log_s = np.log(s)
        denom_sum += (1/s) * (1/(θ2**2 + (log_s - θ1)**2))
    
    # Весовые коэффициенты
    weights = numerator / denom_sum
    
    # Построение графика
    plt.plot(j, weights, 'o-', linewidth=2, markersize=6, 
             label=f'θ₁={θ1}, θ₂={θ2}')

# Настройка оформления
plt.title(f'Log-Couch (d={d})', fontsize=14)
plt.xlabel('Лаг (j)', fontsize=12)
plt.ylabel('Вес B(j, θ₁, θ₂)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(j)
plt.ylim(0, 0.5)  # Ограничение по Y для лучшей видимости

# Добавление пояснения формулы


plt.tight_layout()
plt.show()