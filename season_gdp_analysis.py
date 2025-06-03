import pandas as pd
import matplotlib.pyplot as plt

# 1) Загрузка ряда (предполагаем квартальные данные GDP)
df = pd.read_csv("data\\low_freq_data\\GDP.csv", )

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

plt.plot(df['Value'], marker='o')

plt.show()

df.index = df.index.to_period('Q')
df['Quarter'] = df.index.quarter
df['Year'] = df.index.year

# Создаём сводную таблицу: строки — кварталы, столбцы — годы
pivot = df.pivot_table(index='Quarter', columns='Year', values='Value')

# Определяем последние 5 годов
last_years = sorted(pivot.columns)[-5:]   # при условии, что годы упорядочены численно

# Фильтруем таблицу по последним 5 годам
pivot_last5 = pivot[last_years]

# Рисуем линии для каждого из последних 5 лет
fig, ax = plt.subplots(figsize=(6,4))
for year in pivot_last5.columns:
    ax.plot(
        pivot_last5.index, 
        pivot_last5[year], 
        marker='o', 
        label=str(year)
    )

# Настраиваем ось X и подписи
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(['Q1','Q2','Q3','Q4'])
ax.set_xlabel('Квартал')
ax.set_ylabel('Value')
ax.set_title('Сезонный паттерн квартальных значений за последние 5 лет')
ax.legend(title='Год')
ax.grid(True)

plt.tight_layout()
plt.show()
