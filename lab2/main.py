import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Шаг 1: Загрузка датасета
df = pd.read_parquet('dota2_matches.parquet')

# 1. Выявление пропусков
# Визуальный способ: тепловая карта
cols = df.columns[:]
colours = ['#eeeeee', '#00ff00']
plt.figure(figsize=(10, 6))
sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours))
plt.title('Пропуски данных')
#plt.show()

# 2. Исключение строк и столбцов с наибольшим количеством пропусков
threshold = 0.5  # Пороговое значение для удаления строк/столбцов с пропусками
df_cleaned = df.dropna(thresh=df.shape[1]*threshold, axis=0)  # Удаление строк
df_cleaned = df_cleaned.dropna(thresh=df_cleaned.shape[0]*threshold, axis=1)  # Удаление столбцов
print("Размер датасета после удаления строк и столбцов с пропусками:", df_cleaned.shape)

# 3. Замена оставшихся пропусков на логически обоснованные значения
# Пример для числовых значений: замена числовых пропусков медианой
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].median())

# 4. Построение гистограммы распределения до и после обработки пропусков
# Гистограмма исходного датасета
plt.figure(figsize=(10, 6))
sns.histplot(df['dire_player_1_networth'], bins=30, kde=True, color='blue', label='Исходный')
# Гистограмма после обработки
sns.histplot(df_cleaned['dire_player_1_networth'], bins=30, kde=True, color='red', label='После обработки')
plt.title('Распределение данных до и после обработки пропусков')
plt.xlabel('Значение признака')
plt.ylabel('Частота')
plt.legend()
#plt.show()
# 5. Проверка датасета на наличие выбросов и удаление аномальных записей
# Проверка выбросов на примере числового параметра 'dire_player_1_networth'
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_cleaned['dire_player_1_networth'], color='green')
plt.title('Проверка выбросов для параметра "dire_player_1_networth"')
plt.xlabel('dire_player_1_networth')
#plt.show()

# Удаление выбросов
Q1 = df_cleaned['dire_player_1_networth'].quantile(0.25)
Q3 = df_cleaned['dire_player_1_networth'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df_cleaned[(df_cleaned['dire_player_1_networth'] > lower_bound) & (df_cleaned['dire_player_1_networth'] < upper_bound)]

# 6.
print(df_cleaned.dtypes[df_cleaned.dtypes == 'string[pyarrow]'])
df_encoded = pd.get_dummies(df_cleaned, columns=['league'])

df_encoded.to_parquet('processed_dota2_matches_encoded.parquet', index=False)
