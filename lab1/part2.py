import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#2
df = pd.read_parquet('dota2_matches.parquet')

# 3: гистограммы частот
plt.figure(figsize=(10, 6))
sns.histplot(df['dire_player_1_networth'], bins=30, kde=True)
plt.title('Гистограмма частот для dire_player_1_networth')
plt.xlabel('Networth')
plt.ylabel('Частота')
plt.show()

# 4: расчет медианы и среднего значения
median_networth = df['dire_player_1_networth'].median()
mean_networth = df['dire_player_1_networth'].mean()

print(f"Медиана dire_player_1_networth: {median_networth}")
print(f"Среднее значение dire_player_1_networth: {mean_networth}")

# 5: box plot для 'dire_player_1_networth'
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['dire_player_1_networth'])
plt.title('Box plot для dire_player_1_networth')
plt.xlabel('Networth')
plt.show()

# 6:  .describe() к 'dire_player_1_networth'
networth_description = df['dire_player_1_networth'].describe()

print("\nОписание для dire_player_1_networth:")
print(networth_description)

# Шаг 7: Группировка данных и расчет статистик для 'dire_player_1_position'
grouped_data_position = df.groupby('dire_player_1_position')['dire_player_1_networth'].agg(['mean', 'median', 'count'])

print("\nСтатистики по группам (dire_player_1_position):")
print(grouped_data_position)


