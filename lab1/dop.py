import pandas as pd


df = pd.read_parquet('dota2_matches.parquet')
hero_position_df = pd.DataFrame()

for i in range(1, 6):
    position_col = f'dire_player_{i}_position'
    hero_col = f'dire_player_{i}_hero'
    temp_df = df[[position_col, hero_col]]
    temp_df.columns = ['position', 'hero']
    hero_position_df = pd.concat([hero_position_df, temp_df])

# Убедимся, что позиция не является пустой строкой
hero_position_df = hero_position_df[hero_position_df['position'] != '']

#  Подсчет количества игр для каждого героя на каждой позиции
hero_position_counts = hero_position_df.groupby(['position', 'hero']).size().reset_index(name='count')

# Нахождение пятерки самых популярных героев для каждой позиции
popular_heroes = hero_position_counts.groupby('position').apply(lambda x: x.nlargest(20, 'count')).reset_index(drop=True)

# Результат
pd.set_option('display.max_rows', None)
print(popular_heroes)
