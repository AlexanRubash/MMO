import numpy as np

случайный_массив = np.random.randint(1, 10, size=(4, 5))

print("Случайный Массив:")
print(случайный_массив)

массив1, массив2 = np.split(случайный_массив, 2)

print("\nМассив1:")
print(массив1)
print("\nМассив2:")
print(массив2)

целевое_значение = int(input("\nВвведите искомое значение:"))
найденные_элементы = np.where(массив1 == целевое_значение)
print(f"\nИндексы элементов, равных {целевое_значение} в Массиве1:")
for row, col in zip(*найденные_элементы):
    print(f"({row}, {col})")


количество_найденных_элементов = len(найденные_элементы[0])
print(f"Количество элементов, равных {целевое_значение}: {количество_найденных_элементов}")

print("\n--------------------Часть 2 Пандас------------------")
import pandas as pd
import pyarrow as ar

# Создание объекта Series из массива NumPy
numpy_array = np.array([1, 2, 3, 4, 5])
series_data = pd.Series(numpy_array)

print("Series:")
print(series_data)

series_result = series_data * 2

print("\nРезультат мат операции над Series:")
print(series_result)

# объект DataFrame из массива numpy
dataframe_data = случайный_массив
df = pd.DataFrame(dataframe_data)

print("\nDataFrame:")
print(df)

df.columns = ['A', 'B', 'C', 'D', 'E']

print("\nпосле добавления строки заголовков DataFrame:")
print(df)

df = df.drop(1)  # Удаление строки с индексом 1
df = df.drop('B', axis=1)  # Удаление столбца 'B'

print("\nпосле удаления строки и столбца:")
print(df)

df_shape = df.shape
print(f"\nразмер получившегося DataFramee: {df_shape}")

target_value_df = int(input("\nВведите целевое число для поиска в DataFrame: "))
found_elements_df = df[df == target_value_df]

print(f"{found_elements_df}")
