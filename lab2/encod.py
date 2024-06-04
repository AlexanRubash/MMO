from numpy import array
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
df_cleaned = pd.read_parquet('processed_dota2_matches.parquet')
# Пример данных
data = df_cleaned['league'].tolist()
print("Исходные данные:")
print(data)

# Преобразование в numpy массив
values_of_seq = array(data)


# Integer Encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values_of_seq)
print("Integer Encoding:")
print(integer_encoded)

# One-Hot Encoding
onehot_encoder = OneHotEncoder()
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


# Получение имен столбцов после One-Hot Encoding
encoded_columns = onehot_encoder.get_feature_names_out(input_features=['league'])

# Поиск индексов столбцов, в которых есть 1
columns_with_ones = [i for i, value in enumerate(onehot_encoded.toarray().T) if 1 in value]

# Вывод значений, которые имеют 1 в соответствующих столбцах
for index in columns_with_ones:
    column_name = encoded_columns[index]
    category_values = df_cleaned['league'].iloc[onehot_encoded.toarray()[:, index] == 1]
    print(f"Values for {column_name}: {category_values.tolist()}")
