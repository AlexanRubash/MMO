import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('train.csv')

# Выбор интересующих признаков
selected_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'SalePrice']
df = data[selected_features]
#GrLivArea : жилая площадь над землей, квадратные футы
#OverallQual общее качество материала и отделки
#GarageCars : Размер гаража по вместимости автомобиля

# Построение матрицы корреляций
corr_matrix = df.corr()

# Визуализация тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
sns.pairplot(df)
plt.show()

from sklearn.linear_model import LinearRegression
import numpy as np

# Определение переменных
X = df[['GrLivArea']].values
y = df['SalePrice'].values

# Создание и обучение модели
model = LinearRegression()
model.fit(X, y)

# Предсказание цен
y_pred = model.predict(X)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, r2_score

# Вычисление метрик
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Определение новых переменных
X_multi = df[['GrLivArea', 'OverallQual', 'GarageCars']].values

# Создание и обучение новой модели
model_multi = LinearRegression()
model_multi.fit(X_multi, y)

# Предсказание цен
y_pred_multi = model_multi.predict(X_multi)

# Оценка новой модели
mse_multi = mean_squared_error(y, y_pred_multi)
r2_multi = r2_score(y, y_pred_multi)

print(f'Mean Squared Error (Multiple): {mse_multi}')
print(f'R^2 Score (Multiple): {r2_multi}')

print("Simple Linear Regression Model")
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

print("\nMultiple Linear Regression Model")
print(f'Mean Squared Error: {mse_multi}')
print(f'R^2 Score: {r2_multi}')
