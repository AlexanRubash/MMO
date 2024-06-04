import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Загрузка данных
data = pd.read_csv("train.csv")

# Проверка количества параметров
print("Количество параметров:", data.shape[1])

# 2. Обучение модели случайного леса на исходных данных
X = data.drop('price_range', axis=1)
y = data['price_range']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели на исходных данных: {accuracy:.4f}')

# 3. Сокращение количества параметров с использованием VarianceThreshold
selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_reduced = selector.fit_transform(X)

# 4. Обучение модели случайного леса на сокращенном датасете
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_reduced = accuracy_score(y_test, y_pred)
print(f'Точность модели на сокращенных данных: {accuracy_reduced:.4f}')

# 5. Применение PCA к исходному датасету и нахождение 2 главных компонент
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 6. Визуализация данных по двум главным компонентам
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Первая главная компонента')
plt.ylabel('Вторая главная компонента')
plt.title('PCA с двумя компонентами')
plt.colorbar()
plt.show()

# 7. Обучение модели случайного леса на данных с двумя главными компонентами
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_pca_2 = accuracy_score(y_test, y_pred)
print(f'Точность модели на данных PCA с двумя компонентами: {accuracy_pca_2:.4f}')

# 8. Нахождение количества главных компонент для объяснения 90% дисперсии
pca_full = PCA().fit(X)
explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_90 = np.argmax(explained_variance >= 0.90) + 1

plt.plot(explained_variance)
plt.xlabel('Количество главных компонент')
plt.ylabel('Объясненная дисперсия')
plt.title('График зависимости объясненной дисперсии от количества главных компонент')
plt.axvline(n_components_90, color='r', linestyle='--')
plt.show()

print(f'Необходимое количество компонент для объяснения 90% дисперсии: {n_components_90}')

# 9. Обучение модели случайного леса на данных PCA с количеством компонент для 90% дисперсии
pca_90 = PCA(n_components=n_components_90)
X_pca_90 = pca_90.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_pca_90, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_pca_90 = accuracy_score(y_test, y_pred)
print(f'Точность модели на данных PCA с {n_components_90} компонентами: {accuracy_pca_90:.4f}')
