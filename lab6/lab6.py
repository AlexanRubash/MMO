import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Загрузка данных
data = pd.read_csv('data.csv')

# 1. Выбор наиболее важных параметров
important_columns = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                    'smoothness_mean', 'compactness_mean', 'concavity_mean',
                    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
X = data[important_columns]

# 2. Проверка на пропуски и кодирование категориальных данных (если требуется)
if X.isnull().sum().sum() > 0:
    X.fillna(0, inplace=True)  # Заменяем пропуски на 0 (или используйте другой метод)

# 3. Нормализация значений в матрице X с помощью MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# 4. Определение оптимального количества кластеров методом локтя для K-means
def plot_elbow(X):
    distortions = []# Создаем пустой список distortions для сохранения значений искажений
    K = range(1, 11)# Создаем список значений k (количество кластеров) от 1 до 10
    for k in K:
        # Создаем объект KMeans с k кластерами и заданным 
        # random_state=42  для воспроизводимости
        kmeans = KMeans(n_clusters=k, random_state=42)
        # Производим кластеризацию данных X с 
        # использованием алгоритма KMeans
        kmeans.fit(X)
        # Добавляем значение искажения (inertia_) 
        # в список distortions
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.show()

plot_elbow(X_normalized)

# 5. Кластеризация методом K-means
# Создается объект модели KMeans с указанным числом кластеров 
kmeans = KMeans(n_clusters=3, random_state=42)
# Метод fit_predict одновременно обучает модель на 
# нормализованных данных X_normalized и возвращает 
# метки кластеров для каждого объекта в наборе данных
clusters_kmeans = kmeans.fit_predict(X_normalized)

# 6. Разделение данных на кластеры методом иерархической кластеризации
# linkage выполняет иерархическую кластеризацию на 
# основе заданных данных.
# X_normalized: Матрица данных, которая подвергается
# кластеризации.
# method='ward': Метод расчета расстояний между 
# кластерами. В данном случае используется метод 
# "ward", который минимизирует дисперсию кластеров 
# при объединении.
# metric='euclidean': Метрика расстояния между 
# точками. Здесь используется евклидово расстояние.
linkage_matrix = linkage(X_normalized, method='ward', metric='euclidean')
plt.figure(figsize=(10, 7))
# строим дендрограмму на основе предварительно 
# вычисленной матрицы связей
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 7. Кластеризация методом иерархической кластеризации
# AgglomerativeClustering - агломеративная кластеризацию
agglomerative = AgglomerativeClustering(n_clusters=3)
clusters_agg = agglomerative.fit_predict(X_normalized)

# 8. Оценка качества кластеризации
# вычисляем Silhouette Score для оценки качества кластеризации
silhouette_kmeans = silhouette_score(X_normalized, clusters_kmeans)
# clusters_kmeans и clusters_agg: Метки кластеров, 
# полученные после кластеризации методами K-means и 
# иерархической кластеризации соответственно.
silhouette_agg = silhouette_score(X_normalized, clusters_agg)
print(f"Silhouette Score (K-means): {silhouette_kmeans}")
print(f"Silhouette Score (Hierarchical): {silhouette_agg}")

# 9. Визуализация результатов кластеризации
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters_kmeans, cmap='viridis', s=50)
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.title('K-means Clustering')
plt.colorbar(label='Cluster')
plt.show()

# 10. Визуализация выбранного объекта в виде точки отличного цвета и размера
chosen_object = X.iloc[0]  # Пример выбора первого объекта
plt.figure(figsize=(10, 6))
# диаграмма рассеяния
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters_kmeans, cmap='viridis', s=50)
plt.scatter(chosen_object.iloc[0], chosen_object.iloc[1], c='purple', marker='o', s=200, label='Chosen Object')
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.title('K-means Clustering with Chosen Object')
plt.colorbar(label='Cluster')
plt.legend()
plt.show()


#====================
# Что такое метод локтя
# Метод является способом определения оптимального
# количества кластеров для алгоритма K-means. 
# Он основан на оценке искажения (distortion) 
# или суммы квадратов расстояний объектов до их 
# центроидов
#
# 1) Для каждого значения k (количество кластеров) 
# от 1 до заданного предела (обычно до 10) выполняется 
# кластеризация данных методом K-means.
# После каждой кластеризации вычисляется значение 
# искажения (inertia_), которое представляет собой 
# сумму квадратов расстояний объектов до их 
# центроидов внутри кластеров.
#
# 2)Для каждого k строится график, на котором по оси 
# X отложено количество кластеров (k), а по оси 
# Y - значение искажения (distortion).
# Обычно на графике получается кривая, напоминающая 
# изогнутую руку (отсюда название "метод локтя").
#
# 3)Оптимальное количество кластеров выбирается как 
# точка "локтя" на графике
# "Локоть" соответствует моменту, когда увеличение 
# числа кластеров (k) начинает оказывать меньший 
# эффект на уменьшение искажения. Обычно это момент, 
# когда изменение наклона кривой становится менее 
# значительным.

#====================
#Что такое метод k-means
# 1)Случайным образом инициализируются 
#   центроиды кластеров
# 2.1) Присваивание каждого объекта 
#   ближайшему центроиду
# 2.2) Пересчет центроидов как среднее 
#    арифметическое всех объектов, принадлежащих 
#    кластеру.
# 3) Процесс обновления кластеров и центроидов 
#    повторяется итеративно до сходимости 
#    (когда изменения становятся незначительными).
# 4) Метод k-means стремится минимизировать сумму 
#    квадратов расстояний объектов до центроидов 
#    кластеров
#====================

#1. Что решают задачи кластеризации в машинном 
#   обучении?
#2. Расскажите принцип работы метода K-means.
#3. Как можно выбрать оптимальное количество 
#   кластеров в K-means?
#4. Расскажите принцип работы метода иерархической 
#кластеризации.
#5. Для чего можно использовать дендрограмму в 
#   методе иерархической кластеризации?
#6. Какие метрики используют для оценки качества 
#   кластеризации
##

# в классификации заранее известны параметры классификации
# в кластеризации параметры высчитываются в момент выполнения

"""
Классификация:
Определение: Классификация относится к задаче обучения с учителем, 
где модель обучается на размеченных данных (с данными, для которых 
известны целевые метки или классы). Цель классификации - предсказать 
категорию или класс новых данных на основе известных признаков.
Пример: Распознавание спама по электронной почте, где целью является 
классификация электронных писем как спама или не спама на основе 
содержания электронного письма.

Кластеризация:
Определение: Кластеризация относится к задаче обучения без учителя, 
где модель пытается выделить группы или кластеры в данных без 
предварительно известных меток или классов. Целью кластеризации 
является выявление внутренних структур или неразмеченных паттернов в данных.
Пример: Группировка потребителей на основе их покупательского 
поведения для выявления схожих групп потребителей.
"""