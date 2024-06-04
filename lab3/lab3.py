import pandas as pd

df = pd.read_csv('wine.csv')

print("Исходные данные:")
print(df)

df['quality'] = df['quality'].replace({'good': 1, 'bad': 0}).infer_objects(copy=False)

print("\nОбновленные данные:")
print(df)

X = df.drop('quality', axis=1)
Y = df['quality']

from sklearn.model_selection import train_test_split

# Разделение данных на обучающую и тестовую выборки (80% на обучение, 20% на тест)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Проверка размерности разделенных выборок
print("Размер обучающей выборки X_train:", X_train.shape)
print("Размер тестовой выборки X_test:", X_test.shape)
print("Размер обучающей выборки Y_train:", Y_train.shape)
print("Размер тестовой выборки Y_test:", Y_test.shape)
print("-------")

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Модель дерева решений
tree_model = DecisionTreeClassifier(max_depth=10,random_state=42)
tree_model.fit(X_train, Y_train)
tree_model2 = DecisionTreeClassifier(max_depth=5,random_state=42)
tree_model2.fit(X_train, Y_train)

# Модель k-ближайших соседей
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train, Y_train)

# Предсказание значений на тестовой выборке
tree_pred = tree_model.predict(X_test)
tree_pred1 = tree_model.predict(X_train)
tree_pred2 = tree_model2.predict(X_test)
tree_pred21 = tree_model2.predict(X_train)
knn_pred = knn_model.predict(X_test)

# Рассчет точности моделей
tree_accuracy = accuracy_score(Y_test, tree_pred)
tree_accuracy1 = accuracy_score(Y_train, tree_pred1)
tree_accuracy2 = accuracy_score(Y_test, tree_pred2)
tree_accuracy21 = accuracy_score(Y_train, tree_pred21)
knn_accuracy = accuracy_score(Y_test, knn_pred)

print("Точность модели дерева решений:", tree_accuracy)
print("Точность модели дерева решений:", tree_accuracy1)
print("Точность 2 модели дерева решений:", tree_accuracy2)
print("Точность 2 модели дерева решений:", tree_accuracy21)
print("Точность модели k-ближайших соседей:", knn_accuracy)
print("-------")

from sklearn.metrics import confusion_matrix
tree_confusion_matrix = confusion_matrix(Y_test, tree_pred)
print("Матрица ошибок для модели дерева решений:")
print(tree_confusion_matrix)

tree_confusion_matrix2 = confusion_matrix(Y_test, tree_pred2)
print("Матрица ошибок для 2 модели дерева решений:")
print(tree_confusion_matrix2)

# Рассчет матрицы ошибок для модели k-ближайших соседей
knn_confusion_matrix = confusion_matrix(Y_test, knn_pred)
print("Матрица ошибок для модели k-ближайших соседей:")
print(knn_confusion_matrix)

from sklearn.metrics import precision_score, recall_score, f1_score

# Вычисление precision, recall, F1-score для модели дерева решений
tree_precision = precision_score(Y_test, tree_pred)
tree_recall = recall_score(Y_test, tree_pred)
tree_f1 = f1_score(Y_test, tree_pred)

tree_precision2 = precision_score(Y_test, tree_pred2)
tree_recall2 = recall_score(Y_test, tree_pred2)
tree_f12 = f1_score(Y_test, tree_pred2)

# Вычисление precision, recall, F1-score для модели k-ближайших соседей
knn_precision = precision_score(Y_test, knn_pred)
knn_recall = recall_score(Y_test, knn_pred)
knn_f1 = f1_score(Y_test, knn_pred)

# Вывод метрик для каждой модели
print("Метрики для модели дерева решений:")
print("Precision:", tree_precision)
print("Recall:", tree_recall)
print("F1-score:", tree_f1)
print("-------")

print("Метрики для 2 модели дерева решений:")
print("Precision:", tree_precision2)
print("Recall:", tree_recall2)
print("F1-score:", tree_f12)
print("-------")

print("Метрики для модели k-ближайших соседей:")
print("Precision:", knn_precision)
print("Recall:", knn_recall)
print("F1-score:", knn_f1)
print("-------")

# Сравнение моделей и вывод лучшей
if tree_f1 > knn_f1:
    print("Модель дерева решений является лучшей.")
elif tree_f1 < knn_f1:
    print("Модель k-ближайших соседей является лучшей.")
else:
    print("Обе модели одинаково хороши.")
