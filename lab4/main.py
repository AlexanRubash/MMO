import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

# Удаление ненужных столбцов ('PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked')
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)

# Преобразование категориального признака 'Sex' в числовой
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Заполнение пропущенных значений в столбце 'Age' медианой
data['Age'].fillna(data['Age'].median(), inplace=True)

# Разделение данных на признаки (X) и классы (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Нормализация числовых признаков
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']] = scaler.fit_transform(X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']])

# Разделение данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Обучение модели с параметром регуляризации C=1
model = LogisticRegression(C=1)
model.fit(X_train, y_train)

# Расчет точности на обучающем и тестовом наборах
print("Правильность на обучающем наборе: {:.2f}".format(model.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(model.score(X_test, y_test)))

# Обучение модели с параметром регуляризации C=100
model_100 = LogisticRegression(C=100)
model_100.fit(X_train, y_train)

# Расчет точности на обучающем и тестовом наборах
print("Правильность на обучающем наборе (C=100): {:.2f}".format(model_100.score(X_train, y_train)))
print("Правильность на тестовом наборе (C=100): {:.2f}".format(model_100.score(X_test, y_test)))

# Обучение модели с параметром регуляризации C=0.01
model_001 = LogisticRegression(C=0.01)
model_001.fit(X_train, y_train)

# Расчет точности на обучающем и тестовом наборах
print("Правильность на обучающем наборе (C=0.01): {:.2f}".format(model_001.score(X_train, y_train)))
print("Правильность на тестовом наборе (C=0.01): {:.2f}".format(model_001.score(X_test, y_test)))

# Вычисление метрик качества для модели с наилучшими результатами (например, C=1)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Метрики качества:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("Матрица ошибок:")
print(confusion_mat)

#2------------------------------------------
print("2------------------------------------------")
# Создание модели SVC
model_SVC = SVC()

# Обучение модели и расчет точности
model_SVC.fit(X_train, y_train)
train_accuracy = accuracy_score(y_train, model_SVC.predict(X_train))
test_accuracy = accuracy_score(y_test, model_SVC.predict(X_test))

print("Точность на обучающем наборе: {:.2f}".format(train_accuracy))
print("Точность на тестовом наборе: {:.2f}".format(test_accuracy))

# Определение оптимальных параметров C и gamma с помощью GridSearchCV
SVC_params = {"C": [0.1, 1, 10], "gamma": [0.2, 0.6, 1]}
SVC_grid = GridSearchCV(model_SVC, SVC_params, cv=5, n_jobs=-1)
SVC_grid.fit(X_train, y_train)

# Вывод наилучшей точности и параметров
print("Наилучшая точность: {:.2f}".format(SVC_grid.best_score_))
print("Наилучшие параметры: ", SVC_grid.best_params_)

# Рассчет метрик качества для наилучшей модели SVC
y_pred = SVC_grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("Метрики качества:")
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("Матрица ошибок:")
print(confusion_mat)

#3----------------------------------
print("3------------------------------------------")

# Создание модели дерева решений
model_decision_tree = DecisionTreeClassifier()

# Обучение модели дерева решений
model_decision_tree.fit(X_train, y_train)

# Расчет точности на обучающем и тестовом наборах для модели дерева решений
train_accuracy_dt = accuracy_score(y_train, model_decision_tree.predict(X_train))
test_accuracy_dt = accuracy_score(y_test, model_decision_tree.predict(X_test))

print("Точность на обучающем наборе (дерево решений): {:.2f}".format(train_accuracy_dt))
print("Точность на тестовом наборе (дерево решений): {:.2f}".format(test_accuracy_dt))

# Создание модели K-ближайших соседей
model_knn = KNeighborsClassifier()

# Обучение модели K-ближайших соседей
model_knn.fit(X_train, y_train)

# Расчет точности на обучающем и тестовом наборах для модели K-ближайших соседей
train_accuracy_knn = accuracy_score(y_train, model_knn.predict(X_train))
test_accuracy_knn = accuracy_score(y_test, model_knn.predict(X_test))

print("Точность на обучающем наборе (K-ближайшие соседи): {:.2f}".format(train_accuracy_knn))
print("Точность на тестовом наборе (K-ближайшие соседи): {:.2f}".format(test_accuracy_knn))

#4-----------------------------------------

# Создание области графика
fig, ax = plt.subplots()

# Построение ROC-кривых для моделей и добавление на один график
roc_display_lr = RocCurveDisplay.from_estimator(model, X_test, y_test, name='Логистическая регрессия', ax=ax)
roc_display_svc = RocCurveDisplay.from_estimator(model_SVC, X_test, y_test, name='Опорные вектора', ax=ax)
roc_display_dt = RocCurveDisplay.from_estimator(model_decision_tree, X_test, y_test, name='Дерево', ax=ax)
roc_display_knn  = RocCurveDisplay.from_estimator(model_knn, X_test, y_test, name='K-n', ax=ax)

# Отображение графика
plt.legend()
plt.show()