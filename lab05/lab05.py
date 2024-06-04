import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data = pd.read_csv("/content/SalaryData.csv");
data.head()

cols = data.columns[:]
colours = ['#eeeeee', '#ff0000']
sns.heatmap(data[cols].isnull(), cmap=sns.color_palette(colours))
for col in data.columns:
    pct_missing = data[col].isnull().sum()
    print('{} - {}'.format(col, round(pct_missing)))
data = data.dropna(subset=['Salary'])
data.dtypes
data['Education Level'].value_counts()
data["Education Level"]=data["Education Level"].map({"Bachelor's": 1, "Master's": 2, "PhD": 3})
data.head()
data['Gender'].value_counts()
data["Gender"]=data["Gender"].map({"Male": 0, "Female": 1})
data.head()
data['Job Title'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['Job Title'] = le.fit_transform(data['Job Title'])
print(data.head())
hm = sns.heatmap(data.corr(),
                 cbar=True,
                 annot=True)

g = sns.PairGrid(data)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)


sns.scatterplot(data=data, x="Years of Experience", y="Age")


X = data['Years of Experience'].values.reshape(-1,1)
y = data['Age'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)
print(reg.coef_[0][0])
print(reg.intercept_[0])

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))
predictions = reg.predict(X)

plt.figure(figsize=(16, 8))
plt.scatter(
    data['Years of Experience'],
    data['Age'],
    )
plt.plot(
    data['Years of Experience'],
    predictions,
    c='red',
    linewidth=2
)
plt.xlabel("Years of Experience")
plt.ylabel("Age")
plt.show()
reg.score(X, y, sample_weight=None) #коэффициент детерминации


Xs = data.drop(['Age'], axis=1)
y = data['Age'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(Xs, y)
print(reg.coef_)
print(reg.intercept_)
print("The linear model is: Y = {:.5} + {:.5}*Salary + {:.5}*Gender + {:.5}*Education Level + {:.5}*Job Title + {:.5}*Years of experience".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1], reg.coef_[0][2], reg.coef_[0][3], reg.coef_[0][4]))

reg.score(Xs, y) #коэффициент детерминации
