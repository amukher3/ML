import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

df=pd.read_csv("C:/Users/Abhishek Mukherjee/Downloads/heart-disease-uci/heart.csv")

df.describe()
#df.info()

df.isnull().sum()
#df.isna().sum()

df.sex.value_counts()
df.groupby('sex').target.value_counts()


#For finding the relation between Sex and the target label
df[['sex', 'target']].groupby(['sex'], as_index=False).mean()
sns.barplot(x='sex', y='target', data=df)

df[['fbs', 'target']].groupby(['fbs'], as_index=False).mean()
sns.barplot(x='fbs',y='target',data=df)

df[['restecg', 'target']].groupby(['restecg'], as_index=False).mean()
sns.barplot(x='restecg',y='target',data=df)

df[['cp','target']].groupby(['cp'],as_index=False).mean()
sns.barplot(x='cp',y='target',data=df)

df[['exang','target']].groupby(['exang'],as_index=False).mean()
sns.barplot(x='exang',y='target',data=df)

df[['thal','target']].groupby(['thal'],as_index=False).mean()
sns.barplot(x='thal',y='target',data=df)

df[['slope','target']].groupby('slope',as_index=False).mean()
sns.barplot(x='slope',y='target',data=df)

sns.factorplot('sex','target', hue='restecg', size=4, aspect=2, data=df)

sns.factorplot(x='cp', y='target', hue='sex', col='restecg', data=df)

sns.factorplot(x='fbs', y='target', hue='sex', col='restecg', data=df)


from sklearn.model_selection import train_test_split

X=df.iloc[:,:-1]

Y=df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)

from sklearn.svm import SVC, LinearSVC

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred_linear_svc = clf.predict(X_test)
accuracy = round(clf.score(X_train, y_train) * 100, 2)

print (accuracy)

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 3)
clf.fit(X_train, y_train)
y_pred_knn = clf.predict(X_test)
acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print (acc_knn)





