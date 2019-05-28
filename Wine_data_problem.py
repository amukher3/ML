import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
sns.set()

df=pd.read_csv("C:/Users/Abhishek Mukherjee/Downloads/winequality-red.csv")
df.head()
df.info()
df.describe()

#For finding the relation between fixed acidity and quality using group by
df[['fixed acidity', 'quality']].groupby(['quality'], as_index=False).mean()
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)

#For finding the relation between citric acid and quality using group by
# Citric acid content increases with quality
df[['citric acid', 'quality']].groupby(['quality'], as_index=False).mean()
sns.barplot(x='quality',y='citric acid',data=df)

#For finding the relation between sulphates and quality using group by
# Sulphate content also increases with quality
df[['sulphates', 'quality']].groupby(['quality'], as_index=False).mean()
sns.barplot(x='quality',y='sulphates', data=df)

#For finding the relation between alchohol and quality using group by
df[['alcohol', 'quality']].groupby(['quality'], as_index=False).mean()
sns.barplot(x='quality',y='alcohol',data=df)

#For finding the relation between quality and sulphut dioxide
df[['total sulfur dioxide', 'quality']].groupby(['quality'], as_index=False).mean()
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data =df)

#binning the data into categories
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)

#label encoder for label encoding the categories
label_quality = LabelEncoder()
df['quality'] = label_quality.fit_transform(df['quality'])

#Count plot for showing the difference in counts
sns.countplot(df['quality'])

#Getting the feature matrix by dropping the quality column
X = df.drop('quality', axis = 1)

#Getting the result
y = df['quality']

#Splitting into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)

#Applying Standard scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Random forest classifier #
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)

#accuracy of the classifier
acc = rfc.score(X_test,y_test)
print("Accuracy of Random forest classifier is:",acc)

#Making test predictions
pred_rfc = rfc.predict(X_test)

#Looking at the performance of the model
print(classification_report(y_test, pred_rfc))

#Confusion matrix 
print(confusion_matrix(y_test, pred_rfc))

# Support Vector classifier #
svc = SVC()
svc.fit(X_train, y_train)

#accuracy of the classifier
acc = svc.score(X_test,y_test)
print("Accuracy of Support Vector classifier is:",acc)

#Making test predictions
pred_svc = svc.predict(X_test)

#Looking at the performance of the model
print(classification_report(y_test, pred_svc))


