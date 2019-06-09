import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import cross_val_score

train_data=pd.read_csv("C:/Users/Abhishek Mukherjee/Downloads/train_data_science_london.csv")
train_labels=pd.read_csv("C:/Users/Abhishek Mukherjee/Downloads/trainLabels.csv")
test_data=pd.read_csv("C:/Users/Abhishek Mukherjee/Downloads/test.csv")

train_data.head()

train_data.shape,test_data.shape,train_labels.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(train_data,train_labels, test_size = 0.30, random_state = 101)

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

from sklearn.decomposition import PCA

pca  = PCA(n_components=12)
#x_train = pca.fit_transform(x_train)
#x_test = pca.transform(x_test)
pca_train_data = pca.fit_transform(train_data)
explained_variance = pca.explained_variance_ratio_ 
print(explained_variance)

 NAIBE BAYES
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
#nb_model.fit(pca_train_data,y_train.values.ravel())
#nb_predicted= nb_model.predict(x_norm_test)
#print('Naive Bayes',accuracy_score(y_test, nb_predicted))
print('Naive Bayes',cross_val_score(nb_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver = 'saga')
#lr_model.fit(pca_train_data,y_train.values.ravel())
#lr_predicted = lr_model.predict(x_norm_test)
#print('Logistic Regression',accuracy_score(y_test, lr_predicted))
print('Logistic Regression',cross_val_score(lr_model,pca_train_data, train_labels.values.ravel(), cv=10).mean())


#XGBOOST
from xgboost import XGBClassifier

xgb = XGBClassifier()
#xgb.fit(x_norm_train,y_train.values.ravel())
#xgb_predicted = xgb.predict(x_norm_test)
#print('XGBoost',accuracy_score(y_test, xgb_predicted))
print('XGBoost',cross_val_score(xgb,pca_train_data, train_labels.values.ravel(), cv=10).mean())

#Random Forest Classifier
rfc = RandomForestClassifier(random_state=99)

#USING GRID SEARCH
n_estimators = [10, 50, 100, 200,400]
max_depth = [3, 10, 20, 40]
param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)

grid_search_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 10,scoring='accuracy',n_jobs=-1).fit(gmm_train, train_labels.values.ravel())
rfc_best = grid_search_rfc.best_estimator_
print('Random Forest Best Score',grid_search_rfc.best_score_)
print('Random Forest Best Parmas',grid_search_rfc.best_params_)
print('Random Forest Accuracy',cross_val_score(rfc_best,gmm_train, train_labels.values.ravel(), cv=10).mean())


