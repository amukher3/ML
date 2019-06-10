#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

#loading the data
df=pd.read_csv("C:/Users/Abhishek Mukherjee/Downloads/breast-cancer.data.csv")
#reading the head
df.head()

#checking for the number of different values
df.iloc[:,0].value_counts()

# label encoding the target
label_target = LabelEncoder()
df.iloc[:,0]= label_target.fit_transform(df.iloc[:,0])

#checking for null and nan values
df.isnull().sum()
df.isna().sum()

df.iloc[:,1].unique()
df.iloc[:,1].value_counts()

df.iloc[:,2].unique()
df.iloc[:,2].value_counts()

#one hot encoding for the feature variable
one_hot = pd.get_dummies(df.iloc[:,1])
df=pd.concat([df, one_hot], axis=1)

# dropping to avoid dummy variable trap and dropping
# the feature variable
df.drop('30-39',axis=1,inplace=True)

#one hot encoding for the feature variable
one_hot = pd.get_dummies(df.iloc[:,1])
df=pd.concat([df, one_hot], axis=1)

# dropping to avoid dummy variable trap and dropping
# the feature variable
df.drop('premeno',axis=1,inplace=True)

df.iloc[:,1].unique()

#one hot encoding for the feature variable
one_hot = pd.get_dummies(df.iloc[:,1])
df=pd.concat([df, one_hot], axis=1)

# dropping to avoid dummy variable trap and dropping
# the feature variable
df.drop('30-34',axis=1,inplace=True)

df.iloc[:,1].unique()

#one hot encoding for the feature variable
one_hot = pd.get_dummies(df.iloc[:,1])
df=pd.concat([df, one_hot], axis=1)

# dropping to avoid dummy variable trap and dropping
# the feature variable
df.drop('0-2',axis=1,inplace=True)

df.iloc[:,1].unique()

# Replacing the "?" values with NaN
df=df.replace("?", np.nan)
df.iloc[:,1].unique()

#Replacing NaN values with
df=df.fillna(method='backfill')
df.iloc[:,1].unique()

#Label encoding the binary values
label_binary = LabelEncoder()
df.iloc[:,1]= label_binary.fit_transform(df.iloc[:,1])
df.iloc[:,1].unique()

df.iloc[:,1].unique()

#one hot encoding for the feature variable
one_hot = pd.get_dummies(df.iloc[:,2])
df=pd.concat([df, one_hot], axis=1)

# dropping to avoid dummy variable trap and dropping
# the feature variable
df.drop('3',axis=1,inplace=True)
df.drop(3,axis=1,inplace=True)

#Label encoding the binary values
label_binary = LabelEncoder()
df.iloc[:,2]= label_binary.fit_transform(df.iloc[:,2])
df.iloc[:,2].unique()


df.iloc[:,3].unique()
#one hot encoding for the feature variable
one_hot = pd.get_dummies(df.iloc[:,3])
df=pd.concat([df, one_hot], axis=1)

# dropping to avoid dummy variable trap and dropping
# the feature variable
df.drop('left_low',axis=1,inplace=True)

#Label encoding the binary values
label_binary = LabelEncoder()
df.iloc[:,3]= label_binary.fit_transform(df.iloc[:,3])

df.iloc[:,3].unique()

#Separating into X and y (features and labels)
X=df.iloc[:,1:]
y=df.iloc[:,1]

#Separating into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# SVM classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

#accuracy of the classifier
acc = classifier.score(X_test,y_test)
print("Accuracy of SVM classifier is:",acc)

#Random forest classifier #
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
#accuracy of the classifier
acc = rfc.score(X_test,y_test)
print("Accuracy of Random forest classifier is:",acc)

pred_rfc = rfc.predict(X_test)



