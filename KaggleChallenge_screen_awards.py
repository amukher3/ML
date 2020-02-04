# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 20:46:26 2020

@author: abhi0
"""

import pandas as pd
df=pd.read_csv("C:/Users/abhi0/Downloads/screen-actors-guild-awards/screen_actor_guild_awards.csv")

df.drop(['full_name'],axis=1,inplace=True)
#df.drop(['year'],axis=1,inplace=True)
df.dropna(inplace=True)

#df['year'].fillna('1973', inplace = True) 
#df['show'].fillna('BOMBSHELL', inplace = True) 

df.head()

#Getting dummies for 'category' feature
df_dummies=pd.get_dummies(df['category'],prefix='category')
df=pd.concat([df,df_dummies],axis=1)

df=df.drop(['category'],axis=1)

df.head()

#Getting dummies for 'year' feature
df_dummies=pd.get_dummies(df['year'],prefix='award_category and year')
df=pd.concat([df,df_dummies],axis=1)
#
df=df.drop(['year'],axis=1)

df.head()

#Getting dummies for 'show' feature
df_dummies=pd.get_dummies(df['show'],prefix='show_categories')
df=pd.concat([df,df_dummies],axis=1)

df=df.drop(['show'],axis=1)

#label encoding the 'wining' feature 
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
df['won'] = labelencoder_X.fit_transform(df['won'])

############### Separating the traning set into train and dev sets ##################
    
#For the training data frame separating into dependent and independednt variables. 
#Further,separating the dependednt variable into into training and dev.set
#25-75 ratio adapted. 
    
#Separating the independent variable:
Y=df['won']

X=df.drop(['won'],axis=1)
    
#### Splitting the datasets
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size = 0.5, random_state = 0) 

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.layers import Dropout

################################# training the classifier ##################################
    
#Parameters arrived for
#grid search.
#Activation=['softmax','sigmoid']
Activation=['sigmoid']
#Optimizer=['adam','rmsprop','adagrad','sgd']
Optimizer=['rmsprop']
#BatchSize=[10,20,30,40,50,60]
BatchSize=[40]
#ActivityRegularizer=[0.1,0.5,0.01,0.05,0.001,0.005,0.0001]
#ActivityRegularizer=[0.1,0.01,0.001,0.0001]
ActivityRegularizer=[1e-30]
auc_dev_set=[]
auc_train_set=[]   
iVal=[]
jVal=[]
kVal=[]
opVal=[] 
for i in BatchSize:
   for  j in Activation:
       for k in ActivityRegularizer:
           for op in Optimizer:
               
               # Initialising the ANN
               classifier = Sequential()
              

                # Adding the input layer and the first hidden layer
               classifier.add(Dense(units =X_train.shape[1], kernel_initializer =keras.initializers.he_normal(seed=None), activation = 'relu', input_dim = X_train.shape[1],
                                activity_regularizer=l2(k)))
               
               classifier.add(Dropout(0.4))
            

                # Adding the second hidden layer
               classifier.add(Dense(units =round(X_train.shape[1]/2),kernel_initializer =keras.initializers.he_normal(seed=None), activation = 'relu',
                                activity_regularizer=l2(k)))
               
               classifier.add(Dropout(0.4))

                # Adding the output layer
               classifier.add(Dense(units = 1,kernel_initializer =keras.initializers.he_normal(seed=None), activation =j,activity_regularizer=l2(k)))
             
               
                # Compiling the ANN
               classifier.compile(optimizer = op, loss = 'binary_crossentropy', metrics = ['accuracy'])

                # Fitting the ANN to the Training set
               classifier.fit(X_train, y_train, batch_size = i, epochs = 50)
               
               ######################### checking for the variance trade-off ###############################
               
               #Predictions on the train set
               y_pred_train=classifier.predict(X_train)
               
               #AUC on the train-set
               from sklearn.metrics import roc_curve, auc
               fpr, tpr, _ = roc_curve(y_train, y_pred_train)
               auc_train_set.append(auc(fpr, tpr))
               print(auc_train_set)
               
               iVal.append(i)
               jVal.append(j)
               kVal.append(k)
               opVal.append(op)
               
               
               ##################### Making the predictions and evaluating the model ######################

               # Predicting the Test set results
               y_pred_dev = classifier.predict(X_dev)

               #AUC on the dev-set
               from sklearn.metrics import roc_curve, auc
               fpr, tpr, _ = roc_curve(y_dev, y_pred_dev)
               auc_dev_set.append(auc(fpr, tpr))
               print(auc_dev_set)
#    
    