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
