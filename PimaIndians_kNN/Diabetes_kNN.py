# Pima Indians Diabetes dataset from the UCI repository to experiment with the KNN algorithm and find the optimal value for the number of neighbors.

###### IMPORTS
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 


###### Download Pima.csv data file and load it using pandas.

#Add header to data file. 
data = pd.read_csv('Pima.csv',names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "oneTargOutput"])


X = data.drop('oneTargOutput',axis = 1) # identify only featurs as X
y = data.oneTargOutput  #identify only label(target output) as y 


#static of each feature (only: min, max, avg)
statsX = X.describe().loc[['min','max','mean']] 
print ("Statics (min, max, average(mean)) of each features: \n\n ", statsX,"\n\n")


# histogram of the label (target outputs)
hist_y= y.hist() #data.hist(column=["oneTargOutput"])
print ("The histogram of the label (target output): \n\n",hist_y)
