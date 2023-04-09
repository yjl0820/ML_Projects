# Pima Indians Diabetes dataset from the UCI repository to experiment with the KNN algorithm and find the optimal value for the number of neighbors.

######IMPORTS
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
