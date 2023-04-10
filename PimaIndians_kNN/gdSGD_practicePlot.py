import random
import numpy as np
import matplotlib.pyplot as plt
from Problem5 import *

def bgd_plot(data, y, w, eta, delta, lam, num_iter):
    w, f = bgd_l2(data, y, w, eta, delta, lam, num_iter)

    plt.figure(figsize=(11, 6))  # (width, height)
    plt.plot(f,color = "blue")
    plt.title("bgd_plot(data,y,w,$\eta$={}, $\delta$={}, $\lambda$={},num_inter={} )".format(eta, delta, lam, num_iter))
    plt.xlabel('Iteration Number')
    plt.ylabel('Objective function')
    plt.show()


def sgd_plot(data, y, w, eta, delta, lam, num_iter):
    w, f = sgd_l2(data, y, w, eta, delta, lam, num_iter)

    plt.figure(figsize=(11, 6))  # (width, height)
    plt.plot(f, color = "blue")
    plt.title("sdg_plot(data,y,w,$\eta$={}, $\delta$={}, $\lambda$={},num_inter={} )".format(eta, delta, lam, num_iter))
    plt.xlabel('Iteration Number')
    plt.ylabel('Objective function')
    plt.show()

if __name__ == '__main__':
    # Put the code for the plots here, you can use different functions for each
    # part
    df = np.load('data.npy') #load data (matrix 100 by 2)

    a = np.ones((100,)) # vector 100 by 1 (100rows 1 column)
    x = df[:, 0] #x is the first column of df (100rows 1 column)
    y = df[:, 1] # y is th e second column --> target values (100 rows 1 column)

    n_dat = np.column_stack((x, a)) #add 100 x 1 vector of ones to x vector(100 x 1 to 100 x 2) & get rid of intercept(b) by adding a feature of value 1 to every data point and increase the number of parameters by one.

    w = np.array(([0, 0]), dtype=float) # initialize a 1x2 array of zeroes as [0, 0]

    #GIVEN:
    #bgd_plot(data, y, w, eta, delta, lambda, num_iter)
    bgd_plot(n_dat, y, w, 0.05, 0.1, 0.001, 50)
    bgd_plot(n_dat, y, w, 0.1, 0.01, 0.001, 50)
    bgd_plot(n_dat, y, w, 0.1, 0, 0.001, 100)
    bgd_plot(n_dat, y, w, 0.1, 0, 0, 100)

    # sgd_plot(data, y, w, eta, delta, lambda, num_iter)
    sgd_plot(n_dat, y, w, 1, 0.1, 0.5, 800)
    sgd_plot(n_dat, y, w, 1, 0.01, 0.1, 800)
    sgd_plot(n_dat, y, w, 1, 0, 0, 40)
    sgd_plot(n_dat, y, w, 1, 0, 0, 800)
