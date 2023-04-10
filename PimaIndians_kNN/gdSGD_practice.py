import math
import random
import numpy as np
import pandas as pd

#----------------------------------HELPER FUNCTION----------------------------------------------------


def ssl(data, y, w, delta): #SSL from the given function.
    func = 0
    for j in range(0, 100):
        wx = (w[0] * data[j][0]) + (w[1] * data[j][1])
        if y[j] >= wx + delta:
            func += (y[j] - wx - delta) ** 2
        elif abs(y[j] - wx) < delta:
            func += 0
        elif y[j] <= wx - delta:
            func += (y[j] - wx + delta) ** 2
    return (func / 100)

def apply_sgd(data, y, w, eta, delta, lam, i, j):
    w0 = 0
    w1 = 0
    wx = (w[0] * data[i][0]) + (w[1] * data[i][1]) # data[i][1] always has value 1
    if y[i] >= wx + delta:
        w0 += -2 * data[i][0] * (y[i] - wx - delta) #derivative of (y[i] - wx - delta)^2 by w0
        w1 += -2 * (y[i] - wx - delta)  #derivative of (y[i] - wx - delta)^2 by w1
    elif abs(y[i] - wx) < delta:
        w0 += 0
        w1 += 0
    elif y[i] <= wx - delta:
        w0 += -2 * data[i][0] * (y[i] - wx + delta)
        w1 += -2 * (y[i] - wx + delta)

    w0 += 2 * lam * w[0]
    w1 += 2 * lam * w[1]
    w[0] -= ((eta / math.sqrt(j+1)) * w0)  #learning rate --> (j+1) is t^th iteration .
    w[1] -= ((eta / math.sqrt(j+1)) * w1)

    return w
#--------------------------------------------------------------------------------------------------------



def bgd_l2(data, y, w, eta, delta, lam, num_iter):
# should return new weight vector, history of the value of objective function after each iteration (list).
# eta is the learning rate.
    history_fw= []
    for j in range(0, num_iter):
        #initialize a 1x2 array of zeroes as w ([0, 0]). w0,w1 zeroes
        w0 = 0
        w1 = 0
        for i in range(0, 100):
            wx = (w[0] * data[i][0]) + (w[1] * data[i][1])
            if y[i] >= wx + delta:
                w0 += -(2/100) * data[i][0] * (y[i] - wx - delta)  #derivative of (y[i] - wx - delta)^2 by w0. 100 data points
                w1 += -(2/100) * (y[i] - wx - delta)  #derivative of (y[i] - wx - delta)^2 by w1.
            elif abs(y[i] - wx) < delta:
                w0 += 0
                w1 += 0
            elif y[i] <= wx - delta:
                w0 += -(2/100) * data[i][0] * (y[i] - wx + delta)
                w1 += -(2/100) * (y[i] - wx + delta)

        w0 += 2 * lam * w[0]
        w1 += 2 * lam * w[1]
        w[0] -= (eta * w0)
        w[1] -= (eta * w1)

        reg = lam * ((w[0] ** 2) + (w[1] ** 2))  # lambda(sum of w^2)
        history_fw.append(ssl(data, y, w, delta) + reg) #add functions after each iteration

    new_w = [w[0], w[1]]

    return new_w, history_fw


def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
# should return new weight vector, history of the value of objective function after each iteration (list).
# eta / sqrt(t) is the learning rate.
    history_fw = []
    if (i == -1): #apply normal SGD (randomly select the data point) which runs for num_iter
        for j in range(0, num_iter):
            i = random.randint(0, 99) #100 rows. randomly select data point
            w = apply_sgd(data, y, w, eta, delta, lam, i, j)

            reg = lam * ((w[0] ** 2) + (w[1] ** 2))  # lambda(sum of w^2)
            history_fw.append(ssl(data, y, w, delta) + reg)

    else: #SGD for that specific data point (in this case, the num_iter will be 1!)
        w = apply_sgd(data, y, w, eta, delta, lam, i, 0)

        reg = lam * ((w[0] ** 2) + (w[1] ** 2))  # lambda(sum of w^2)
        history_fw.append(ssl(data, y, w, delta) + reg)

    new_w = [w[0], w[1]]

    return new_w, history_fw

