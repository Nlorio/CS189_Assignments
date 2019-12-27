
# coding: utf-8

# In[61]:

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import scipy.io as spio
import pandas as pd
import itertools
from scipy.special import comb



plot_col = ['r', 'g', 'b', 'k', 'm']
plot_mark = ['o', '^', 'v', 'D', 'x', '+']

# Plots the rows in 'ymat' on the y-axis vs. 'xvec' on the x-axis
# with labels 'ylabels'
# and saves figure as pdf to 'dirname/filename' 
def plotmatnsave(ymat, xvec, ylabels, dirname, filename):
    no_lines = len(ymat)
    fig = plt.figure(0)

    if len(ylabels) > 1:
        for i in range(no_lines):
            xs = np.array(xvec)
            ys = np.array(ymat[i])
            plt.plot(xs, ys, color = plot_col[i % len(plot_col)], lw=1, label=ylabels[i])
        
        lgd = plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

    savepath = os.path.join(dirname, filename)
    plt.xlabel('$x$', labelpad=10)
    plt.ylabel('$f(x)$', labelpad=10)
    plt.savefig(savepath, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

# Sets the labels
labels = ['$e^x$', '1st order', '2nd order', '3rd order', '4th order']

# TODO: Given x values in "x_vec", save the respective function values e^x,
# and its first to fourth degree Taylor approximations
# as rows in the matrix "y_mat"

#scipy.interpolate.approximate_taylor_polynomial(np.exp(x), 0, 0) 
xvex = np.arange(-20,8, 1)
y_mat = [[np.exp(x) for x in xvex], 
        [1 + x for x in xvex],
       [1 + x + (x**2)/2 for x in xvex],
       [1 + x + (x**2)/2 + (x**3)/6 for x in xvex],
       [1 + x + (x**2)/2 + (x**3)/6 + (x**4)/24 for x in xvex]]



# Define filename, invoke plotmatnsave
filename = 'approx_plot_2.pdf'
plotmatnsave(y_mat, xvex, labels, '.', filename)


# In[ ]:




# In[135]:


# There is numpy.linalg.lstsq, which you should use outside of this classs
def lstsq(A, b):
    return np.linalg.solve(A.T @ A, A.T @ b)

def fit(x, y, d):
    X = np.array([[x_t**i for i in range(d)] for x_t in x])
    return X, lstsq(X, y)

data = spio.loadmat('1D_poly.mat', squeeze_me=True)
x_train = np.array(data['x_train'])
y_train = np.array(data['y_train']).T

n = 20  # max degree

err = np.zeros(n - 1)

for i in range(n - 1):
    X, w = fit(x_train, y_train, i)
    err[i] = np.sum(np.square(y_train - X.dot(w))) / len(y_train)
                                        
plt.plot(err)
plt.xlabel('Degree of Polynomial')
plt.ylabel('Training Error')
plt.show()



# # 5.c

# Average training error decreases as a function of D. Increasing the degree of the polynomial function used to fit the training data decreases the average error. Once the degree hits d-1, 1 - the number of data points used by our model, our model predicts the training values perfectly/exactly. We will not be able to fit a polynomial of degree n with the standad matrix inversion method as this will have a non-trivial null space. 

# # 5.d

# In[138]:

data = spio.loadmat('1D_poly.mat', squeeze_me=True)
x_train = np.array(data['x_train'])
y_train = np.array(data['y_train']).T
y_fresh = np.array(data['y_fresh']).T

n = 20  # max degree
err_train = np.zeros(n - 1)
err_fresh = np.zeros(n - 1)

for i in range(n - 1):
    X, w = fit(x_train, y_train, i)
    err_train[i] = np.sum(np.square(y_train - X.dot(w))) / len(y_train)

for i in range(n - 1):
    X, w = fit(x_train, y_fresh, i)
    err_fresh[i] = np.sum(np.square(y_fresh - X.dot(w))) / len(y_fresh)

plt.figure()
plt.ylim([0, 6])
plt.plot(err_train, label='train')
plt.plot(err_fresh, label='fresh')
plt.legend()
plt.show()


# This plot shows how our model predictions for fresh are less accurate than they were for the training data. This is due to the fact that as the degree of the polynomial used to fit the data increases, the predictions become more complex and more tailored specifically to the training values. There is a tradeoff between reducing the error we get in the training data and the generalizability of the model to other data that is not part of the training set. 

# # 5.e

# "Peach lady" should use the model which best minimizes errors for the fresh data. This occurs when a 5 degree polynomial is used to fit the data. Beyond this degree the model becomes overfitted to the training data. The complexity serves to hamper the effectiveness of our model for fresh data. 

# # 5.F 5.G

# In[ ]:




# In[57]:

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6


def gen(D,k): 
    " Creates the powers of all multivariate conditions with k variables and max degree n.\n",
    " Code from karakfa from\n",
    " https://stackoverflow.com/questions/37711817/generate-all-possible-outcomes-of-k-balls-in-n-bins-sum-of-multinomial-catego\n",
    " \"\"\"\n"
    if(k==1):
        return [[D]]
    if(D==0):
        return [[0]*k]
    return [ g2 for x in range(D+1) for g2 in [ u+[D-x] for u in gen(x,k-1) ] ]

def multivariate(data_x, D):
    if D == 0:
        return [1]
    X = []
    for i in range(len(data_x)):
        X.append(multivariate_row(data_x[i], D))
    return X


def multivariate_row(row, D):
    data_row = [1]
    for x in row:
        data_row.append(x)
    data_row = np.array(data_row)
    
    powers = gen(D, len(data_row))
    result = []
    for j in range(len(powers)):
        result.append(np.prod(data_row**powers[j]))
    return result

#multivariate_features = multivariate(data_x, 2)
D_multivariate_features = []
for k in range(KD):
    D_multivariate_features.append(multivariate(data_x, k))
    
    


# In[146]:

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']



def lstsqMULTIVARIATE(A, b, lambda_):
    return np.linalg.solve((A.T @ A) + lambda_*np.identity(len(A[0])), A.T @ b)
        
X_train_1 = D_multivariate_features[k][:250]
X_test_1 = D_multivariate_features[k][250:]
Y_train_1 = data_y[:250]
Y_test_1 = data_y[250:]
#lambda_ = 0.1
#w = lstsqMULTIVARIATE(np.vstack(X_train_1), Y_train_1, lambda_)
#np.array(X_train_1).dot(w)
#D_multivariate_features[2][:50]




# In[148]:

data = spio.loadmat('polynomial_regression_samples.mat', squeeze_me=True)
data_x = data['x']
data_y = data['y']
Kc = 4  # 4-fold cross validation
KD = 6  # max D = 6
LAMBDA = [0, 0.05, 0.1, 0.15, 0.2]

def lstsqMULTIVARIATE(A, b, lambda_):
    return np.linalg.solve((A.T @ A) + lambda_*np.eye(len(A[0])), A.T @ b)

#def fit(D, lambda_):    

#    err_train = np.zeros(Kc)
#    err_test = np.zeros(Kc)
#    for k in range(Kc):
#        
#        X_train = D_multivariate_features[k][k*250:250*(k+1)]
#        X_test = np.append(D_multivariate_features[k][(k+1)*250:], (D_multivariate_features[k][:250*k]))
#        Y_train = data_y[k*250:250*(k+1)]
#        Y_test = np.append(data_y[(k+1)*250:], (D_multivariate_features[k][:250*k]))
#
#       w = lstsqMULTIVARIATE(np.vstack(X_train), Y_train, lambda_)
#        print('hi mark')
#        err_train[k] = np.sum(np.square((Y_train - np.array(X_train).dot(w)))) / len(Y_train)
#        err_test[k] = np.sum(np.square((Y_test - np.array(X_test).dot(w)))) / len(Y_test)
#    return np.sum(err_train) / len(err_train), np.sum(err_test) / len(err_test)


def fit(D, lambda_):    

    err_train = np.zeros(Kc)
    err_test = np.zeros(Kc)
    for k in range(Kc):
        np.random.shuffle(D_multivariate_features[k])
        np.random.shuffle(data_y)
    
        X_train = D_multivariate_features[k][250:]
        X_test = D_multivariate_features[k][:250]
        
        Y_train = data_y[250:]
        Y_test = data_y[:250]

        w = lstsqMULTIVARIATE(np.vstack(X_train), Y_train, lambda_)
        err_train[k] = np.sum(np.square((Y_train - np.array(X_train).dot(w)))) / len(Y_train)
        err_test[k] = np.sum(np.square((Y_test - np.array(X_test).dot(w)))) / len(Y_test)
    return np.sum(err_train) / len(err_train), np.sum(err_test) / len(err_test)


    

np.set_printoptions(precision=11)
Etrain = np.zeros((KD, len(LAMBDA)))
Evalid = np.zeros((KD, len(LAMBDA)))
for D in range(KD):
    print(D)
    for i in range(len(LAMBDA)):
        Etrain[D, i], Evalid[D, i] = fit(D + 1, LAMBDA[i])

print('Average train error:', Etrain, sep='\n')
print('Average valid error:', Evalid, sep='\n')

# YOUR CODE to find best D and i



# Best D for 5.f for lambda = 0.1 appears to be D = 4. This is the case as the average error for the test/valid is smallest here. 

# In[ ]:




# In[ ]:



