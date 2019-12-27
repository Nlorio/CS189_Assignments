
# coding: utf-8

# In[295]:

import numpy as np
import math
import scipy.io
import random
import sklearn
import numpy.linalg as LA
import time
import matplotlib.pyplot as plt






# ## 4.A V, VI

# In[99]:

A = [[2, -4], [-1, -1]]
B = [[3, 1], [1,3]]


# In[53]:

LA.eig(A)


# In[ ]:

# Do we have to adjust the scale of the eigenvectors based off of the fact that the norm is mentioned to be = 1 ??


# In[48]:

G = np.multiply(A, B)


# In[49]:

LA.eig(G)


# In[29]:

D = np.multiply(B, A)


# In[30]:

LA.eig(D)


# ## 4.B 

# C = U$\Sigma$V$^T$
# 
# Therefore:
# 
# C$^T$C = V$\Sigma$$^T$$\Sigma$V$^T$
# 
# CV = U$\Sigma$

# In[46]:

np.linalg.svd(A)


# In[42]:

np.linalg.svd(np.multiply(A,A))


# In[50]:

#G = AB
np.linalg.svd(G)


# In[43]:

#D = BA
np.linalg.svd(D)


# In[52]:

C = [[3, 1], [1, 3], [2, -4], [-1, -1]]
np.linalg.svd(C)


# ## 6.a

# In[283]:

mdict = scipy.io.loadmat("a.mat")

x = mdict['x']
u = mdict['u']

X = np.transpose([x[0][:-1], u[0][:-1]]) # Features Matrix, 2 features x, u

a, b = np.linalg.lstsq(X, x[0][1:])[0]
print("A is ", a)
print("B is ", b)


# In[ ]:




# # 6.b

# In[291]:

mdict = scipy.io.loadmat("b.mat")

x = mdict['x']
u = mdict['u']

x1 = []
x2 = []
x3 = []
u1 = []
u2 = []
u3 = []


for i in range(len(x)):
    x1.append(x[i][0][0])
    x2.append(x[i][1][0])
    x3.append(x[i][2][0])
    u1.append(u[i][0][0])
    u2.append(u[i][1][0])
    u3.append(u[i][2][0])

X = np.transpose([x1[:-1], x2[:-1], x3[:-1], u1[:-1], u2[:-1], u3[:-1]])

truex = np.transpose([x1[1:], x2[1:], x3[1:]])

D = np.linalg.lstsq(X, truex)[0]
A = [D[0], D[1], D[2]]
B = [D[3], D[4], D[5]]
print("A is ", A)
print("                ")
print("B is ", B)


# In[ ]:




# In[ ]:




# ## 6.c

# In[292]:

mdict = scipy.io.loadmat("train.mat")

# Assemble xu matrix
x = mdict["x"]   # position of a car
v = mdict["xd"]  # velocity of the car
xprev = mdict["xp"]   # position of the car ahead
vprev = mdict["xdp"]  # velocity of the car ahead

acc = mdict["xdd"]  # acceleration of the car

one = []
for i in range(len(x[0])):
    one.append(1)

a, b, c, d, e = 0, 0, 0, 0, 0

X = np.transpose([x[0], v[0], xprev[0], vprev[0], one])

values = np.linalg.lstsq(X, acc[0])[0]
a = values[0]
b = values[1]
c = values[2]
d = values[3]
e = values[4]



print("Fitted dynamical system:")
print("xdd_i = {:.3f} x_i + {:.3f} xd_i + {:.3f} x_i-1 + {:.3f} xd_i-1 + {:.3f}".format(a, b, c, d, e))
values


# # 6.d

# Why is this reasonable:
# 
# Our results are physically reasonable. Assume the i-1 car is in front of the ith car. 
# 
# $+h(x_{i-1} - x_i)$
# 
# The car in the i-1 place in line will be ahead of car i. If not the positive sign means that the acceleration will be positively influenced. 
# 
# $+f(\dot{x}_{i-1} - \dot{x}_i)$
# 
# The velocity of the car in the i position is kept in check by the velocity of the next car in line, the i - 1 th position. This has a positive coefficient as a positive difference in velocity between the two cars will contribute to the ith car increasing its velocity. 
# 
# $-g(\dot{x}_{i} - L)$
# 
# Speed is kept in check by the speed limit. If the ith car is above the speed limit than the acceleration of the car will be decrease due to the negative sign of this function. 
# 
# 
# 
# $+W_i$
# 
# Extra factor which accounts for situations not covered by the previous functions. These factors are not explicitly stated our modeled but can have an affect on our acceleration. 
# 
# 

# # 7

# In[479]:

# Load the training dataset
train_features = np.load("train_features.npy")
train_labels = np.load("train_labels.npy").astype("int8")

n_train = train_labels.shape[0]

def visualize_digit(features, label):
    # Digits are stored as a vector of 400 pixel values. Here we
    # reshape it to a 20x20 image so we can display it.
    plt.imshow(features.reshape(20, 20), cmap="binary")
    plt.xlabel("Digit with label " + str(label))
    plt.show()

# Visualize a digit
# visualize_digit(train_features[0,:], train_labels[0])

# TODO: Plot three images with label 0 and three images with label 1
visualize_digit(train_features[0,:], train_labels[0])
visualize_digit(train_features[7,:], train_labels[7])
visualize_digit(train_features[400,:], train_labels[400])
visualize_digit(train_features[5,:], train_labels[5])
visualize_digit(train_features[6,:], train_labels[6])
visualize_digit(train_features[3,:], train_labels[3])

# Linear regression

# TODO: Solve the linear regression problem, regressing
# X = train_features against y = 2 * train_labels - 1

X = train_features
y = 2 * train_labels - 1

W = np.linalg.lstsq(X, y)[0]

# TODO: Report the residual error and the weight vector

A = np.dot(X, W) - y
# ||A||_2^2 = A^TA 
error = np.dot(A.T, A)

print("Error is: ", error)
print()
print()
print("Weight Vectors for W shown below")
print()
for i in range(200):
    print(W[i])

print()
print()

# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")

n_test = test_labels.shape[0]

# TODO: Implement the classification rule and evaluate it
# on the training and test set
success_training = 0 
for i in range(len(train_features)):
    if np.dot(train_features[i], W) <= 0:
        if train_labels[i] == 0:
            success_training = success_training + 1 
    else:
        if train_labels[i] == 1:
            success_training = success_training + 1
            
print("Training Success Percentage: ", success_training/len(train_features))

success = 0 
for i in range(len(test_features)):
    if np.dot(test_features[i], W) <= 0:
        if test_labels[i] == 0:
            success = success + 1 
    else:
        if test_labels[i] == 1:
            success = success + 1
            
print("Test Success Percentage: ", success/len(test_features))
  
print()
print()
    
print("7.d Why is the performance typically evaluated on a separate test set (instead of the training set) and why is the performance on the training and test set similar in our case? \ If we evaluate our performance only on our training set than there is the potential to make our model overly complicated, specific. We could build a super complicated model which fits our training data exactly. - Not overly deep, complicated. We want our model to be generalizable to more than just the training data - Our results are similar because our model is gooood.")
      
print()
print()  

# TODO: Try regressing against a vector with 0 for class 0
# and 1 for class 1
      
W = np.linalg.lstsq(train_features, train_labels)[0]
      
success_training = 0 
for i in range(len(train_features)):
    if np.dot(train_features[i], W) <= 0.5:
        if train_labels[i] == 0:
            success_training = success_training + 1 
    else:
        if train_labels[i] == 1:
            success_training = success_training + 1
            
print("01 Training Success Percentage: ", success_training/len(train_features))

success = 0 
for i in range(len(test_features)):
    if np.dot(test_features[i], W) <= 0.5:
        if test_labels[i] == 0:
            success = success + 1 
    else:
        if test_labels[i] == 1:
            success = success + 1
            
print("01 Test Success Percentage: ", success/len(test_features))



# In[470]:


# TODO: Form a new feature matrix with a column of ones added
# and do both regressions with that matrix


# Linear regression 2 (with new feature matrix)

# TODO: Solve the linear regression problem, regressing
# X = train_features against y = 2 * train_labels - 1


ones = np.ones((train_features.shape[0], 1))
ones2 = np.ones((test_features.shape[0], 1))
X = np.hstack((train_features, ones))
X2 = np.hstack((test_features, ones2))

y = 2 * train_labels - 1

W = np.linalg.lstsq(X, y)[0]


# Load the test dataset
# It is good practice to do this after the training has been
# completed to make sure that no training happens on the test
# set!
test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy").astype("int8")

n_test = test_labels.shape[0]

# TODO: Implement the classification rule and evaluate it
# on the training and test set
success_training = 0 
for i in range(len(X)):
    if np.dot(X[i], W) <= 0:
        if train_labels[i] == 0:
            success_training = success_training + 1 
    else:
        if train_labels[i] == 1:
            success_training = success_training + 1
            
print("Xprime Training Success Percentage: ", success_training/len(X))

success = 0 
for i in range(len(X2)):
    if np.dot(X2[i], W) <= 0:
        if test_labels[i] == 0:
            success = success + 1 
    else:
        if test_labels[i] == 1:
            success = success + 1
            
print("Xprime Test Success Percentage: ", success/len(X2))
  
print()
print()

# TODO: Try regressing against a vector with 0 for class 0
# and 1 for class 1
      
W = np.linalg.lstsq(X, train_labels)[0]
      
success_training = 0 
for i in range(len(X)):
    if np.dot(X[i], W) <= 0.5:
        if train_labels[i] == 0:
            success_training = success_training + 1 
    else:
        if train_labels[i] == 1:
            success_training = success_training + 1
            
print("01 Xprime Training Success Percentage: ", success_training/len(X))

success = 0 
for i in range(len(X2)):
    if np.dot(X2[i], W) <= 0.5:
        if test_labels[i] == 0:
            success = success + 1 
    else:
        if test_labels[i] == 1:
            success = success + 1
            
print("01 Xprime Test Success Percentage: ", success/len(X2))





# Logistic Regression       

# You can also compare against how well logistic regression is doing.
# We will learn more about logistic regression later in the course.

#import sklearn.linear_model

#lr = sklearn.linear_model.LogisticRegression()
#lr.fit(X, train_labels)

#test_error_lr = 1.0 * sum(lr.predict(test_features) != test_labels) / n_test



# In[ ]:




# In[ ]:




# In[ ]:



