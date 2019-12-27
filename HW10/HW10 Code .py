
# coding: utf-8

# # Question 3

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## 3.a, 3.b, 3.c

# In[3]:


from numpy.random import uniform
from numpy.random import randn
import random
import time

import matplotlib.pyplot as plt

from scipy.linalg import eig
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import svd

from utils import create_one_hot_label
from utils import subtract_mean_from_data
from utils import compute_covariance_matrix

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import svd
import IPython



class Project2D():

    '''
    Class to draw projection on 2D scatter space
    '''

    def __init__(self,projection, clss_labels):

        self.proj = projection
        self.clss_labels = clss_labels


    def project_data(self,X,Y,white=None):

        '''
        Takes list of state space and class labels
        State space should be 2D
        Labels shoud be int
        '''

        p_a = []
        p_b = []
        p_c = []

        ###PROJECT ALL DATA###
        proj = np.matmul(self.proj,white)

        X_P = np.matmul(proj,np.array(X).T)

        for i in range(len(Y)):

            if Y[i] == 0:
                p_a.append(X_P[:,i])
            elif Y[i] == 1:
                p_b.append(X_P[:,i])
            else:
                p_c.append(X_P[:,i])


        p_a = np.array(p_a)
        p_b = np.array(p_b)
        p_c = np.array(p_c)

        plt.scatter(p_a[:,0],p_a[:,1],label = 'apple')
        plt.scatter(p_b[:,0],p_b[:,1],label = 'banana')
        plt.scatter(p_c[:,0],p_c[:,1],label = 'eggplant')

        plt.legend()

        plt.show()



class Projections():

    def __init__(self,dim_x,classes):

        '''
        dim_x: the dimension of the state space x
        classes: The list of class labels
        '''

        self.d_x = dim_x
        self.NUM_CLASSES = len(classes)


    def get_random_proj(self):

        '''
        Return A which is size 2 by 729
        '''

        return randn(2,self.d_x)


    def pca_projection(self,X,Y):

        '''
        Return U_2^T
        '''

        X,Y= subtract_mean_from_data(X,Y)

        C_XX = compute_covariance_matrix(X,X)

        u,s,d = svd(C_XX)

        return u[:,0:2].T




    def cca_projection(self,X,Y,k=2):

        '''
        Return U_K^T, \Simgma_{XX}^{-1/2}
        '''

        Y = create_one_hot_label(Y,self.NUM_CLASSES)
        X,Y = subtract_mean_from_data(X,Y)


        C_XY = compute_covariance_matrix(X,Y)
        C_XX = compute_covariance_matrix(X,X)
        C_YY = compute_covariance_matrix(Y,Y)

        dim_x = C_XX.shape[0]
        dim_y = C_YY.shape[0]

        A = inv(sqrtm(C_XX+1e-5*np.eye(dim_x)))
        B = inv(sqrtm(C_YY+1e-5*np.eye(dim_y)))


        C = np.matmul(A,np.matmul(C_XY,B))



        u,s,d = svd(C)

        return u[:,0:k].T, A

    def project(self,proj,white,X):
        '''
        proj, numpy matrix to perform projection
        whit, numpy matrix to perform whitenting
        X, list of states
        '''

        proj = np.matmul(proj,white)

        X_P = np.matmul(proj,np.array(X).T)

        return list(X_P.T)



if __name__ == "__main__":

    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))

    CLASS_LABELS = ['apple','banana','eggplant']

    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim,CLASS_LABELS)


    rand_proj = projections.get_random_proj()
    # #Show Random 2D Projection
    proj2D_viz = Project2D(rand_proj,CLASS_LABELS)
    proj2D_viz.project_data(X,Y, white = np.eye(feat_dim))

    #PCA Projection
    pca_proj = projections.pca_projection(X,Y)

    #Show PCA 2D Projection
    proj2D_viz = Project2D(pca_proj,CLASS_LABELS)
    proj2D_viz.project_data(X,Y, white = np.eye(feat_dim))

    #CCA Projection
    cca_proj,white_cov = projections.cca_projection(X,Y)
    #Show CCA 2D Projection
    proj2D_viz = Project2D(cca_proj,CLASS_LABELS)
    proj2D_viz.project_data(X,Y,white = white_cov)


# ## c (Observations)
# 
# Of the three projections, CCA performs the best while random projections performs the worst. 
# 
# The Random projections are the worst because when bringing a value down from a dimension of 729 to 2, the 2 features you choose can drastically influence your results since some will have much more of a significant impact than others. With such a large sample space, it is highly improbable that we would pick the exact 2 components that maximize differences (it is as probable as picking the exact 2 components that minimize differences). Thus our results are all clustered together.
# 
# PCA and CCA both try to pick the directions of most variance so it makes sense that their results are much different and better than random projections. CCA performs better than PCA because PCA only takes into account the input data in order to determine the areas of largest variance $\Sigma_{XX}$ while CCA uses this the variance of the observations and the relationship between both to figure out the relationship between the inputs and their respective outputs. Therefore CCA's results are slightly better than those of PCA.

# ## 3.d

# In[4]:


from numpy.random import uniform
import random
import time


import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys


from projection import Project2D, Projections
from confusion_mat import getConfusionMatrixPlot

from ridge_model import Ridge_Model
from qda_model import QDA_Model
from lda_model import LDA_Model
from svm_model import SVM_Model
from logistic_model import Logistic_Model


CLASS_LABELS = ['apple','banana','eggplant']





class Model():
    """ Generic wrapper for specific model instance. """

    def __init__(self, model):
        """ Store specific pre-initialized model instance. """

        self.model = model


    def train_model(self,X,Y): 
        """ Train using specific model's training function. """

        self.model.train_model(X,Y)

    def test_model(self,X,Y):
        """ Test using specific model's eval function. """
        if hasattr(self.model, "evals"):
            labels = np.array(Y)
            p_labels = self.model.evals(X)

        else:
            labels = [] # List of actual labels
            p_labels = [] # List of model's predictions
            success = 0 # Number of correct predictions
            total_count = 0 # Number of images

            for i in range(len(X)):

                x = X[i] # Test input
                y = Y[i] # Actual label
                y_ = self.model.eval(x) # Model's prediction
                labels.append(y)
                p_labels.append(y_)

                if y == y_:
                    success += 1
                total_count +=1 

        print("Computing Confusion Matrix")
        # Compute Confusion Matrix
        getConfusionMatrixPlot(labels,p_labels,CLASS_LABELS)


if __name__ == "__main__":

    # Load Training Data and Labels
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))

    # Load Validation Data and Labels
    X_val = list(np.load('little_x_val.npy'))
    Y_val = list(np.load('little_y_val.npy'))

    CLASS_LABELS = ['apple','banana','eggplant']


    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim,CLASS_LABELS)
    cca_proj,white_cov = projections.cca_projection(X,Y,k=2)

    X = projections.project(cca_proj,white_cov,X)
    X_val = projections.project(cca_proj,white_cov,X_val)




    ####RUN RIDGE REGRESSION#####
    ridge_m = Ridge_Model(CLASS_LABELS)
    model = Model(ridge_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN LDA REGRESSION#####

    lda_m = LDA_Model(CLASS_LABELS)
    model = Model(lda_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN QDA REGRESSION#####

    qda_m = QDA_Model(CLASS_LABELS)
    model = Model(qda_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN SVM REGRESSION#####

    svm_m = SVM_Model(CLASS_LABELS)
    model = Model(svm_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)

    ####RUN Logistic REGRESSION#####
    lr_m = Logistic_Model(CLASS_LABELS)
    model = Model(lr_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


# In[5]:


import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys
from numpy.linalg import inv
from numpy.linalg import det
from sklearn.svm import LinearSVC
from projection import Project2D, Projections


class LDA_Model(): 

    def __init__(self,class_labels):

        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.001
        self.NUM_CLASSES = len(class_labels)
        self.Xs = {}
        self.mean = {}
        #self.cov = None
        self.cov_inv = None



    def train_model(self,X,Y): 
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

        '''
        n = len(Y)
        for i in range(0, self.NUM_CLASSES):
            self.Xs[i] = []

        for i in range(n):
            # Y is a single class number, not one-hot encoded
            self.Xs[Y[i]].append(X[i])
        
        for x, arr in self.Xs.items():
            self.mean[x] = np.mean(np.array(arr), axis = 0).reshape((-1,1))
        np_X = np.array(X)
        # self.cov = compute_covariance_matrix(np_X, np_X) + self.reg_cov*np.eye(np_X.shape[1])
        self.cov_inv = np.linalg.inv(compute_covariance_matrix(np_X, np_X) + self.reg_cov*np.eye(np_X.shape[1]))


    def eval(self,x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        probs = []
        x = x.reshape((-1, 1))
        
        for i in range(self.NUM_CLASSES):
            diff = x - self.mean[i]
            probs.append((diff).T@(self.cov_inv)@(diff))
        
        return np.argmin(probs)
        

        
################# BELOW COPIED FROM PART E

if __name__ == "__main__":

    # Load Training Data and Labels
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))

    # Load Validation Data and Labels
    X_val = list(np.load('little_x_val.npy'))
    Y_val = list(np.load('little_y_val.npy'))

    CLASS_LABELS = ['apple','banana','eggplant']


    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim,CLASS_LABELS)
    cca_proj,white_cov = projections.cca_projection(X,Y,k=2)

    X = projections.project(cca_proj,white_cov,X)
    X_val = projections.project(cca_proj,white_cov,X_val)




    ####RUN RIDGE REGRESSION#####
    ridge_m = Ridge_Model(CLASS_LABELS)
    model = Model(ridge_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN LDA REGRESSION#####

    lda_m = LDA_Model(CLASS_LABELS)
    model = Model(lda_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN QDA REGRESSION#####

    qda_m = QDA_Model(CLASS_LABELS)
    model = Model(qda_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN SVM REGRESSION#####

    svm_m = SVM_Model(CLASS_LABELS)
    model = Model(svm_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)

    ####RUN Logistic REGRESSION#####
    lr_m = Logistic_Model(CLASS_LABELS)
    model = Model(lr_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


# In[6]:


import random
import time


import numpy as np
import numpy.linalg as LA


from numpy.linalg import inv
from numpy.linalg import det

from projection import Project2D, Projections


class QDA_Model(): 

    def __init__(self,class_labels):

        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.01
        self.NUM_CLASSES = len(class_labels)
        self.Xs = {}
        self.mean = {}
        # self.cov = {}
        self.cov_inv = {}
        self.reg_mtx = None
        self.reg = {}


    def train_model(self,X,Y): 
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

        '''
        n = len(Y)
        for i in range(0, self.NUM_CLASSES):
            self.Xs[i] = []

        for i in range(n):
            # Y is a single class number, not one-hot encoded
            self.Xs[Y[i]].append(X[i])
            
        self.reg_mtx = self.reg_cov*np.eye(len(X[0]))
        
        for x, arr in self.Xs.items():
            np_arr = np.array(arr)
            self.mean[x] = np.mean(np_arr, axis = 0).reshape((-1,1))
            # self.cov[x] = compute_covariance_matrix(np_arr, np_arr) + self.reg_mtx
            cov = compute_covariance_matrix(np_arr, np_arr) + self.reg_mtx
            self.cov_inv[x] = np.linalg.inv(cov)
            self.reg[x] = np.log(np.linalg.det(cov))



    def eval(self,x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        probs = []
        x = x.reshape((-1, 1))
        
        for i in range(self.NUM_CLASSES):
            diff = x - self.mean[i]
            probs.append((diff.T@(self.cov_inv[i])@ diff)
                         + self.reg[i])
        
        return np.argmin(probs)



################# BELOW COPIED FROM PART E

if __name__ == "__main__":

    # Load Training Data and Labels
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))

    # Load Validation Data and Labels
    X_val = list(np.load('little_x_val.npy'))
    Y_val = list(np.load('little_y_val.npy'))

    CLASS_LABELS = ['apple','banana','eggplant']


    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim,CLASS_LABELS)
    cca_proj,white_cov = projections.cca_projection(X,Y,k=2)

    X = projections.project(cca_proj,white_cov,X)
    X_val = projections.project(cca_proj,white_cov,X_val)




    ####RUN RIDGE REGRESSION#####
    ridge_m = Ridge_Model(CLASS_LABELS)
    model = Model(ridge_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN LDA REGRESSION#####

    lda_m = LDA_Model(CLASS_LABELS)
    model = Model(lda_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN QDA REGRESSION#####

    qda_m = QDA_Model(CLASS_LABELS)
    model = Model(qda_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


    ####RUN SVM REGRESSION#####

    svm_m = SVM_Model(CLASS_LABELS)
    model = Model(svm_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)

    ####RUN Logistic REGRESSION#####
    lr_m = Logistic_Model(CLASS_LABELS)
    model = Model(lr_m)

    model.train_model(X,Y)
    model.test_model(X,Y)
    model.test_model(X_val,Y_val)


# In[7]:


from numpy.random import uniform
import random
import time


import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys


from projection import Project2D, Projections
from confusion_mat import getConfusionMatrixPlot

from ridge_model import Ridge_Model
from qda_model import QDA_Model
from lda_model import LDA_Model
from svm_model import SVM_Model
from logistic_model import Logistic_Model

import matplotlib.pyplot as plt


CLASS_LABELS = ['apple','banana']




def compute_tp_fp(thres, scores, labels):
    scores = np.array(scores)
    prediction = (scores > thres)
    tp = np.sum(prediction * labels)
    tpr = 1.0 * tp / np.sum(labels)

    fp = np.sum(prediction * (1-labels))
    fpr = 1.0*fp / np.sum(1-labels)
    return tpr, fpr

def plot_ROC(tps, fps):
    # plot
    plt.plot(fps, tps)
    plt.ylabel("True Positive Rates")
    plt.xlabel("False Positive Rates")



def ROC(scores, labels):
    thresholds = sorted(np.unique(scores))
    thresholds = [-float("Inf")] + thresholds + [float("Inf")]
    tps = []
    fps = []

    # student code start here
    # TODO: Your code
    for thresh in thresholds:
        t,f = compute_tp_fp(thresh, scores, labels)
        tps.append(t)
        fps.append(f)
    # student code end here

    return tps, fps


def eval_with_ROC(method, train_X, train_Y, val_X, val_Y, C):
    m = method(CLASS_LABELS)
    m.C = C
    m.train_model(train_X, train_Y)
    scores = m.scores(val_X)
    # change the scores here
    # scores = 10.0 * np.array(scores)
    
    tps, fps = ROC(scores, val_Y)
    plot_ROC(tps, fps)

def trim_data(X, Y):
    # throw away the 3rd class data
    X = np.array(X)
    Y = np.array(Y)
    retain = (Y < 2)
    return X[retain, :], Y[retain]

if __name__ == "__main__":

    # Load Training Data and Labels
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))
    X, Y = trim_data(X, Y)

    # Load Validation Data and Labels
    X_val = list(np.load('little_x_val.npy'))
    Y_val = list(np.load('little_y_val.npy'))
    X_val, Y_val = trim_data(X_val, Y_val)

    CLASS_LABELS = ['apple','banana']


    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim,CLASS_LABELS)
    cca_proj,white_cov = projections.cca_projection(X,Y,k=2)

    X = projections.project(cca_proj,white_cov,X)
    X_val = projections.project(cca_proj,white_cov,X_val)


    ####RUN SVM REGRESSION#####
    eval_with_ROC(SVM_Model, X, Y, X_val, Y_val, 1.0)
    eval_with_ROC(SVM_Model, X, Y, X_val, Y_val, 100000.0)
    plt.legend(["C=1.0", "C=100000.0"])
    plt.show()


# ## i (Observations)
# 
# The regularization weight C = 1.0 is better because for all points it seems as it is just as good or better than C = 100000.0 at reaching the goal of TPR = 1.0 and FPR = 0.0. An estimation of success could be the area under the curve because this measures accuracy and C = 1.0 has the larger area under the curve. Interpreting the graph, it seems like an increase in guessing positive at the beginning of the curve causes the TPR to have a greater increase relative to FPR when C = 1.0 compared to C = 100000.0, while for the rest of the curve the two are basically the same.

# In[10]:


def eval_with_ROC(method, train_X, train_Y, val_X, val_Y, C):
    m = method(CLASS_LABELS)
    m.C = C
    m.train_model(train_X, train_Y)
    scores = m.scores(val_X)
    # change the scores here
    scores = 10.0 * np.array(scores)
    
    tps, fps = ROC(scores, val_Y)
    plot_ROC(tps, fps)

if __name__ == "__main__":

    # Load Training Data and Labels
    X = list(np.load('little_x_train.npy'))
    Y = list(np.load('little_y_train.npy'))
    X, Y = trim_data(X, Y)

    # Load Validation Data and Labels
    X_val = list(np.load('little_x_val.npy'))
    Y_val = list(np.load('little_y_val.npy'))
    X_val, Y_val = trim_data(X_val, Y_val)

    CLASS_LABELS = ['apple','banana']


    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim,CLASS_LABELS)
    cca_proj,white_cov = projections.cca_projection(X,Y,k=2)

    X = projections.project(cca_proj,white_cov,X)
    X_val = projections.project(cca_proj,white_cov,X_val)


    ####RUN SVM REGRESSION#####
    eval_with_ROC(SVM_Model, X, Y, X_val, Y_val, 1.0)
    eval_with_ROC(SVM_Model, X, Y, X_val, Y_val, 100000.0)
    plt.legend(["C=1.0", "C=100000.0"])
    plt.show()


# ## j (Observations)
# 
# If you multiply the scores output by the classifier by a factor of 10.0, the ROC curve's general trend as well as the better C value (C = 1.0) does not change. Multiple trials shows that the graphs themselves change a little bit every time and a good portion of the time C=1.0 and C = 100000.0 are very similar.

# ## 3.k

# In[12]:


import cv2
import IPython
from numpy.random import uniform
import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys


from projection import Project2D, Projections

from confusion_mat import getConfusionMatrix
from confusion_mat import plotConfusionMatrix

from ridge_model import Ridge_Model
from qda_model import QDA_Model
from lda_model import LDA_Model
from svm_model import SVM_Model


class LDA_Model(): 

    def __init__(self,class_labels):

        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.001
        self.NUM_CLASSES = len(class_labels)
        self.Xs = {}
        self.mean = {}
        #self.cov = None
        self.cov_inv = None



    def train_model(self,X,Y): 
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

        '''
        n = len(Y)
        for i in range(0, self.NUM_CLASSES):
            self.Xs[i] = []

        for i in range(n):
            # Y is a single class number, not one-hot encoded
            self.Xs[Y[i]].append(X[i])
        
        for x, arr in self.Xs.items():
            self.mean[x] = np.mean(np.array(arr), axis = 0).reshape((-1,1))
        np_X = np.array(X)
        # self.cov = compute_covariance_matrix(np_X, np_X) + self.reg_cov*np.eye(np_X.shape[1])
        self.cov_inv = np.linalg.inv(compute_covariance_matrix(np_X, np_X) + self.reg_cov*np.eye(np_X.shape[1]))


    def eval(self,x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        probs = []
        x = x.reshape((-1, 1))
        
        for i in range(self.NUM_CLASSES):
            diff = x - self.mean[i]
            probs.append((diff).T@(self.cov_inv)@(diff))
        
        return np.argmin(probs)

    

class QDA_Model(): 

    def __init__(self,class_labels):

        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.28
        self.NUM_CLASSES = len(class_labels)
        self.Xs = {}
        self.mean = {}
        # self.cov = {}
        self.cov_inv = {}
        self.reg_mtx = None
        self.reg = {}


    def train_model(self,X,Y): 
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

        '''
        n = len(Y)
        for i in range(0, self.NUM_CLASSES):
            self.Xs[i] = []

        for i in range(n):
            # Y is a single class number, not one-hot encoded
            self.Xs[Y[i]].append(X[i])
            
        self.reg_mtx = self.reg_cov*np.eye(len(X[0]))
        
        for x, arr in self.Xs.items():
            np_arr = np.array(arr)
            self.mean[x] = np.mean(np_arr, axis = 0).reshape((-1,1))
            # self.cov[x] = compute_covariance_matrix(np_arr, np_arr) + self.reg_mtx
            cov = compute_covariance_matrix(np_arr, np_arr) + self.reg_mtx
            self.cov_inv[x] = np.linalg.inv(cov)
            self.reg[x] = np.log(np.linalg.det(cov))



    def eval(self,x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        probs = []
        x = x.reshape((-1, 1))
        
        for i in range(self.NUM_CLASSES):
            diff = x - self.mean[i]
            probs.append((diff.T@(self.cov_inv[i])@ diff)
                         + self.reg[i])
        
        return np.argmin(probs)
 

    

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple', 
                'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

def eval_model(X,Y,k,model_key,proj):
    # PROJECT DATA 
    cca_proj,white_cov = proj.cca_projection(X,Y,k=k)

    X_p = proj.project(cca_proj,white_cov,X)
    X_val_p = proj.project(cca_proj,white_cov,X_val)

    # TRAIN MODEL 
    model = models[model_key]

    model.train_model(X_p,Y)
    acc,cm = model.test_model(X_val_p,Y_val)

    return acc,cm


class Model(): 
    """ Generic wrapper for specific model instance. """


    def __init__(self,model):
        """ Store specific pre-initialized model instance. """

        self.model = model


    def train_model(self,X,Y): 
        """ Train using specific model's training function. """

        self.model.train_model(X,Y)


    def test_model(self,X,Y):
        """ Test using specific model's eval function. """
        if hasattr(self.model, "evals"):
            labels = np.array(Y)
            p_labels = self.model.evals(X)
            success =np.sum(labels == p_labels)
            total_count = len(X)

        else:
            labels = [] # List of actual labels
            p_labels = [] # List of model's predictions
            success = 0 # Number of correct predictions
            total_count = 0 # Number of images

            for i in range(len(X)):

                x = X[i] # Test input
                y = Y[i] # Actual label
                y_ = self.model.eval(x) # Model's prediction
                labels.append(y)
                p_labels.append(y_)

                if y == y_:
                    success += 1
                total_count +=1

        return 1.0*success/total_count, getConfusionMatrix(labels,p_labels)


if __name__ == "__main__":

    # Load Training Data and Labels
    X = list(np.load('big_x_train.npy'))
    Y = list(np.load('big_y_train.npy'))

    # Load Validation Data and Labels
    X_val = list(np.load('big_x_val.npy'))
    Y_val = list(np.load('big_y_val.npy'))


    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim,CLASS_LABELS)



    models = {} # Dictionary of key: model names, value: model instance

    #########MODELS TO EVALUATE############
    qda_m = QDA_Model(CLASS_LABELS)
    models['qda'] =  Model(qda_m)

    lda_m = LDA_Model(CLASS_LABELS)
    models['lda'] = Model(lda_m)

    ridge_m = Ridge_Model(CLASS_LABELS)
    models['ridge'] = Model(ridge_m)

    ridge_m_10 = Ridge_Model(CLASS_LABELS)
    ridge_m_10.lmbda = 10.0
    models['ridge_lmda_10'] = Model(ridge_m_10)

    ridge_m_01 = Ridge_Model(CLASS_LABELS)
    ridge_m_01.lmbda = 0.1
    models['ridge_lmda_01'] = Model(ridge_m_01)

    svm_m = SVM_Model(CLASS_LABELS)
    models['svm'] = Model(svm_m)

    svm_m_10 = SVM_Model(CLASS_LABELS)
    svm_m_10.C = 10.0
    models['svm_C_10'] = Model(svm_m_10)

    svm_m_01 = SVM_Model(CLASS_LABELS)
    svm_m_01.C = 0.1
    models['svm_C_01'] = Model(svm_m_01)




    #########GRID SEARCH OVER MODELS############
    highest_accuracy = 0 # Highest validation accuracy
    best_model_name = None # Best model name
    best_model = None # Best model instance

    K = [50,200,600,800] # List of dimensions


    for model_key in models.keys():
        print(model_key)

        val_acc = [] # List of model's accuracies for each dimension 
        for k in K:
            print("k =", k)

            # Evaluate specific model's validation accuracy on specific dimension
            acc,c_m = eval_model(X,Y,k,model_key,projections)

            val_acc.append(acc)

            if acc > highest_accuracy: 
                highest_accuracy = acc
                best_model_name = model_key
                best_cm = c_m

        # Plot specific model's accuracies across validation error
        plt.plot(K,val_acc,label=model_key)


    # Display aggregate plot of models across validation error
    plt.legend()
    plt.xlabel('Dimension') 
    plt.ylabel('Accuracy') 
    plt.show()


    # Plot best model's confusion matrix
    plotConfusionMatrix(best_cm,CLASS_LABELS)


# In[28]:


import cv2
import IPython
from numpy.random import uniform
import random
import time


import glob
import os
import pickle
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as LA

import sys


from projection import Project2D, Projections

from confusion_mat import getConfusionMatrix
from confusion_mat import plotConfusionMatrix

from ridge_model import Ridge_Model
from qda_model import QDA_Model
from lda_model import LDA_Model
from svm_model import SVM_Model


class LDA_Model(): 

    def __init__(self,class_labels):

        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.001
        self.NUM_CLASSES = len(class_labels)
        self.Xs = {}
        self.mean = {}
        #self.cov = None
        self.cov_inv = None



    def train_model(self,X,Y): 
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

        '''
        n = len(Y)
        for i in range(0, self.NUM_CLASSES):
            self.Xs[i] = []

        for i in range(n):
            # Y is a single class number, not one-hot encoded
            self.Xs[Y[i]].append(X[i])
        
        for x, arr in self.Xs.items():
            self.mean[x] = np.mean(np.array(arr), axis = 0).reshape((-1,1))
        np_X = np.array(X)
        # self.cov = compute_covariance_matrix(np_X, np_X) + self.reg_cov*np.eye(np_X.shape[1])
        self.cov_inv = np.linalg.inv(compute_covariance_matrix(np_X, np_X) + self.reg_cov*np.eye(np_X.shape[1]))


    def eval(self,x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        probs = []
        x = x.reshape((-1, 1))
        
        for i in range(self.NUM_CLASSES):
            diff = x - self.mean[i]
            probs.append((diff).T@(self.cov_inv)@(diff))
        
        return np.argmin(probs)

    

class QDA_Model(): 

    def __init__(self,class_labels):

        ###SCALE AN IDENTITY MATRIX BY THIS TERM AND ADD TO COMPUTED COVARIANCE MATRIX TO PREVENT IT BEING SINGULAR ###
        self.reg_cov = 0.01
        self.NUM_CLASSES = len(class_labels)
        self.Xs = {}
        self.mean = {}
        # self.cov = {}
        self.cov_inv = {}
        self.reg_mtx = None
        self.reg = {}


    def train_model(self,X,Y): 
        ''''
        FILL IN CODE TO TRAIN MODEL
        MAKE SURE TO ADD HYPERPARAMTER TO MODEL 

        '''
        n = len(Y)
        for i in range(0, self.NUM_CLASSES):
            self.Xs[i] = []

        for i in range(n):
            # Y is a single class number, not one-hot encoded
            self.Xs[Y[i]].append(X[i])
            
        self.reg_mtx = self.reg_cov*np.eye(len(X[0]))
        
        for x, arr in self.Xs.items():
            np_arr = np.array(arr)
            self.mean[x] = np.mean(np_arr, axis = 0).reshape((-1,1))
            # self.cov[x] = compute_covariance_matrix(np_arr, np_arr) + self.reg_mtx
            cov = compute_covariance_matrix(np_arr, np_arr) + self.reg_mtx
            self.cov_inv[x] = np.linalg.inv(cov)
            self.reg[x] = np.log(np.linalg.det(cov))



    def eval(self,x):
        ''''
        Fill in code to evaluate model and return a prediction
        Prediction should be an integer specifying a class
        '''
        probs = []
        x = x.reshape((-1, 1))
        
        for i in range(self.NUM_CLASSES):
            diff = x - self.mean[i]
            probs.append((diff.T@(self.cov_inv[i])@ diff)
                         + self.reg[i])
        
        return np.argmin(probs)
 

    

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple', 
                'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

def eval_model(X,Y,k,model_key,proj):
    # PROJECT DATA 
    cca_proj,white_cov = proj.cca_projection(X,Y,k=k)

    X_p = proj.project(cca_proj,white_cov,X)
    X_val_p = proj.project(cca_proj,white_cov,X_val)

    # TRAIN MODEL 
    model = models[model_key]

    model.train_model(X_p,Y)
    acc,cm = model.test_model(X_val_p,Y_val)

    return acc,cm


class Model(): 
    """ Generic wrapper for specific model instance. """


    def __init__(self,model):
        """ Store specific pre-initialized model instance. """

        self.model = model


    def train_model(self,X,Y): 
        """ Train using specific model's training function. """

        self.model.train_model(X,Y)


    def test_model(self,X,Y):
        """ Test using specific model's eval function. """
        if hasattr(self.model, "evals"):
            labels = np.array(Y)
            p_labels = self.model.evals(X)
            success =np.sum(labels == p_labels)
            total_count = len(X)

        else:
            labels = [] # List of actual labels
            p_labels = [] # List of model's predictions
            success = 0 # Number of correct predictions
            total_count = 0 # Number of images

            for i in range(len(X)):

                x = X[i] # Test input
                y = Y[i] # Actual label
                y_ = self.model.eval(x) # Model's prediction
                labels.append(y)
                p_labels.append(y_)

                if y == y_:
                    success += 1
                total_count +=1

        return 1.0*success/total_count, getConfusionMatrix(labels,p_labels)


if __name__ == "__main__":

    # Load Training Data and Labels
    X = list(np.load('big_x_train.npy'))
    Y = list(np.load('big_y_train.npy'))

    # Load Validation Data and Labels
    X_val = list(np.load('big_x_val.npy'))
    Y_val = list(np.load('big_y_val.npy'))


    # Project Data to 200 Dimensions using CCA
    feat_dim = max(X[0].shape)
    projections = Projections(feat_dim,CLASS_LABELS)



    models = {} # Dictionary of key: model names, value: model instance

    #########MODELS TO EVALUATE############
    qda_m = QDA_Model(CLASS_LABELS)
    models['qda'] =  Model(qda_m)

    lda_m = LDA_Model(CLASS_LABELS)
    models['lda'] = Model(lda_m)

    ridge_m = Ridge_Model(CLASS_LABELS)
    models['ridge'] = Model(ridge_m)

    ridge_m_10 = Ridge_Model(CLASS_LABELS)
    ridge_m_10.lmbda = 10.0
    models['ridge_lmda_10'] = Model(ridge_m_10)

    ridge_m_01 = Ridge_Model(CLASS_LABELS)
    ridge_m_01.lmbda = 0.1
    models['ridge_lmda_01'] = Model(ridge_m_01)

    svm_m = SVM_Model(CLASS_LABELS)
    models['svm'] = Model(svm_m)

    svm_m_10 = SVM_Model(CLASS_LABELS)
    svm_m_10.C = 10.0
    models['svm_C_10'] = Model(svm_m_10)

    svm_m_01 = SVM_Model(CLASS_LABELS)
    svm_m_01.C = 0.1
    models['svm_C_01'] = Model(svm_m_01)




    #########GRID SEARCH OVER MODELS############
    highest_accuracy = 0 # Highest validation accuracy
    best_model_name = None # Best model name
    best_model = None # Best model instance

    K = [50,200,600,800] # List of dimensions


    for model_key in models.keys():
        print(model_key)

        val_acc = [] # List of model's accuracies for each dimension 
        for k in K:
            print("k =", k)

            # Evaluate specific model's validation accuracy on specific dimension
            acc,c_m = eval_model(X,Y,k,model_key,projections)

            val_acc.append(acc)

            if acc > highest_accuracy: 
                highest_accuracy = acc
                best_model_name = model_key
                best_cm = c_m

        # Plot specific model's accuracies across validation error
        plt.plot(K,val_acc,label=model_key)


    # Display aggregate plot of models across validation error
    plt.legend()
    plt.xlabel('Dimension') 
    plt.ylabel('Accuracy') 
    plt.show()


    # Plot best model's confusion matrix
    plotConfusionMatrix(best_cm,CLASS_LABELS)



# # Question 4
# 
# ## 4.m

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_data(mu_1, mu_2, n_examples, sigma_1 = 0.5, sigma_2 = 0.5):
    # Generate sample data points from 2 distributions: (mu_1, sigma_1) and (mu_2, sigma_2)
    d_1 = np.random.normal(mu_1, sigma_1, n_examples)
    d_2 = np.random.normal(mu_2, sigma_2, n_examples)
    x = np.concatenate([d_1, d_2])
    return d_1, d_2, x

def plot_data(d_1, d_2):
    # Plot scatter plot of data samples, labeled by class
    plt.figure()
    plt.scatter(d_1, np.zeros(len(d_1)), c='b', s=80., marker='+')
    plt.scatter(d_2, np.zeros(len(d_2)), c='r', s=80.)
    plt.title("Sample data using: mu = " + str(mu) + " n_train = " + str(len(d_1)+len(d_2)))
    plt.show()
    return

def plot_data_and_distr(d_1, d_2, mu_1, mu_2, sigma_1=0.5, sigma_2=0.5, title = ""):
    # Plot scatter plot of data samples overlayed with distribution of estimated means: mu_1 and mu_2
    plt.scatter(d_1, np.zeros(len(d_1)), c='b')
    scale = [min(mu_1-3*sigma_1, mu_2-3*sigma_2), max(mu_1+3*sigma_1, mu_2+3*sigma_2)]
    plt.scatter(d_2, np.zeros(len(d_2)), c='r')
    x_axis = np.arange(scale[0], scale[1], 0.001)
    plt.plot(x_axis, norm.pdf(x_axis,mu_1, sigma_1), c='b')
    x_axis = np.arange(scale[0], scale[1], 0.001)
    plt.plot(x_axis, norm.pdf(x_axis,mu_2,sigma_2), c='r')
    plt.title(title)
    plt.show()


def grad_ascent(x, mu, mu_true, sigma = 0.5, iterations = 1000):
    # Run gradient ascent on the likelihood of a point belonging to a class and compare the estimates of the mean
    #     at each iteration with the true mean of the distribution
    # Note: the original dataset comes from 2 distributions centered at mu and -mu, which this takes into account
    #     with each update
    diff_mu = []
    alpha = 0.05
    for i in range(iterations):
        phi_1 = np.exp(-np.square(x-mu)/(2*np.square(sigma)))
        phi_2 = np.exp(-np.square(x+mu)/(2*np.square(sigma)))
        w = phi_1/(phi_1 + phi_2)
        em = (1/len(x))*np.sum((2*w - 1)*x)
        mu = mu*(1-alpha) + alpha*em
        diff_mu.append(np.abs(mu-mu_true))
    return mu, sigma,diff_mu


def em(x, mu, mu_true, sigma = 0.5, iterations = 1000):
    # Run the EM algorithm to find the estimated mean of the dataset
    # Note: the original dataset comes from 2 distributions centered at mu and -mu, which this takes into account
    #     with each update
    diff_mu = np.zeros(iterations)
    for i in range(iterations):
        phi_1 = np.exp(-np.square(x-mu)/(2*np.square(sigma)))
        phi_2 = np.exp(-np.square(x+mu)/(2*np.square(sigma)))
        w = phi_1/(phi_1 + phi_2)
    
        mu = (1/len(x))*np.sum((2*w - 1)*x)
        diff_mu[i]  = np.abs(mu-mu_true)
    return mu, sigma,diff_mu

def kmeans(x, mu, mu_true, sigma = 0.5, iterations = 1000):
    # Run the K means algorithm to find the estimated mean of the dataset
    # Note: the original dataset comes from 2 distributions centered at mu and -mu, which this takes into account
    #     with each update
    diff_mu = np.zeros(iterations)

    for i in range(iterations):
        mu_1 = mu
        mu_2 = -mu
        set1 = []
        set2 = []
        for x_i in x:
            if np.abs(x_i - mu_1) <= np.abs(x_i - mu_2):
                set1.append(x_i)
            else:
                set2.append(x_i)
        mu_1_new = np.mean(set1)
        mu_2_new = np.mean(set2)
        # Estimates two means and combines them to get mu for the next iteration
        mu = np.abs(mu_1_new - mu_2_new)/2
        diff_mu[i]  = np.abs(mu-mu_true)
    return mu, sigma, diff_mu


def plot_differences(iterations, diff_mu_ga, diff_mu_em, mu, n_examples):
    # Make plot comparing convergence of means to true mean for gradient descent and EM
    plt.plot(np.arange(iterations), diff_mu_ga, c = 'r', label='GD')
    plt.plot(np.arange(iterations), diff_mu_em, c = 'b', label='EM')
    plt.legend()
    plt.title("Difference between estimated and true mean when: mu = " + str(mu) + " n_train = " + str(n_examples))
    plt.xlabel("iterations for Training")
    plt.ylabel("Absolute value of difference to true mean")
    plt.show()


def plot_ll(x, scale=[-5, 5]):
    # if you want to visualize the likelihood function as a function of mu
    mus = np.linspace(scale[0], scale[1], 200)
    ll = np.zeros(mus.shape)
    for j, mu in enumerate(mus):
        ll[j] = 0.
        for xi in x:
            ll[j] += np.log(np.exp(-(xi-mu)**2/2)+np.exp(-(xi+mu)**2/2)) - np.log(2*np.sqrt(2*np.pi))
    plt.plot(mus, ll)
    plt.show()
    return

##############################################################################
part_m = True
part_n = False

#1-part_m

# Set this to True if you want to visualize the distribution estimated by each method
visualize_distr = True
##############################################################################

np.random.seed(12312)
mu_list = [.5, 3.]
n_train_list = [50]
mu_start = 0.1

for mu in mu_list:
    for n_train in n_train_list:
        d_1, d_2, x = generate_data(mu, -mu, n_examples = n_train)
        plot_data(d_1, d_2)

        # Code for part m
        if part_m:
            mu_ga, sigma_gd, diff_mu_ga = grad_ascent(x, mu_start, mu)
            mu_em, sigma_em, diff_mu_em = em(x, mu_start, mu)
            if visualize_distr:
                plot_data_and_distr(d_1, d_2, mu_ga, -mu_ga, title = "Estimated distribution using Gradient Ascent")
                plot_data_and_distr(d_1, d_2, mu_em, -mu_em, title = "Estimated distribution using EM")
            print ("------------")
            print("n_points:", n_train*2, ", True mean:{:.3f}, GA (final) estimate:{:.3f}, EM (final) estimate:{:.3f}".format(mu, mu_ga, mu_em)) #, "GA (final) mean: {.3f}", mu_ga, "EM (final) mean: {.3f}", mu_em)
            print ("------------")
            plot_differences(1000, diff_mu_ga, diff_mu_em, mu, 2*n_train)

        # Code for part n
        if part_n:
            mu_ga, sigma_gd, diff_mu_ga = grad_ascent(x, mu_start, mu)
            mu_em, sigma_em, diff_mu_em = em(x, mu_start, mu)
            mu_k, sigma_k, diff_mu_k = kmeans(x, mu_start, mu)
            if visualize_distr:
                plot_data_and_distr(d_1, d_2, mu_ga, -mu_ga, title = "Estimated distribution using Gradient Ascent")
                plot_data_and_distr(d_1, d_2, mu_em, -mu_em, title = "Estimated distribution using EM")
                plot_data_and_distr(d_1, d_2, mu_k, -mu_k, title = "Estimated distribution using K Means")
            print ("------------")
            print("True mean:{:.3f}, GA (final) estimate:{:.3f}, EM (final) estimate:{:.3f}, K-Means (final) estimate:{:.3f}".format(mu, mu_ga, mu_em, mu_k))
            print ("------------")
          


# **4.m Observations**
# 
# Similarities: Both reach the final estimation succesfully. 
# 
# Differences: EM converges to the final estimation faster and also does not overshoot the target estimation or fluctuate around it. 
# 
# 
# From part (l) we derived the gradient descent algorithm by taking the derivative of the EM algorithm and setting it to 0. We showed that the EM algorithm and the Gradient Ascent algorithm are equivalent when we scale by (1/n). The updates derived in the previous parts show how similar the two algorithms are. 
# 
# From our plots above we see that EM and Gradient Ascet both create very similar distributions and have the same estimates for the mean. 
# 
# EM appears to be the better option for both of the cases, near and far, plotted above. It converges faster and does not fluctuate. EM always updates Q and theta to make the free energy a closer and closer bound to the true log likelihood estimate while gradient descent uses derivatives to find the direction of increase to approach the likelihood, so it makes sense the both converge.
# 
# As well, EM does not depend on the hyper parameter alpha in order to succeed.The plots complement the similarity of the two algorithms we saw in the previous two parts.
# 

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_data(mu_1, mu_2, n_examples, sigma_1 = 0.5, sigma_2 = 0.5):
    # Generate sample data points from 2 distributions: (mu_1, sigma_1) and (mu_2, sigma_2)
    d_1 = np.random.normal(mu_1, sigma_1, n_examples)
    d_2 = np.random.normal(mu_2, sigma_2, n_examples)
    x = np.concatenate([d_1, d_2])
    return d_1, d_2, x

def plot_data(d_1, d_2):
    # Plot scatter plot of data samples, labeled by class
    plt.figure()
    plt.scatter(d_1, np.zeros(len(d_1)), c='b', s=80., marker='+')
    plt.scatter(d_2, np.zeros(len(d_2)), c='r', s=80.)
    plt.title("Sample data using: mu = " + str(mu) + " n_train = " + str(len(d_1)+len(d_2)))
    plt.show()
    return

def plot_data_and_distr(d_1, d_2, mu_1, mu_2, sigma_1=0.5, sigma_2=0.5, title = ""):
    # Plot scatter plot of data samples overlayed with distribution of estimated means: mu_1 and mu_2
    plt.scatter(d_1, np.zeros(len(d_1)), c='b')
    scale = [min(mu_1-3*sigma_1, mu_2-3*sigma_2), max(mu_1+3*sigma_1, mu_2+3*sigma_2)]
    plt.scatter(d_2, np.zeros(len(d_2)), c='r')
    x_axis = np.arange(scale[0], scale[1], 0.001)
    plt.plot(x_axis, norm.pdf(x_axis,mu_1, sigma_1), c='b')
    x_axis = np.arange(scale[0], scale[1], 0.001)
    plt.plot(x_axis, norm.pdf(x_axis,mu_2,sigma_2), c='r')
    plt.title(title)
    plt.show()


def grad_ascent(x, mu, mu_true, sigma = 0.5, iterations = 1000):
    # Run gradient ascent on the likelihood of a point belonging to a class and compare the estimates of the mean
    #     at each iteration with the true mean of the distribution
    # Note: the original dataset comes from 2 distributions centered at mu and -mu, which this takes into account
    #     with each update
    diff_mu = []
    alpha = 0.05
    for i in range(iterations):
        phi_1 = np.exp(-np.square(x-mu)/(2*np.square(sigma)))
        phi_2 = np.exp(-np.square(x+mu)/(2*np.square(sigma)))
        w = phi_1/(phi_1 + phi_2)
        em = (1/len(x))*np.sum((2*w - 1)*x)
        mu = mu*(1-alpha) + alpha*em
        diff_mu.append(np.abs(mu-mu_true))
    return mu, sigma,diff_mu


def em(x, mu, mu_true, sigma = 0.5, iterations = 1000):
    # Run the EM algorithm to find the estimated mean of the dataset
    # Note: the original dataset comes from 2 distributions centered at mu and -mu, which this takes into account
    #     with each update
    diff_mu = np.zeros(iterations)
    for i in range(iterations):
        phi_1 = np.exp(-np.square(x-mu)/(2*np.square(sigma)))
        phi_2 = np.exp(-np.square(x+mu)/(2*np.square(sigma)))
        w = phi_1/(phi_1 + phi_2)
    
        mu = (1/len(x))*np.sum((2*w - 1)*x)
        diff_mu[i]  = np.abs(mu-mu_true)
    return mu, sigma,diff_mu

def kmeans(x, mu, mu_true, sigma = 0.5, iterations = 1000):
    # Run the K means algorithm to find the estimated mean of the dataset
    # Note: the original dataset comes from 2 distributions centered at mu and -mu, which this takes into account
    #     with each update
    diff_mu = np.zeros(iterations)

    for i in range(iterations):
        mu_1 = mu
        mu_2 = -mu
        set1 = []
        set2 = []
        for x_i in x:
            if np.abs(x_i - mu_1) <= np.abs(x_i - mu_2):
                set1.append(x_i)
            else:
                set2.append(x_i)
        mu_1_new = np.mean(set1)
        mu_2_new = np.mean(set2)
        # Estimates two means and combines them to get mu for the next iteration
        mu = np.abs(mu_1_new - mu_2_new)/2
        diff_mu[i]  = np.abs(mu-mu_true)
    return mu, sigma, diff_mu


def plot_differences(iterations, diff_mu_ga, diff_mu_em, mu, n_examples):
    # Make plot comparing convergence of means to true mean for gradient descent and EM
    plt.plot(np.arange(iterations), diff_mu_ga, c = 'r', label='GD')
    plt.plot(np.arange(iterations), diff_mu_em, c = 'b', label='EM')
    plt.legend()
    plt.title("Difference between estimated and true mean when: mu = " + str(mu) + " n_train = " + str(n_examples))
    plt.xlabel("iterations for Training")
    plt.ylabel("Absolute value of difference to true mean")
    plt.show()


def plot_ll(x, scale=[-5, 5]):
    # if you want to visualize the likelihood function as a function of mu
    mus = np.linspace(scale[0], scale[1], 200)
    ll = np.zeros(mus.shape)
    for j, mu in enumerate(mus):
        ll[j] = 0.
        for xi in x:
            ll[j] += np.log(np.exp(-(xi-mu)**2/2)+np.exp(-(xi+mu)**2/2)) - np.log(2*np.sqrt(2*np.pi))
    plt.plot(mus, ll)
    plt.show()
    return

##############################################################################
part_m = False
part_n = True

# Set this to True if you want to visualize the distribution estimated by each method
visualize_distr = True
##############################################################################

np.random.seed(12312)
mu_list = [.5, 3.]
n_train_list = [50]
mu_start = 0.1

for mu in mu_list:
    for n_train in n_train_list:
        d_1, d_2, x = generate_data(mu, -mu, n_examples = n_train)
        plot_data(d_1, d_2)

        # Code for part m
        if part_m:
            mu_ga, sigma_gd, diff_mu_ga = grad_ascent(x, mu_start, mu)
            mu_em, sigma_em, diff_mu_em = em(x, mu_start, mu)
            if visualize_distr:
                plot_data_and_distr(d_1, d_2, mu_ga, -mu_ga, title = "Estimated distribution using Gradient Ascent")
                plot_data_and_distr(d_1, d_2, mu_em, -mu_em, title = "Estimated distribution using EM")
            print ("------------")
            print("n_points:", n_train*2, ", True mean:{:.3f}, GA (final) estimate:{:.3f}, EM (final) estimate:{:.3f}".format(mu, mu_ga, mu_em)) #, "GA (final) mean: {.3f}", mu_ga, "EM (final) mean: {.3f}", mu_em)
            print ("------------")
            plot_differences(1000, diff_mu_ga, diff_mu_em, mu, 2*n_train)

        # Code for part n
        if part_n:
            mu_ga, sigma_gd, diff_mu_ga = grad_ascent(x, mu_start, mu)
            mu_em, sigma_em, diff_mu_em = em(x, mu_start, mu)
            mu_k, sigma_k, diff_mu_k = kmeans(x, mu_start, mu)
            if visualize_distr:
                plot_data_and_distr(d_1, d_2, mu_ga, -mu_ga, title = "Estimated distribution using Gradient Ascent")
                plot_data_and_distr(d_1, d_2, mu_em, -mu_em, title = "Estimated distribution using EM")
                plot_data_and_distr(d_1, d_2, mu_k, -mu_k, title = "Estimated distribution using K Means")
            print ("------------")
            print("True mean:{:.3f}, GA (final) estimate:{:.3f}, EM (final) estimate:{:.3f}, K-Means (final) estimate:{:.3f}".format(mu, mu_ga, mu_em, mu_k))
            print ("------------")
          


# **4.n Observations**
# 
# The final estimates for EM and GA are the same as before. K means has a slightly higher final estiamte when the two mixtures from the data are close. However when the two mixtures from the data are farther apart than the final estimates, estimated distributions for all three methods are the same. K means depends on whether or not Mu is large or small. 
# 
# To conclude, EM converges to a good estimates of Mu faster than GA. GA is a weighted version of EM, it is slower than EM. K means will perform differently based off of the size of mu. 

# # Question 5

# In[13]:


from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os

#This function generate random mean and covariance
def gauss_params_gen(num_clusters, num_dims, factor):
    mu = np.random.randn(num_clusters,num_dims)*factor
    sigma = np.random.randn(num_clusters,num_dims,num_dims)
    for k in range(num_clusters):
        sigma[k] = np.dot(sigma[k],sigma[k].T)

    return (mu, sigma)

#Given mean and covariance generate data
def data_gen(mu, sigma, num_clusters, num_samples):
    labels = []
    X = []
    cluster_prob = np.array([np.random.rand() for k in  range(num_clusters)])
    cluster_num_samples = (num_samples * cluster_prob / sum(cluster_prob)).astype(int)
    cluster_num_samples[-1] = num_samples-sum(cluster_num_samples[:-1])

    for k, ks in enumerate(cluster_num_samples):
        labels.append([k]*ks)
        X.append(np.random.multivariate_normal(mu[k], sigma[k], ks))

    # shuffle data
    randomize = np.arange(num_samples)
    np.random.shuffle(randomize)
    X =  np.vstack(X)[randomize]
    labels =  np.array(sum(labels,[]))[randomize]

    return X, labels


def data2D_plot(ax, x, labels, centers, cmap, title):
    data = {'x0': x[:,0], 'x1': x[:,1], 'label': labels}
    ax.scatter(data['x0'], data['x1'], c=data['label'], cmap=cmap, s=20, alpha=0.3)
    ax.scatter(centers[:, 0], centers[:, 1], c=np.arange(np.shape(centers)[0]), cmap=cmap, s=50, alpha=1)
    ax.scatter(centers[:, 0], centers[:, 1], c='black', cmap=cmap, s=20, alpha=1)
    ax.title.set_text(title)

def plot_init_means(x, mus, algs, fname):
    import matplotlib.cm as cm
    fig = plt.figure()
    plt.scatter(x[:,0], x[:,1], c='gray', cmap='viridis', s=20, alpha= 0.4, label='data')
    for mu, alg, clr in zip(mus, algs, cm.viridis(np.linspace(0, 1, len(mus)))):
        plt.scatter(mu[:,0], mu[:, 1], c=clr, s=50, label=alg)
        plt.scatter(mu[:, 0], mu[:, 1], c='black', s=10, alpha=1)
    legend = plt.legend(loc='upper right', fontsize='small')
    plt.title('Initial guesses for centroids')
    fig.savefig(fname)

def loss_plot(loss, title, xlabel, ylabel, fname):
	fig = plt.figure(figsize = (13, 6))
	plt.plot(np.array(loss))
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	fig.savefig(fname)


def gaussian_pdf (X, mu, sigma):
    # Gaussian probability density function
    return np.linalg.det(sigma) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.)                     * np.exp(-.5 * np.einsum('ij, ij -> i',                    X - mu, np.dot(np.linalg.inv(sigma) , (X - mu).T).T ) )

def EM_initial_guess (num_dims, data, num_samples, num_clusters):
    # randomly choose the starting centroids/means
    # as num_clusters of the points from datasets
    mu = data[np.random.choice(num_samples, num_clusters, False), :]

    # initialize the covariance matrice for each gaussian
    sigma = [np.eye(num_dims)] * num_clusters

    # initialize the probabilities/weights for each gaussian
    # begin with equal weight for each gaussian
    alpha = [1./num_clusters] * num_clusters

    return mu, sigma, alpha

def EM_E_step (num_clusters, num_samples, data, mu, sigma, alpha):
    ## Vectorized implementation of e-step equation to calculate the
    ## membership for each of k -gaussians
    Q = np.zeros((num_samples, num_clusters))
    for k in range(num_clusters):
        Q[:, k] = alpha[k] * gaussian_pdf(data, mu[k], sigma[k])

    ## Normalize so that the responsibility matrix is row stochastic
    Q = (Q.T / np.sum(Q, axis = 1)).T

    return Q

def EM_M_step (num_clusters, num_dims, num_samples, Q, data):

    # M Step
    ## calculate the new mean and covariance for each gaussian by
    ## utilizing the new responsibilities
    mu      = np.zeros((num_clusters, num_dims))
    sigma   = np.zeros((num_clusters, num_dims, num_dims))
    alpha = np.zeros(num_clusters)

    ## The number of datapoints belonging to each gaussian
    num_samples_per_cluster = np.sum(Q, axis = 0)

    for k in range(num_clusters):
        ## means
        mu[k] = 1. / num_samples_per_cluster[k] * np.sum(Q[:, k] * data.T, axis = 1).T
        centered_data = np.matrix(data - mu[k])

        ## covariances
        sigma[k] = np.array(1. / num_samples_per_cluster[k] * np.dot(np.multiply(centered_data.T,  Q[:, k]), centered_data))

        ## and finally the probabilities
        alpha[k] = 1. / (num_clusters*num_samples) * num_samples_per_cluster[k]

    return mu, sigma, alpha

def EM_log_likelihood_calc (num_clusters, num_samples, data, mu, sigma, alpha):
    L = np.zeros((num_samples, num_clusters))
    for k in range(num_clusters):
        L[:, k] = alpha[k] * gaussian_pdf(data, mu[k], sigma[k])
    return np.sum(np.log(np.sum(L, axis = 1)))


def EM_calc (num_dims, num_samples, num_clusters, x):
    log_likelihoods = []
    labels 			= []
    iter_cnt        = 0
    epsilon         = 0.0001
    max_iters       = 200
    update          = 2*epsilon

    # initial guess
    mu, sigma, alpha = EM_initial_guess(num_dims, x, num_samples, num_clusters)
    mus = [mu]
    sigmas = [sigma]
    alphas = [alpha]
    while (update > epsilon) and (iter_cnt < max_iters):
        iter_cnt += 1

        # E - Step
        Q = EM_E_step (num_clusters, num_samples, x, mu, sigma, alpha)

        # M - Step
        mu, sigma, alpha = EM_M_step (num_clusters, num_dims, num_samples, Q, x)

        mus.append(mu)
        sigmas.append(sigma)
        alphas.append(alpha)

        # Likelihood computation
        log_likelihoods.append(EM_log_likelihood_calc(num_clusters, num_samples, x, mu, sigma, alpha))

        # check convergence
        if iter_cnt >= 2 :
            update = np.abs(log_likelihoods[-1] - log_likelihoods[-2])

        # logging
        print("iteration {}, update {}".format(iter_cnt, update))

        # print current iteration
        labels.append(np.argmax(Q, axis = 1))

    return labels, log_likelihoods, {'mu': mus, 'sigma': sigmas, 'alpha': alphas}

def kmeans_initial_guess (data, num_samples, num_clusters):
    # randomly choose the starting centroids/means
    # as num_clusters of the points from datasets
    mu = data[np.random.choice(num_samples, num_clusters, False), :]
    return mu

def kmeans_get_labels(num_clusters, num_samples, num_dims, data, mu):
    # set all dataset points to the best cluster according to minimal distance
    #from centroid of each cluster
    dist = np.zeros((num_clusters, num_samples))
    for k in range(num_clusters):
        dist[k] = np.linalg.norm(data - mu[k], axis=1)

    labels = np.argmin(dist, axis=0)

    return labels

def kmeans_get_means(num_clusters, num_dims, data, labels):
    # Compute the new means given the reclustering of the data
    mu = np.zeros((num_clusters, num_dims))
    for k in range(num_clusters):
        idx_list = np.where(labels == k)[0]
        if (len(idx_list) == 0):
            r = np.random.randint(len(data))
            mu[k] = data[r,:]
        else:
            mu[k] = np.mean(data[idx_list], axis=0)
    return mu

def kmeans_calc_loss(num_clusters, num_samples, data, mu, labels):
    dist = np.zeros((num_samples, num_clusters))
    for j in range(num_samples):
        for k in range(num_clusters):
            if (labels[j] == k) :
                dist[j,k] = np.linalg.norm(data[j] - mu[k])
    return sum(sum(dist))


def k_means_calc (num_dims, num_samples, num_clusters, x):
    loss            = []
    labels			= []
    iter_cnt        = 0
    epsilon         = 0.00001
    max_iters       = 100
    update          = 2*epsilon

    # initial guess
    mu = [kmeans_initial_guess(x, num_samples, num_clusters)]

    while (update > epsilon) and (iter_cnt < max_iters):
        iter_cnt += 1
        # Assign labels to each datapoint based on centroid
        labels.append(kmeans_get_labels(num_clusters, num_samples, num_dims, x, mu[-1]))

        # Assign centroid based on labels
        mu.append(kmeans_get_means(num_clusters, num_dims, x, labels[-1]))
        # check convergence
        if iter_cnt >= 2 :
            update = np.linalg.norm(mu[-1] - mu[-2], None)

        # Print distance to centroids vs iteration
        loss.append(kmeans_calc_loss(num_clusters, num_samples, x, mu[-1], labels[-1]))

        # logging
        print("iteration {}, update {}".format(iter_cnt, update))


    return labels, loss, mu

def k_qda_initial_guess (num_dims, data, num_samples, num_clusters):
	# randomly choose the starting centroids/means
	# as num_clusters of the points from datasets
    mu = data[np.random.choice(num_samples, num_clusters, False), :]

	# initialize the covariance matrice for each gaussian
    sigma = [np.eye(num_dims)] * num_clusters

    return mu, sigma

def k_qda_get_parms(num_clusters, num_dims, data, labels):
	## calculate the new mean and covariance for each gaussian
    mu      = np.zeros((num_clusters, num_dims))
    sigma   = np.zeros((num_clusters, num_dims, num_dims))

    for k in range(num_clusters):
        c_k = labels==k
        if (len(data[c_k]) == 0):
            r = np.random.randint(len(data))
            mu[k] = data[r,:]
        else:
            mu[k] = np.mean(data[c_k], axis=0)

        if (len(data[c_k]) > 1):
            centered_data = np.matrix(data[c_k] - mu[k])
            sigma[k] = np.array(1. / len(data[c_k]) * np.dot(centered_data.T, centered_data))
        else:
            sigma[k] = np.eye(num_dims)

    return mu, sigma

def k_qda_get_labels(num_clusters, num_samples, mu, sigma, data):
    # set all dataset points to the best cluster according to best
    # probability given calculated means and covariances
    dist = np.zeros((num_clusters, num_samples))
    for k in range(num_clusters):
        data_center = (data - mu[k])
        dist[k] = np.einsum('ij, ij -> i', data_center, np.dot(np.linalg.inv(sigma[k]) , data_center.T).T )
        labels = np.argmin(dist, axis=0)
    return labels


def k_qda_calc(num_dims, num_samples, num_clusters, x):
    loss            = []
    labels			= []
    iter_cnt        = 0
    epsilon         = 0.00001
    max_iters       = 100
    update          = 2*epsilon

    # initial guess
    mu, sigma = k_qda_initial_guess(num_dims, x, num_samples, num_clusters)
    mus = [mu]
    sigmas = [sigma]
    while (update > epsilon) and (iter_cnt < max_iters):
        iter_cnt += 1

	   # Assign labels to each datapoint based on probability
        labels.append(k_qda_get_labels(num_clusters, num_samples, mus[-1], sigmas[-1], x))

	   # Assign centroid and covarince based on labels
        mu, sigma = k_qda_get_parms(num_clusters, num_dims, x, labels[-1])

        mus.append(mu)
        sigmas.append(sigma)

	   # check convergence
        if iter_cnt >= 2 :
            update = np.linalg.norm(mus[-1] - mus[-2], None)
            update += np.linalg.norm(sigmas[-1] - sigmas[-2], None)

	   # logging
        print("iteration {}, update {}".format(iter_cnt, update))
    return labels, {'mu': mus, 'sigma': sigmas}


def experiments(seed, factor, dir='plots', num_samples=500, num_clusters=3):

    if not os.path.exists(dir):
        os.makedirs(dir)

    np.random.seed(seed)
    num_dims     = 2

    # generate data samples
    (mu, sigma) = gauss_params_gen(num_clusters, num_dims, factor)
    x, true_labels   = data_gen(mu, sigma, num_clusters, num_samples)
    #### Expectation-Maximization
    EM_labels, log_likelihoods, EM_parms = EM_calc (num_dims, num_samples, num_clusters, x)
    #### K QDA
    kqda_labels, kqda_parms = k_qda_calc(num_dims, num_samples, num_clusters, x)
    #### K means
    kmeans_labels, loss, kmean_mus = k_means_calc (num_dims, num_samples, num_clusters, x)

    #Collect all results
    labels = [true_labels, EM_labels[-1], kqda_labels[-1], kmeans_labels[-1]]
    mus_fin = np.array([mu, EM_parms['mu'][-1], kqda_parms['mu'][-1], kmean_mus[-1]])
    algs = np.array(['True', 'EM', 'KQDA', 'Kmeans'])

    #### Plot
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, (lbl,alg, mu) in enumerate(zip(labels, algs, mus_fin)):
        ax = fig.add_subplot(2, 2, i+1)
        data2D_plot(ax, x, lbl, mu, 'viridis', alg)

    fname = os.path.join(dir, 'Results_s{}_f{}_n{}_k{}.png'.format(seed,factor,num_samples, num_clusters))
    fig.savefig(fname)

    mus_init = np.array([mu, EM_parms['mu'][0], kqda_parms['mu'][0], kmean_mus[0]])
    init_mu_fname = os.path.join(dir, 'init_mu_s{}_f{}_n{}_k{}.png'.format(seed,factor, num_samples, num_clusters))
    plot_init_means(x, mus_init, algs, init_mu_fname)




# # 5.a

# In[14]:


if __name__ == "__main__":
    # Question asks us to run with factor =1. Error when factor=1.
    experiments(seed=11, factor=1.1, num_samples=500, num_clusters=3)


# Seed: Determines randomness.
# Factor: Determines how far apart the clusters are.
# 
# The initial guesses for centroids of all the methods are determined randomly. 
# 
# Expectation maximization clearly performs the best. In this simulation, EM correctly determines cluster membership. 
# 
# KQDA, a variant of the hard EM method, does not fully determine cluster membership correctly. 
# 
# Kmeans does not fully determine cluster membership correctly. 
# 
# 
# EM works better on data sets that actually have gaussian clusters, it will typically perform much better on these forms of data sets than on those that dont provide a good fit at all. If there is not a good fit, there will be high variance in runtime.
# 
# 
# 
# 
# 

# # 5.b

# In[9]:


if __name__ == "__main__":
    experiments(seed=63, factor=10, num_samples=500, num_clusters=3)


# Seed: Determines randomness
# Factor: Determines how far apart the clusters are 
# 
# Increasing factor size increases the distance between the clusters. In this case, all methods for classification correctly determine cluster membership. EM has the fastest convergence, when there is not too much data missing, therefore it is the preferred method. 
# 
# 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4433949/
