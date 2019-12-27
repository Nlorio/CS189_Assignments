
# coding: utf-8

# # 3. Bias and Variance of Sparse Linear Regression
# 

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


def ground_truth(n, d, s):
    """
    Input: Two positive integers n, d. Requires n >= d >=s. If d<s, we let s = d
    Output: A tuple containing i) random matrix of dimension n X d with orthonormal columns. and
             ii) a d-dimensional, s-sparse wstar with (large) Gaussian entries
    """
    if d > n:
        print("Too many dimensions")
        return None
    
    if d < s:
        s = d
    A = np.random.randn(n, d) #random Gaussian matrix
    U, S, V = np.linalg.svd(A, full_matrices=False) #reduced SVD of Gaussian matrix
    wstar = np.zeros(d)
    wstar[:s] = 10 * np.random.randn(s)
    
    np.random.shuffle(wstar)
    return U, wstar

def get_obs(U, wstar):
    """
    Input: U is an n X d matrix and wstar is a d X 1 vector.
    Output: Returns the n-dimensional noisy observation y = U * wstar + z.
    """
    n, d = np.shape(U)
    z = np.random.randn(n) #i.i.d. noise of variance 1
    y = np.dot(U, wstar) + z
    return y


# In[8]:


def LS(U, y):
    """
    Input: U is an n X d matrix with orthonormal columns and y is an n X 1 vector.
    Output: The OLS estimate what_{LS}, a d X 1 vector.
    """
    wls = np.dot(U.T, y) #pseudoinverse of orthonormal matrix is its transpose
    return wls


def thresh(U, y, lmbda):
    """
    Input: U is an n X d matrix and y is an n X 1 vector; lambda is a scalar threshold of the entries.
    Output: The estimate what_{T}(lambda), a d X 1 vector that is hard-thresholded (in absolute value) at level lambda.
            When code is unfilled, returns the all-zero d-vector.
    """
    n, d = np.shape(U)
    wls = LS(U, y)
    what = np.zeros(d)
    
    #print np.shape(wls)
    ##########
    #TODO: Fill in thresholding function; store result in what
    #####################
    #YOUR CODE HERE:
    
    for i in range(len(what)):
        if np.abs(wls[i]) > lmbda:
            what[i] = wls[i]
        

    ###############
    return what
    
    
def topk(U, y, s):
    """
    Input: U is an n X d matrix and y is an n X 1 vector; s is a positive integer.
    Output: The estimate what_{top}(s), a d X 1 vector that has at most s non-zero entries.
            When code is unfilled, returns the all-zero d-vector.
    """
    n, d = np.shape(U)
    what = np.zeros(d)
    wls = LS(U, y)
    
    ##########
    #TODO: Fill in thresholding function; store result in what
    #####################
    #YOUR CODE HERE: Remember the absolute value!
    inds = np.argpartition(np.abs(wls), -s)[-s:]
    what[inds] = wls[inds]
    ###############
    return what


# In[9]:


def error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=True, true_s=5):
    """
    Plots the prediction error 1/n || U(what - wstar)||^2 = 1/n || what - wstar ||^2 for the three estimators
    averaged over num_iter experiments.
    
    Input:
    Output: 4 arrays -- range of parameters, errors of LS, topk, and thresh estimator, respectively. If thresh and topk
            functions have not been implemented yet, then these errors are simply the norm of wstar.
    """
    wls_error = []
    wtopk_error = []
    wthresh_error = []
    
    if param == 'n':
        arg_range = np.arange(100, 2000, 50)
        lmbda = 2 * np.sqrt(np.log(d))
        for n in arg_range:
            U, wstar = ground_truth(n, d, s) if s_model else ground_truth(n, d, true_s)
            error_wls = 0
            error_wtopk = 0
            error_wthresh = 0
            for count in range(num_iters):
                y = get_obs(U, wstar)
                wls = LS(U, y)
                wtopk = topk(U, y, s)
                wthresh = thresh(U, y, lmbda)
                error_wls += np.linalg.norm(wstar - wls)**2
                error_wtopk += np.linalg.norm(wstar - wtopk)**2
                error_wthresh += np.linalg.norm(wstar - wthresh)**2
            wls_error.append(float(error_wls)/ n / num_iters)
            wtopk_error.append(float(error_wtopk)/ n / num_iters)
            wthresh_error.append(float(error_wthresh)/ n / num_iters)
        
    elif param == 'd':
        arg_range = np.arange(10, 1000, 50)
        for d in arg_range:
            lmbda = 2 * np.sqrt(np.log(d))
            U, wstar = ground_truth(n, d, s) if s_model else ground_truth(n, d, true_s)
            error_wls = 0
            error_wtopk = 0
            error_wthresh = 0
            for count in range(num_iters):
                y = get_obs(U, wstar)
                wls = LS(U, y)
                wtopk = topk(U, y, s)
                wthresh = thresh(U, y, lmbda)
                error_wls += np.linalg.norm(wstar - wls)**2
                error_wtopk += np.linalg.norm(wstar - wtopk)**2
                error_wthresh += np.linalg.norm(wstar - wthresh)**2
            wls_error.append(float(error_wls)/ n / num_iters)
            wtopk_error.append(float(error_wtopk)/ n / num_iters)
            wthresh_error.append(float(error_wthresh)/ n / num_iters)
    
    elif param == 's':
        arg_range = np.arange(5, 55, 5)
        lmbda = 2 * np.sqrt(np.log(d))
        for s in arg_range:
            U, wstar = ground_truth(n, d, s) if s_model else ground_truth(n, d, true_s)
            error_wls = 0
            error_wtopk = 0
            error_wthresh = 0
            for count in range(num_iters):
                y = get_obs(U, wstar)
                wls = LS(U, y)
                wtopk = topk(U, y, s)
                wthresh = thresh(U, y, lmbda)
                error_wls += np.linalg.norm(wstar - wls)**2
                error_wtopk += np.linalg.norm(wstar - wtopk)**2
                error_wthresh += np.linalg.norm(wstar - wthresh)**2
            wls_error.append(float(error_wls)/ n / num_iters)
            wtopk_error.append(float(error_wtopk)/ n / num_iters)
            wthresh_error.append(float(error_wthresh)/ n / num_iters)
    
    return arg_range, wls_error, wtopk_error, wthresh_error


# # a 

# In[10]:


#nrange contains the range of n used, ls_error the corresponding errors for the OLS estimate
nrange, ls_error, _, _ = error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=True, true_s=5)
########
#TODO: Your code here: call the helper function for d and s, and plot everything
nrange_d, ls_error_d, _, _ = error_calc(num_iters=10, param='d', n=1000, d=100, s=5, s_model=True, true_s=5)
nrange_s, ls_error_s, _, _ = error_calc(num_iters=10, param='s', n=1000, d=100, s=5, s_model=True, true_s=5)

########
#YOUR CODE HERE:

plt.title("n log-log")
plt.plot(np.log(nrange), np.log(ls_error))
plt.xlabel('log(nrange_n)')
plt.ylabel('log(ls_error_n)')
plt.show()

plt.title("d log-log")
plt.plot(np.log(nrange_d), np.log(ls_error_d))
plt.xlabel('log(range_d)')
plt.ylabel('log(ls_error_d)')
plt.show()

plt.title("s log-log")
plt.plot(np.log(nrange_s), np.log(ls_error_s))
plt.xlabel('log(range_s)')
plt.ylabel('log(ls_error_s)')
plt.show()


# Are these plots as expected? Discuss. Also put down your parameter choices (either here or in plot captions.) It's fine to use the default values, but put them down nonetheless.
# 
# **Answer Here**
# 
# Paramaters for plot of error as function of n:
# 
# 
# n=1000, d=100, s=5, s_model=True, true_s=5
# 
# 
# Paramaters for plot of error as function of d:
# 
# 
# n=1000, d=100, s=5, s_model=True, true_s=5
# 
# 
# Paramaters for plot of error as function of s:
# 
# 
# n=1000, d=100, s=5, s_model=True, true_s=5
# 
# 
# - **First Plot: ** Error decreases as we increase the number of data points n. First plot makes intutive sense. The bias decreases as a function of n increasing. 
# 
# 
# 
# - **Second Plot: ** Error increases as the degree increases. The number of data points n is fixed. The complexity increases as the degree of the polynomial increases. Increasing complexity increases the variance which contributes to the increased error. 
# 
# 
# 
# - **Third Plot: ** Error is not correlated with the sparcity. 
# 
# 
# 
# 

# # b

# In[17]:


#TODO: Part (b)
##############
#YOUR CODE HERE:

nrange, ls_error, thresh_error_n, topk_error_n = error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=True, true_s=5)
nrange_d, ls_error_d, thresh_error_d, topk_error_d = error_calc(num_iters=10, param='d', n=1000, d=100, s=5, s_model=True, true_s=5)
nrange_s, ls_error_s, thresh_error_s, topk_error_s = error_calc(num_iters=10, param='s', n=1000, d=100, s=5, s_model=True, true_s=5)



plt.title("Error of all estimators as a function of n")
plt.plot(np.log(nrange), np.log(ls_error), label = 'OLS')
plt.plot(np.log(nrange), np.log(thresh_error_n), label = 'thresh')
plt.plot(np.log(nrange), np.log(topk_error_n), label = 'topk')
plt.xlabel('log(n)')
plt.ylabel('log(Error)')
plt.legend(loc='upper right', shadow=True)

plt.show()

plt.title("Error of all estimators as a function of d")
plt.plot(np.log(nrange_d), np.log(ls_error_d), label = 'OLS')
plt.plot(np.log(nrange_d), np.log(thresh_error_d), label = 'thresh')
plt.plot(np.log(nrange_d), np.log(topk_error_d), label = 'topk')
plt.xlabel('log(n)')
plt.ylabel('log(Error_d)')
plt.legend(loc='upper left', shadow=True)



plt.show()


plt.title("Error of all estimators as a function of s")
plt.plot(np.log(nrange_s), np.log(ls_error_s), label = 'OLS')
plt.plot(np.log(nrange_s), np.log(thresh_error_s), label = 'thresh')
plt.plot(np.log(nrange_s), np.log(topk_error_s), label = 'topk')
plt.xlabel('log(n)')
plt.ylabel('log(Error_s)')
plt.legend(loc='lower right', shadow=True)



plt.show()




# # c

# In[18]:


#TODO: Part (c)
##############
#YOUR CODE HERE:
nrange, ls_error, thresh_error_n, topk_error_n = error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=False, true_s=100)
nrange_d, ls_error_d, thresh_error_d, topk_error_d = error_calc(num_iters=10, param='d', n=1000, d=100, s=5, s_model=False, true_s=100)
nrange_s, ls_error_s, thresh_error_s, topk_error_s = error_calc(num_iters=10, param='s', n=1000, d=100, s=5, s_model=False, true_s=100)


plt.title("Error of all estimators as a function of n")
plt.plot(np.log(nrange), np.log(ls_error), label = 'OLS')
plt.plot(np.log(nrange), np.log(thresh_error_n), label = 'thresh')
plt.plot(np.log(nrange), np.log(topk_error_n), label = 'topk')
plt.xlabel('log(n)')
plt.ylabel('log(Error)')
plt.legend(loc='upper right', shadow=True)

plt.show()

plt.title("Error of all estimators as a function of d")
plt.plot(np.log(nrange_d), np.log(ls_error_d), label = 'OLS')
plt.plot(np.log(nrange_d), np.log(thresh_error_d), label = 'thresh')
plt.plot(np.log(nrange_d), np.log(topk_error_d), label = 'topk')
plt.xlabel('log(n)')
plt.ylabel('log(Error_d)')
plt.legend(loc='upper left', shadow=True)



plt.show()


plt.title("Error of all estimators as a function of s")
plt.plot(np.log(nrange_s), np.log(ls_error_s), label = 'OLS')
plt.plot(np.log(nrange_s), np.log(thresh_error_s), label = 'thresh')
plt.plot(np.log(nrange_s), np.log(topk_error_s), label = 'topk')
plt.xlabel('log(n)')
plt.ylabel('log(Error_s)')
plt.legend(loc='lower right', shadow=True)



plt.show()


# **Param First Plot **
# 
# error_calc(num_iters=10, param='n', n=1000, d=100, s=5, s_model=False, true_s=100)
# 
# **Param Second Plot **
# 
# error_calc(num_iters=10, param='d', n=1000, d=100, s=5, s_model=False, true_s=100)
# 
# **Param Third Plot **
# 
# error_calc(num_iters=10, param='s', n=1000, d=100, s=5, s_model=False, true_s=100)
# 
# 
# **Answer Here**
# 
# 
# 
# - **First Plot: ** Error as a function of n decreases as n increases. The bias decreases. For data from a non-sparse linear model: Same statements made earlier regarding the general behavior of the OLS plots are true here for for the plotted error of the estimators and the OLS as a function of n.
# 
# 
# 
# - **Second Plot: ** For data from a non-sparse linear model: Same statements made earlier regarding the general behavior of the OLS plots are true here for the plotted error of the estimators and the OLS as a function of d.
# 
# *From Part a*
# 
# 
# "Error increases as the degree increases. The number of data points n is fixed. The complexity increases as the degree of the polynomial increases. Increasing complexity increases the variance which contributes to the increased error. 
# "
# 
# 
# - **Third Plot: ** As we increase the sparsity, we note that this reduces the error as the sparsity of the true model approaches the sparcity our estimators assume the data to have. 
# 
# 
# 
# The true sparsity, s, of the model is greater than the s-sparsity assumed for our estimators. Our estimators want sparse data, the estimators will incorrectly estimate at most (true_s - s) weights in $w^{hat}$ as zero when they are in fact non-zero/non-sparse coefficients of the weight vector. Thus our estimates have worse performance as they are assuming sparsity in the data when the true data is generated from a non-sparse linear model. 

# # 4.  Decision Trees and Random Forests

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## a)

# In[28]:


# You may want to install "gprof2dot"
import io
from collections import Counter

import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.tree
from numpy import genfromtxt
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

import pydot

eps = 1e-5  # a small number

class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        def entropy(ys):
            total = len(ys)
            if total == 0:
                return 0
            arr = [0, 0]
            for label in ys:
                arr[label] += 1
            return sum([-(pi/total)*np.log((pi/total)) for pi in arr if pi != 0])
        total = len(y)
        y_left = []
        y_right = []
        for i in range(total):
            if X[i] < thresh:
                y_left.append(y[i])
            else:
                y_right.append(y[i])
        return entropy(y) - (len(y_left)/total)*entropy(y_left) - (len(y_right)/total)*entropy(y_right)
 
    @staticmethod
    def gini_purification(X, y, thresh):
        # TODO implement gini_purification function
        def gini(ys):
            total = len(ys)
            if total == 0:
                return 0
            arr = [0, 0]
            for label in ys:
                arr[label] += 1
            return 1 - sum([(pi/total)**2 for pi in arr])
        total = len(y)
        y_left = []
        y_right = []
        for i in range(total):
            if X[i] < thresh:
                y_left.append(y[i])
            else:
                y_right.append(y[i])
        return gini(y) - (len(y_left)/total)*gini(y_left) - (len(y_right)/total)*gini(y_right)
        
        
    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth > 0:
            # compute entropy gain for all single-dimension splits,
            # thresholding with a linear interpolation of 10 values
            gains = []
            # The following logic prevents thresholding on exactly the minimum
            # or maximum values, which may not lead to any meaningful node
            # splits.
            thresh = np.array([
                np.linspace(np.min(X[:, i]) + eps, np.max(X[:, i]) - eps, num=10)
                for i in range(X.shape[1])
            ])
            for i in range(X.shape[1]):
                gains.append([self.information_gain(X[:, i], y, t) for t in thresh[i, :]])

            gains = np.nan_to_num(np.array(gains))
            self.split_idx, thresh_idx = np.unravel_index(np.argmax(gains), gains.shape)
            self.thresh = thresh[self.split_idx, thresh_idx]
            X0, y0, X1, y1 = self.split(X, y, idx=self.split_idx, thresh=self.thresh)
            if X0.size > 0 and X1.size > 0:
                self.left = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.left.fit(X0, y0)
                self.right = DecisionTree(
                    max_depth=self.max_depth - 1, feature_labels=self.features)
                self.right.fit(X1, y1)
            else:
                self.max_depth = 0
                self.data, self.labels = X, y
                self.pred = stats.mode(y).mode[0]
        else:
            self.data, self.labels = X, y
            self.pred = stats.mode(y).mode[0]
        return self

    def predict(self, X):
        if self.max_depth == 0:
            return self.pred * np.ones(X.shape[0])
        else:
            X0, idx0, X1, idx1 = self.split_test(X, idx=self.split_idx, thresh=self.thresh)
            yhat = np.zeros(X.shape[0])
            yhat[idx0] = self.left.predict(X0)
            yhat[idx1] = self.right.predict(X1)
            return yhat

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        n = X.shape[0]
        for tree in self.decision_trees:
            indices = np.random.randint(0, n, n)
            tree.fit(X[indices], y[indices])
        return self

    def predict(self, X):
        result = []
        for tree in self.decision_trees:
            result.append(tree.predict(X))
        return np.array(np.round(np.mean(result, axis=0)))


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        # TODO implement function
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, max_features=m, **self.params)
            for i in range(self.n)
        ]
    


class BoostedRandomForest(RandomForest):
    
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO implement function
        for j, tree in enumerate(self.decision_trees):
            indices = np.random.choice(X.shape[0], size=X.shape[0], replace=True, p=self.w)
            tree.fit(X[indices], y[indices])
            yhat = tree.predict(X)
            
            e_j = 0
            incorrect = []
            i = 0
            for y_i, yhat_i in zip(y,yhat):
                if y_i != yhat_i:
                    e_j += self.w[i]
                    incorrect.append(1)
                else:
                    incorrect.append(0)
                i += 1
                
            self.a[j] = (1/2)*np.log((1-e_j)/e_j)
            
            i = 0
            for indicator in incorrect:
                if indicator == 1:
                    self.w[i]*=(np.exp(self.a[j]))
                else:
                    self.w[i]*=(np.exp(-self.a[j]))
                i += 1
                
            self.w = self.w / np.sum(self.w)
            
        return self

    def predict(self, X):
        # TODO implement function
        pred_0 = np.zeros(X.shape[0])
        pred_1 = np.zeros(X.shape[0])
        for j, tree in enumerate(self.decision_trees):
            y_hat = tree.predict(X)
            for i, y in enumerate(y_hat):
                if y == 0:
                    pred_0[i] += self.a[j]
                elif y == 1:
                    pred_1[i] += self.a[j]
        result = []
        for z,o in zip(pred_0, pred_1):
            if z > o:
                result.append(0)
            else:
                result.append(1)
        return result


def preprocess(data, fill_mode=True, min_freq=10, onehot_cols=[]):
    # fill_mode = False

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(np.float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack([np.array(data, dtype=np.float), np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    if fill_mode:
        for i in range(data.shape[-1]):
            mode = stats.mode(data[((data[:, i] < -1 - eps) +
                                    (data[:, i] > -1 + eps))][:, i]).mode[0]
            data[(data[:, i] > -1 - eps) * (data[:, i] < -1 + eps)][:, i] = mode

    return data, onehot_features


def evaluate(clf, features, X, y):
    print("Cross validation", sklearn.model_selection.cross_val_score(clf, X, y))
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        for term in counter.most_common():
            z = term[0]
            o = term[1]
            a = features[z]
        first_splits = [(features[term[0]], term[1]) for term in counter.most_common()]
        print("First splits", first_splits)
    
def main(dataset=None):
    if not dataset:
        dataset = "titanic"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data
        path_train = 'titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=np.int)
        print("\n\nPart (b): preprocessing the titanic dataset")
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription", "creative",
            "height", "featured", "differ", "width", "other", "energy", "business", "message",
            "volumes", "revision", "path", "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis", "square_bracket",
            "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features:", features)
    print("Train/test size:", X.shape, Z.shape)

    print("\n\nPart 0: constant classifier")
    print("Accuracy", 1 - np.sum(y) / y.size)

    # Basic decision tree
    print("\n\nPart (a-b): simplified decision tree")
    dt = DecisionTree(max_depth=3, feature_labels=features)
    dt.fit(X, y)
    print("Predictions", dt.predict(Z)[:100])

    print("\n\nPart (c): sklearn's decision tree")
    clf = sklearn.tree.DecisionTreeClassifier(random_state=0, **params)
    clf.fit(X, y)
    evaluate(clf, features, X, y)
    #out = io.StringIO()
    #sklearn.tree.export_graphviz(
    #    clf, out_file=out, feature_names=features, class_names=class_names)
    #graph = pydot.graph_from_dot_data(out.getvalue())
    #pydot.graph_from_dot_data(out.getvalue())[0].write_pdf("%s-tree.pdf" % dataset)

    # TODO implement and evaluate parts d-h
    print("\n\nPart (e): Our Bagged Tree for %s" % dataset)
    clf = BaggedTrees(params)
    clf.fit(X, y)
    evaluate(clf, features, X, y)
    print(repr(clf))
    
    # TODO implement and evaluate parts d-h
    print("\n\nPart (g): Our RandomForest for %s" % dataset )
    clf = RandomForest(params)
    clf.fit(X, y)
    evaluate(clf, features, X, y)
    print(repr(clf))
    
    # TODO implement and evaluate parts d-h
    print("\n\nPart (i, j): Our RandomForest for %s" % dataset )
    clf = BoostedRandomForest(params)
    clf.fit(X, y)
    evaluate(clf, features, X, y)
    print(repr(clf))
    print(clf.predict(Z))


# ## b)
# 
# All answers can be found in preprocess function.
# 
# - Some data points are misssing class labels;
#     
#     \# Temporarily assign -1 to missing data <br>
#     
#     
# - Some features are not numerical values;
# 
#     \# Hash the columns (used for handling strings)
#     
# 
# - Some data points are missing some features.
# 
#     \# Replace missing data with the mode value. We use the mode instead of <br>
#     \# the mean or median because this makes more sense for categorical <br>
#     \# features such as gender or cabin type, which are not ordered. <br>
#     

# ## Titanic Dataset

# In[29]:


if __name__ == "__main__":
    main("titanic")


# ## Spam Dataset

# In[30]:


if __name__ == "__main__":
    main(dataset="spam")


# ## h)
# 
# Part H Observations:
# 
# Qualitatively describe what this algorithm is doing. What does it mean when a_i < 0, how does the algorithm handle such trees? 
# 
# 
# Boosting:
# 
# Fit:
# 
# - We determine the weighted error of the current iteration decision tree. a_i is computed based off of this weighted error in accordance with AdaBoost algorithm.  
# 
# - a_i is always >= 0. The sign assigned to a_i depends on whether or not its respective data point was correctly classified by the current iteration
# 
# 
# For data points that are correctly classified we lower the precedence of said data point by reducing its weight, wi ,by a factor of exp(-a_i).  Vice a versa for incorreclty classified data points, it increases the weight wi on the data point by exp(+a_i). The new weights on the data points are used in the next iteration. 
# 
# Prediction:
# 
# The performance of the classification in the current iteration is determined by the score.
# 
# Our algorithm is repeated until we have M trees and have fit and predicted our classification M times, with updated weights at each iteration.

# # i) 
# 
# 
# - Most challenging data to classify with decision trees are : Data with high variance, as for decision trees small variations in the data might result in a completely different tree being generated. Furthermore if some classes of data dominate than decision tree learners can create biased trees. 
# 
# 
# 
# 
