
# coding: utf-8

# # Question 2: Gradient Boosting & Early Stopping

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from sklearn.datasets import make_sparse_coded_signal
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier

# globals
n_estimators = 200
DT1 = DecisionTreeClassifier(max_depth=1, min_samples_leaf=15)
DT2 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=15)
DT4 = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15)
DT9 = DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)

"""Loads the training data from the SPAM dataset used in HW12."""
def load_data():
    # load data
    data = loadmat("boosting/datasets/spam_data/spam_data.mat")
    # training data
    data_, labels_ = data["training_data"], np.squeeze(data["training_labels"])
    X_train, y_train = data_, labels_
    # test data
    y_test=[]
    with open("boosting/datasets/spam_data/spam_test_labels.txt","r") as f:
        for l in f.readlines():
            y_test.append(int(l.split(",")[1]))
    y_test = np.array(y_test)
    X_test = data['test_data']

    return X_train, y_train, X_test, y_test

"""Runs the maching pursuit algorithm."""
def mp(y, X, w_true, y_test, X_test):
    train_err = []
    test_err = []
    X_ = X
    y = np.copy(y); X = np.copy(X)
    curr = np.copy(y)
    w_est = np.zeros(len(X[0]))
    for j in range(len(X[0])):
        i = np.argmax(np.abs(np.dot(X.T, curr)))
        col = np.copy(X[:,i])
        # use each column only once
        X[:,i] = 0
        w_est[i] = np.dot(col, curr)
        curr = curr - col*w_est[i]
        # error defined here as ||y - D x_hat||_2
        train_err.append(np.linalg.norm(X_.dot(w_est) - y))
        test_err.append(np.linalg.norm(X_test.dot(w_est)-y_test))

    return w_est, train_err, test_err

if __name__ == "__main__":
    ###### CHANGE THESE VARIABLES TO RUN PROBLEM PARTS
    PART_H = True
    PART_I = True
    PART_K = True
    ######
    X_train, y_train, X_test, y_test = load_data()

    ### PART H
    if PART_H:
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111)
        styles=["k-", "k--", "k-."]
        depths=[1,2,4]
        j=0
        # for each weak classifier, train it and an AdaBoost instance based on it
        for w in [DT1, DT2,DT4]:
            # Weak classifier
            w.fit(X_train, y_train)
            err = 1.0 - w.score(X_train, y_train)
            ax.plot([1, n_estimators], [err] * 2, styles[j],
                label="Decision Tree, max depth %d (DT%d)" % (depths[j],depths[j]))
            # AdaBoost classifier
            ada = AdaBoostClassifier(base_estimator=w,
                                     n_estimators=n_estimators,
                                     random_state=0)
            
            ada_train_err = np.zeros((n_estimators,))
            ada.fit(X_train, y_train)
            for i, y_pred in enumerate(ada.staged_predict(X_train)):
                ada_train_err[i] = zero_one_loss(y_pred, y_train)

            smoothed = []
            # use moving average filter to smooth plots -- done to make easier
            # to see trends; you are encouraged to also plot 'ada_train_err' to
            # see the actual error plots!!
            for i in range(len(ada_train_err)):
                temp = 0.
                counter = 0.
                for k in range(i-5, i+1):
                    if k >= 0: 
                        temp += ada_train_err[k]
                        counter += 1.
                smoothed.append(temp/counter)

            ax.plot(np.arange(n_estimators) + 1, smoothed, styles[j],
                label="AdaBoost on DT%d" % depths[j],
                color="red")

            j += 1

        ax.set_ylim((0.1, 0.3))
        ax.set_yscale('log')
        ax.set_xlabel("Number of Classifiers (for AdaBoost)")
        ax.set_ylabel("Error [%], log scale")
        ax.set_title("Weak Classifiers and AdaBoost vs. Training Error")
        leg = ax.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        plt.show()
        plt.close()

    ### PART I
    if PART_I:
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111)
        # Basline classifier (a "deep" tree)
        DT9.fit(X_train, y_train)
        err = 1.0 - DT9.score(X_test, y_test)
        ax.plot([1, n_estimators], [err] * 2, "k-",
            label="Baseline Classifier -- Decision Tree, max depth 9")
        # AdaBoost
        styles=["k-", "k--", "k-."]
        depths=[1,2,4]
        j=0
        # for each weak classifier, train an AdaBoost instance based on it
        for w in [DT1, DT2, DT4]:

            # AdaBoost classifier
            ada = AdaBoostClassifier(base_estimator=w,
                                     n_estimators=n_estimators,
                                     random_state=0)
            ada_train_err = np.zeros((n_estimators,))

            ada.fit(X_train, y_train)
            for i, y_pred in enumerate(ada.staged_predict(X_test)):
                ada_train_err[i] = zero_one_loss(y_pred, y_test)

            smoothed = []
            # use moving average filter to smooth plots -- done to make easier
            # to see trends; you are encouraged to also plot 'ada_train_err' to 
            # see the actual error plots!!
            for i in range(len(ada_train_err)):
                temp = 0.
                counter = 0.
                for k in range(i-5, i+1):
                    if k >= 0: 
                        temp += ada_train_err[k]
                        counter += 1.
                smoothed.append(temp/counter)

            ax.plot(np.arange(n_estimators) + 1, smoothed, styles[j],
                label="AdaBoost on DT%d" % depths[j],
                color="red")

            j += 1

        ax.set_ylim((0.1, 0.3))
        ax.set_yscale('log')
        ax.set_xlabel("Number of Classifiers (for AdaBoost)")
        ax.set_ylabel("Error [%], log scale")
        ax.set_title("Decision Tree Classifier and AdaBoost vs. Test Error")
        leg = ax.legend(loc='lower right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        plt.show()
        plt.close()

    ### PART K
    if PART_K:
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot(111)

        n_components = 100
        n_features = 30
        n_nonzero_coefs = 5
        # y = Xw; w is a sparse vector
        y_train, X_train, w = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
        # test set
        _, X_test, _ = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
        y_test = np.dot(X_test, w)

        np.random.seed(10)
        y_noised_train = y_train + 2e-1*np.random.randn(len(y_train))
        y_noised_test = y_test + 2e-1*np.random.randn(len(y_test))
        w_est, train_err, test_err = mp(y_noised_train, X_train, w,
                                        y_noised_test, X_test)
        
        ax.plot(np.arange(n_components), test_err, label="Maching Pursuit test error")
        ax.plot(np.arange(n_components), train_err, label="Maching Pursuit train error")
        ax.set_ylim((0., 2.0))
        ax.set_xlabel("Number of features used")
        ax.set_ylabel("Reconstruction error")
        ax.set_title("Maching Pursuit Train and Test Reconstruction Error")
        
        leg = ax.legend(loc='upper right', fancybox=True)
        leg.get_frame().set_alpha(0.7)
        
        plt.show()
        plt.close()



# ## Part H Observations
# 
# - All adaboost function errors decrease as a function of increasing classifiers/number of trees we are fitting. Adaboost results in lower error than standard decision trees. 
# 
# 
# 
# - Deeper trees result in lower training error. 
# 
# 
# 
# - **Why?:** Deeper trees mean that the data points can be classified to a greater extent. Deeper trees = more classes, greater fit and less potential for erroneously mislabeling a data point at each iteration of adaboost algorithm. If less points are mislabeled at each iteration than those data points that are mislabeled get higher probabilities. This means they will be more likeley to be sampled from the feature matrix X to be classified in the tree of the next iteration of the adaboost algorithm. The deeper tree means that the data points that we're missclassified (and that given their higher probability of being sampled we're sampled) previously are less likely to be misclassified again in the next tree. 
# 
# 
# 
# - Our loss function is exponential.
# 
# 
# 
# 
# 
# ## Part I Observations
# 
# - Yes, there is a difference in the behavior of the training and test error. The decision tree of depth 1 works best for AdaBoost algorithm on the test data. This is potentially due to overfitting which arise when the depth of the trees is too high. 
# 
# 
# - Decision tree depth 4 has lowest error for training data. Greater depth leads to lower bias and thus lower error. However, the plot of the testing error performance in part I shows that this overfits and does not generalize well to the testing data. 
# 
# 
# - Test error decreases as a function of boosting iterations in the beginning but eventually it starts to increase when the number of decision trees in Adaboostis pretty large.
# 
# 
# 
# 
# 
# ## Part J
# 
# - **Do you think limiting the number of base classifiers used for AdaBoost would help?** Yes. From the plots we see that the error initially decreases up to a certain point (to a minimum error) as a function of the number of base classifiers used for adaboost. Intuitively it makes sense that we would want to limit/use up to X number of classifiers before the error begins to increase again as it is shown to do for the test data in the plot of part I.
# 
# 
# - **Which base classifier can we run more boosting iterations on before the test error starts increasing?** 
# 
# - *Types of base classifiers(?): Tree based classifiers, Random Forests, Bagging, Bayes classifier * 
# 
# 
# Tree Based Decision Tree Classifiers are the most common. 
# 
# "Each round of boosting chooses a new classifier from a set of potential classifiers constructed from training data weighted, or resampled, according to the mis-classifications of the previous round. The new classifier is selected so as to minimise the total ensemble error.
# 
# Each new base classifier is specifically required to focus on the weak points of the existing ensemble, with the result that boosting aggressively drives down training error."
# 
# In early stage of ensemble, boosting has few weak classifiers, each focused on different areas of training. Primarily reduces bias, as ensemble size grows, scope for bias reduction decreases and the error from variance is improved. 
# 
# Instability in base classifiers for boosting is good because as the ensemble grows, the number of remaining mis-classified examples decreases. Thus a higher degree of difference is needed to generated a classifier with a different view of the remaning samples than its predecessors (something that we do not want).
# 
# We can run more boosting iterations on instable base classifiers before the test error starts increasisng. 
# 
# 
# 
# https://stats.stackexchange.com/questions/25121/base-classifiers-for-boosting
# 
# 
# 
# 
# ## Part K Observations
# 
# - **Explain the shape of the training error plot.  Does the plot for test error look similar to the one from part (i)** Yes, the plot for test error looks very similar to the test error plot for adaboost as a function of number of classifiers utilized. Similarily the training data error plot looks very similar to the plot in part H, the curves follow an exponential decay as well, the loss function is exponential with respect to the number of features used. 
# 
# 
# - For MP, the behavior is the same as that of adaboost, however, its behavior is the same as a function of number of features used rather than of number of classifiers used for adaboost. 
# 
# 
# - **The shape of the training error plot:** As t increases matching pursuit builds an estimate using a greater number of features. It fits the data closer. In the case of the training data, it reduces the bias and hence the error. As discussed earlier for the adaboost algorithm this overfits as can be seen from the behavior of the test data. MP works on sparcity of the data set, if we used too many features than the estimator has been overfitted and performs worse (at a certain point) as can be seen in the plot. 
# 

# ## Question 3: CNN On Fruits

# ## 3.a

# In[6]:



import numpy as np
import tensorflow as tf
#import yolo.config_card as cfg

import IPython

slim = tf.contrib.slim


class CNN(object):

    def __init__(self,classes,image_size):
        '''
        Initializes the size of the network
        '''

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size

        self.output_size = self.num_class
        self.batch_size = 40

        self.images = tf.placeholder(tf.float32, [None, self.image_size,self.image_size,3], name='images')


        self.logits = self.build_network(self.images, num_outputs=self.output_size)

        self.labels = tf.placeholder(tf.float32, [None, self.num_class])

        self.loss_layer(self.logits, self.labels)
        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      scope='yolo'):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):

                '''
                Fill in network architecutre here
                Network should start out with the images function
                Then it should return net
                '''
                
                ###SLIM BY DEFAULT ADDS A RELU AT THE END OF conv2d and fully_connected
                ###SLIM SPECIFYING A CONV LAYER WITH 5 FILters as SIZE 15 by 15
                net = slim.conv2d(images, 5, [15,15], scope='conv_0')
                ### SLIM USING MAX POOLING ON THE NETWORK. THE POOLING REGION CONSIDERED IS 3 by 3
                net = slim.max_pool2d(net, [3, 3], scope='pool')
                ## Need to flatten because convolution happens in 2D arrays
                ## and fully connected layers are usually done in 1D arrays.
                net = slim.flatten(net)
                ###SLIM SPECIFYING A FULLY CONNECTED LAYER WHOSE OUT IS 512
                net = slim.fully_connected(net, 512, scope='fc_2')
                ###SLIM SPECIFYING A FULLY CONNECTED LAYER WHOSE OUT IS 25
                net = slim.fully_connected(net, 25, scope='fc_3', activation_fn=None)
                

        return net



    def get_acc(self,y_,y_out):

        '''
        compute accurracy given two tensorflows arrays
        y_ (the true label) and y_out (the predict label)
        '''

        cp = tf.equal(tf.argmax(y_out,1), tf.argmax(y_,1))

        ac = tf.reduce_mean(tf.cast(cp, tf.float32))
        return ac

    def loss_layer(self, predicts, classes, scope='loss_layer'):
        '''
        The loss layer of the network, which is written for you.
        You need to fill in get_accuracy to report the performance
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

            self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = classes,logits = predicts))

            self.accurracy = self.get_acc(classes,predicts)


# In[7]:


from release_code_cnn_tensorflow.data_manager import data_manager
#from release_code_cnn_tensorflow.cnn import CNN
from release_code_cnn_tensorflow.trainer import Solver
import tensorflow as tf
import random

from release_code_cnn_tensorflow.confusion_mat import Confusion_Matrix

random.seed(0)

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape','orange','strawberry','pineapple', 
    'radish','carrot','potato','tomato','bellpepper','broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

image_size = 90
classes = CLASS_LABELS
dm = data_manager(classes, image_size)

cnn = CNN(classes,image_size)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

val_data = dm.val_data
train_data = dm.train_data



cm = Confusion_Matrix(val_data,train_data,CLASS_LABELS,sess)

cm.test_net(cnn)

