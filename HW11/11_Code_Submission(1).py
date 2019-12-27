
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


"""
The world_values data set is available online at http://54.227.246.164/dataset/. In the data,
    residents of almost all countries were asked to rank their top 6 'priorities'. Specifically,
    they were asked "Which of these are most important for you and your family?"

This code and world-values.tex guides the student through the process of training several models
    to predict the HDI (Human Development Index) rating of a country from the responses of its
    citizens to the world values data. The new model they will try is k Nearest Neighbors (kNN).
    The students should also try to understand *why* the kNN works well.
"""

from math import sqrt
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from world_values_utils import import_world_values_data
from world_values_utils import hdi_classification
from world_values_utils import calculate_correlations
from world_values_utils import plot_pca

from world_values_pipelines import ridge_regression_pipeline
from world_values_pipelines import lasso_regression_pipeline
from world_values_pipelines import k_nearest_neighbors_regression_pipeline
from world_values_pipelines import svm_classification_pipeline
from world_values_pipelines import k_nearest_neighbors_classification_pipeline
from world_values_pipelines import tree_classification_pipeline

from world_values_parameters import regression_ridge_parameters
from world_values_parameters import regression_lasso_parameters
from world_values_parameters import regression_knn_parameters
from world_values_parameters import classification_svm_parameters
from world_values_parameters import classification_knn_parameters
from world_values_parameters import classification_tree_parameters


def main():
    print("Predicting HDI from World Values Survey")
    print()

    # Import Data #
    print("Importing Training and Testing Data")
    values_train, hdi_train, values_test = import_world_values_data()

    # Center the HDI Values #
    # hdi_scaler = StandardScaler(with_std=False)
    # hdi_shifted_train = hdi_scaler.fit_transform(hdi_train)

    # Classification Data #
    hdi_class_train = hdi_train['2015'].apply(hdi_classification)

    # Data Information #
    print('Training Data Count:', values_train.shape[0])
    print('Test Data Count:', values_test.shape[0])
    print()

    # Part b and c: Calculate Correlations #
    # calculate_correlations(values_train, hdi_train)

    # Part d, r: PCA #
    # plot_pca(values_train, hdi_train, hdi_class_train)

    # Part e,f,and g: Regression Grid Searches #
    # regression_grid_searches(training_features=values_train,training_labels=hdi_train)

    # Part i, m, : Nearest neighbors to the US. n_neighbors is 8 because the first entry is the US.
    # nbrs = NearestNeighbors(n_neighbors=8).fit(values_train)
    # distances, indices = nbrs.kneighbors(values_train.iloc[45])

    # Part j, l, n, t, u, v, w,  Classification Grid Searches #
    # classification_grid_searches(training_features=values_train,training_classes=hdi_class_train)

def _rmse_grid_search(training_features, training_labels, pipeline, parameters, technique):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set
        pipeline: regression model specific pipeline
        parameters: regression model specific parameters
        technique: regression model's name

    Output:
        Prints best RMSE and best estimator
        Prints feature weights for Ridge and Lasso Regression
        Plots RMSE vs k for k Nearest Neighbors Regression
    """
    grid = GridSearchCV(estimator=pipeline,
                        param_grid=parameters,
                        scoring='neg_mean_squared_error')
    grid.fit(training_features,
             training_labels)
    print("RMSE:", sqrt(-grid.best_score_))
    print(grid.best_estimator_)

    # Check Ridge or Lasso Regression
    if hasattr(grid.best_estimator_.named_steps[technique], 'coef_'):
        print("Coefficients")
        print(grid.best_estimator_.named_steps[technique].coef_)
    else:
        # Plot RMSE vs k for k Nearest Neighbors Regression
        plt.plot(grid.cv_results_['param_knn__n_neighbors'],
                 (-grid.cv_results_['mean_test_score'])**0.5)
        plt.xlabel('k')
        plt.ylabel('RMSE')
        plt.title('RMSE versus k in kNN')
        plt.show()

    print()
    return grid


def regression_grid_searches(training_features, training_labels, testing_features=None):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set

    Output:
        Prints best RMSE, best estimator, feature weights for Ridge and Lasso Regression
        Prints best RMSE, best estimator, and plots RMSE vs k for k Nearest Neighbors Regression
    """

    print("Ridge Regression")
    _rmse_grid_search(training_features, training_labels,
                ridge_regression_pipeline, regression_ridge_parameters, 'ridge')

    print("Lasso Regression")
    _rmse_grid_search(training_features, training_labels,
                lasso_regression_pipeline, regression_lasso_parameters, 'lasso')

    print("k Nearest Neighbors Regression")
    grid = _rmse_grid_search(training_features, training_labels,
                k_nearest_neighbors_regression_pipeline,
                regression_knn_parameters, 'knn')

    if testing_features is not None:
        print(grid.predict(testing_features))


def _accuracy_grid_search(training_features, training_classes, pipeline, parameters):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set
        pipeline: classification model specific pipeline
        parameters: classification model specific parameters

    Output:
        Prints best accuracy and best estimator of classification model
    """
    grid = GridSearchCV(estimator=pipeline,
                        param_grid=parameters,
                        scoring='accuracy')
    grid.fit(training_features, training_classes)
    print("Accuracy:", grid.best_score_)
    print(grid.best_estimator_)
    print()
    return grid


def classification_grid_searches(training_features, training_classes):
    """
    Input:
        training_features: world_values responses on the training set
        training_labels: HDI (human development index) on the training set

    Output:
        Prints best accuracy and best estimator for SVM and k Nearest Neighbors Classification
    """
    print("SVM Classification")
    _accuracy_grid_search(training_features, training_classes,
                        svm_classification_pipeline,
                        classification_svm_parameters)

    print("k Nearest Neighbors Classification")
    _accuracy_grid_search(training_features, training_classes,
                        k_nearest_neighbors_classification_pipeline,
                        classification_knn_parameters)

    # print("Decision Tree Classification")
    # decision_grid = _accuracy_grid_search(training_features, training_classes,
    #                                       tree_classification_pipeline,
    #                                       classification_tree_parameters)
    # print("Classes", decision_grid.best_estimator_.named_steps['tree'].classes_)
    # print("Feature Importances", decision_grid.best_estimator_.named_steps['tree'].feature_importances_)
    # estimator = decision_grid.best_estimator_.named_steps['tree']

    # n_nodes = estimator.tree_.node_count
    # print("Node Count", n_nodes)
    # children_left = estimator.tree_.children_left
    # print("Left Children", children_left)
    # children_right = estimator.tree_.children_right
    # print("Right Children", children_right)
    # feature = estimator.tree_.feature
    # print("Feature", feature)
    # threshold = estimator.tree_.threshold
    # print("Threshold", threshold)


if __name__ == '__main__':
    main()


# In[3]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# # 3. Nearest Neighbors for Regression, from A to Z

# ## b)

# In[4]:


print("Predicting HDI from World Values Survey")
print()
# Import Data #
print("Importing Training and Testing Data")
print()
values_train, hdi_train, values_test = import_world_values_data()
# Part b and c: Calculate Correlations #
calculate_correlations(values_train, hdi_train)


# In[5]:


import numpy as np
import pandas as pd

correlations = [0.4733, -0.4396, -0.3362, -0.0182, -0.422, -0.304, 0.3294, -0.3516, -0.2854, 0.1952, 0.6135, 0.1433, 0.2381, 0.4329, 0.2765, -0.3973]

print("Most negatively correlated with HDI: ")
print(np.min(correlations))
print("Better transport and roads -0.4396336386224581, index: 1")
print()

print("Most positively corrleated with HDI: ")
print(np.max(correlations))
print("Protecting forests rivers and oceans 0.6134587562712407, index: 10")
print()

print("Least correlated with HDI: ") 
least = correlations[0]
for i in correlations:
    if np.abs(i - 0) < np.abs(least - 0):
        least = i
print(least)
print("Access to clean water and sanitation -0.018169084455954738, index: 3")


# ## c)

# In[6]:


hdi_col = hdi_train['2015']

negative_values = values_train['Better transport and roads']

plt.scatter(hdi_col, negative_values)
plt.title("Most negatively Correlated")
plt.xlabel("HDI")
plt.ylabel("Better transport and roads")
plt.show()

positive_values = values_train['Protecting forests rivers and oceans']

plt.scatter(hdi_col, positive_values)
plt.title("Most positively Correlated")
plt.xlabel("HDI")
plt.ylabel("Protecting forests rivers and oceans")
plt.show()

least_values = values_train['Access to clean water and sanitation']

plt.scatter(hdi_col, least_values)
plt.title("Least Correlated")
plt.xlabel("HDI")
plt.ylabel("Access to clean water and sanitation")
plt.show()


# ## d)

# In[7]:


print("Predicting HDI from World Values Survey")
print()

# Import Data #
print("Importing Training and Testing Data")
values_train, hdi_train, values_test = import_world_values_data()

# Center the HDI Values #
# hdi_scaler = StandardScaler(with_std=False)
# hdi_shifted_train = hdi_scaler.fit_transform(hdi_train)

# Classification Data #
hdi_class_train = hdi_train['2015']

#Part d, r: PCA #
plot_pca(values_train, hdi_train, hdi_class_train)


# ## e, f, g)

# In[8]:


# Regression Grid Searches for Part e, f, and g
regression_grid_searches(training_features=values_train,training_labels=hdi_train)


# ## g)
# Ridge Regression
# RMSE: 0.12303337350607803
# 
# [[ 0.80823467 -0.74985758 -0.17800015 -1.28408103 -0.66293176 -0.82203172
#    0.73733884 -0.92891581 -0.82049672  0.39614952  2.0708291  -0.06718981
#    0.48310656  0.72671425  0.42921192 -0.13808023]]
# 
#     
# Lasso Regression
# RMSE: 0.12602242808947525
# 
# [ 0.1590192  -0.72844929 -0.         -0.85945074 -0.66274144 -0.02556703
#   0.33904781 -0.29897158 -0.          0.          3.48536375  0. 0.          0.87057995  0.32897045 -0.        ]
# 
# 
# #### Lasso regression gives more 0 weights. The above respective sets of coefficients show this.

# # h)

# - How would you adapt the k nearest neighbors algorithm for a regression problem?
# 
# Let us use a weighted average of the k nearest neighbors, weighted by the inverse of their distances.
# 
# **Algorithm:**
# - Compute euclidian distance from i to the labeled data points. 
# - Order the labeled data points by increasing distance
# - Use cross validation to find optimal number of k nearest neighbors using RMSE. 
# - Calculate inverse distance weighted average with the k-nearest multivariate neighbors.
# 
# Or
# 
# "Take average of the k closest regression values"
# 

# # i)

# In[9]:


from sklearn.neighbors import NearestNeighbors

countries_train = pd.read_csv('world-values-train2.csv')
countries_train = countries_train['Country']

# Nearest neighbors to the US. n_neighbors is 8 because the first entry is the US.
nbrs = NearestNeighbors(n_neighbors=8).fit(values_train)
distances, indices = nbrs.kneighbors([values_train.iloc[45]])

print('These are the 7 nearest neighbors of the USA in order.')
print()
for i in indices:
    print(countries_train[i])


# # j)

# In[10]:


print("k Nearest Neighbors Regression")
grid = _rmse_grid_search(values_train, hdi_train,
            k_nearest_neighbors_regression_pipeline,
            regression_knn_parameters, 'knn')


# #### Best value of k: 12
# ##### RMSE at this point: 0.11824589460776892

# ## 3.k
# 
# - Explain your plot in (j) in terms of bias and variance.
# 
# 
# In general, a large K value is more precise as it reduces the overall noise; however, the compromise is that the distinct boundaries within the feature space are blurred. 10 is typically the best k in practice, however cross validation should be used to find the optimal. 
# 
# 
# When we increase k the bias decreases to a certain point (around 20). We get closer to the true model as we get to this point, but then the error starts to increase because are algorithm is considering too many data points. This increases bias for our model, but more specifically variance error is increasing because our algorithm has increasingly higher sensitivity to small fluctuations in the dataset. Our algorithm begins to model this random noise and we don't get the correct output.
# 
# 
#  

# ## l)

# In[11]:


regression_knn_parameters = {
    # 'pca__n_components': np.arange(1, 17),

    'knn__n_neighbors': np.arange(1, 50),

    # Apply uniform weighting vs k for k Nearest Neighbors Regression (Part a-k)
    #'knn__weights': ['uniform']

    # Apply distance weighting vs k for k Nearest Neighbors Regression (Part l)
    'knn__weights': ['distance']
}

print("k Nearest Neighbors Regression")
grid = _rmse_grid_search(values_train, hdi_train,
            k_nearest_neighbors_regression_pipeline,
            regression_knn_parameters, 'knn')


# ##### Best value of k: 14
# ##### RMSE at this point: 0.1171925270311745

# ## m)

# In[12]:


# Nearest neighbors to the US. n_neighbors is 8 because the first entry is the US.
scaled_values_train = StandardScaler().fit_transform(values_train)
nbrs = NearestNeighbors(n_neighbors=8).fit(scaled_values_train)
distances, indices = nbrs.kneighbors([pd.DataFrame(data=scaled_values_train).iloc[45]])

print('These are the 7 nearest neighbors of the USA in order. SCALED')
print()
for i in indices:
    print(countries_train[i])


# ## n)

# In[13]:


from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

k_nearest_neighbors_regression_pipeline = Pipeline(
        [
            # Apply PCA to k Nearest Neighbors Regression
            # ('pca', PCA()),

            # Apply scaling to k Nearest Neighbors Regression
            ('scale', StandardScaler()),

            ('knn', KNeighborsRegressor())
        ]
    )

print("k Nearest Neighbors Regression")
grid = _rmse_grid_search(values_train, hdi_train,
            k_nearest_neighbors_regression_pipeline,
            regression_knn_parameters, 'knn')


# ##### Best value of k: 3
# ##### RMSE at this point: 0.11488547357936414

# ## o)
# 
# Scaling the features uniformly does not appear to help. 

# ## p)

# In[14]:


values_train, hdi_train, values_test = import_world_values_data()

k_nearest_neighbors_regression_pipeline = Pipeline(
        [
            # Apply PCA to k Nearest Neighbors Regression
            # ('pca', PCA()),

            # Apply scaling to k Nearest Neighbors Regression
            ('scale', StandardScaler()),

            ('knn', KNeighborsRegressor())
        ]
    )

grid = _rmse_grid_search(values_train, hdi_train,
            k_nearest_neighbors_regression_pipeline,
            regression_knn_parameters, 'knn')

print(grid.predict(values_test))


# ## q)
# 
# 1/K. Because if the data is split perfectly evenly into k classes, your best estimator will have an accuracy of 1/k. In any other case, in which one of the classes has less than a proportion 1/k of the data, we know that there exists another class that has greater than a proportion 1/k of the data, so we know this class would provide a better naive classifier than the smaller proportion. This classifier may not even be the best but it is a lower bound that is higher than 1/k because. This is because we know that if we picked this classifier, we would have an accuracy greater than 1/k because we would get greater than 1/k of the data points classified correctly.

# ## r)

# In[15]:


# Classification Data #
hdi_class_train = hdi_train['2015'].apply(hdi_classification)

#Part d, r: PCA #
plot_pca(values_train, hdi_train, hdi_class_train)


# ## s)
# 
# 
# Badly because using PCA there are not clear decision boundaries between the data, So if you would we would need very soft margins.

# ## t)

# In[16]:


classification_svm_parameters = {
    # Use linear kernel for SVM Classification
    'svm__kernel': ['linear'],

    # Use rbf kernel for SVM Classification
    # 'svm__kernel': ['rbf'],

    # Original hyperparameters
    'svm__C': np.arange(1.0, 100.0, 1.0),

    # Original hyperparameters scaled by 1/100
    # 'svm__C': np.arange(0.01, 1.0, 0.01),

    # Hyperparameter search over all possible dimensions for PCA reduction
    # 'pca__n_components': np.arange(1, 17),

    # 'svm__gamma': np.arange(0.001, 0.1, 0.001)
}

svm_classification_pipeline = Pipeline(
        [
            # Apply PCA to SVM Classification
            #('pca', PCA()),

            # Apply scaling to SVM Classification
            #('scale', StandardScaler()),

            ('svm', SVC())
        ]
    )

_accuracy_grid_search(values_train, hdi_class_train,
                        svm_classification_pipeline,
                        classification_svm_parameters)


# ## u)

# In[17]:


classification_svm_parameters = {
    # Use linear kernel for SVM Classification
    'svm__kernel': ['linear'],

    # Use rbf kernel for SVM Classification
    # 'svm__kernel': ['rbf'],

    # Original hyperparameters
    'svm__C': np.arange(1.0, 100.0, 1.0),

    # Original hyperparameters scaled by 1/100
    # 'svm__C': np.arange(0.01, 1.0, 0.01),

    # Hyperparameter search over all possible dimensions for PCA reduction
    # 'pca__n_components': np.arange(1, 17),

    # 'svm__gamma': np.arange(0.001, 0.1, 0.001)
}

svm_classification_pipeline = Pipeline(
        [
            # Apply PCA to SVM Classification
            ('pca', PCA()),

            # Apply scaling to SVM Classification
            ('scale', StandardScaler()),

            ('svm', SVC())
        ]
    )

_accuracy_grid_search(values_train, hdi_class_train,
                        svm_classification_pipeline,
                        classification_svm_parameters)

print()
print("Nope, it got worse")


# ## v)

# In[18]:


svm_classification_pipeline = Pipeline(
        [
            # Apply PCA to SVM Classification
            #('pca', PCA()),

            # Apply scaling to SVM Classification
            #('scale', StandardScaler()),

            ('svm', SVC())
        ]
    )

classification_svm_parameters = {
    # Use linear kernel for SVM Classification
    #'svm__kernel': ['linear'],

    # Use rbf kernel for SVM Classification
    'svm__kernel': ['rbf'],

    # Original hyperparameters
    'svm__C': np.arange(1.0, 100.0, 1.0),

    # Original hyperparameters scaled by 1/100
    # 'svm__C': np.arange(0.01, 1.0, 0.01),

    # Hyperparameter search over all possible dimensions for PCA reduction
    # 'pca__n_components': np.arange(1, 17),

    # 'svm__gamma': np.arange(0.001, 0.1, 0.001)
}

_accuracy_grid_search(values_train, hdi_class_train,
                        svm_classification_pipeline,
                        classification_svm_parameters)


# ## w)

# In[19]:


k_nearest_neighbors_classification_pipeline = Pipeline(
        [
            # Apply scaling to k Nearest Neighbors Classification
            # ('scale', StandardScaler()),

            ('knn', KNeighborsClassifier())
        ]
    )

_accuracy_grid_search(values_train, hdi_class_train,
                        k_nearest_neighbors_classification_pipeline,
                        classification_knn_parameters)

k_nearest_neighbors_classification_pipeline = Pipeline(
        [
            # Apply scaling to k Nearest Neighbors Classification
            ('scale', StandardScaler()),

            ('knn', KNeighborsClassifier())
        ]
    )

_accuracy_grid_search(values_train, hdi_class_train,
                        k_nearest_neighbors_classification_pipeline,
                        classification_knn_parameters)

print()
print("YES, SCALING HELPS")


# ## x)
# 
# I would predict it to be around 0.72 for our class. 

# ## y)
# 
# This could potentially work quite well, if we used the measured sensor data and used the data in kNN regression to predict the location. 

# ## z)
# 
# We explored kNN regressoin. We looked at its shortcomings that occur when we use too many data points.  This helped me to understand where it is most useful. 
