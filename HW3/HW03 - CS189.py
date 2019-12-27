
# coding: utf-8

# In[167]:


import numpy as np
import matplotlib.pyplot as plt
import pickle


# # HW03
# 
# # Question 2
# 

# # 2.E
# 

# In[55]:


import numpy as np
import matplotlib.pyplot as plt

sample_size = [5,25,125,625]
plt.figure(figsize=[12, 10])   
low_bound = -0.5
high_bound = 0.5
N = 10001
W = np.linspace(40, 60, num=N)
w_true = 50
print(W)

for k in range(4):
    n = sample_size[k]
    
    # generate data
    # np.linspace, np.random.normal and np.random.uniform might be useful functions
    Xs = np.linspace(1, 100, num=n)
    Ys = np.array([x*w_true + np.random.uniform(low_bound, high_bound) for x in Xs])
        

    likelihood = np.ones(N) # likelihood as a function of w

    for i in range(N):
        w_i = W[i]
        in_bound = True
        for j in range(n):
            y_j = Ys[j]
            x_j = Xs[j]

            if w_i > (y_j + 0.5)/x_j or w_i < (y_j - 0.5)/x_j:
                in_bound = False
                break
        if in_bound:
            likelihood[i] = 1
            print("Likelihood is 1 for w equal to " + str(w_i) + " in sample" + str(n))
        else:
            likelihood[i] = 0
        
        # compute likelihood
    #print(sum(likelihood))
    likelihood /= sum(likelihood) # normalize the likelihood
    
    plt.figure()
    # plotting likelihood for different n
    plt.plot(W, likelihood)
    plt.xlabel('w', fontsize=10)
    plt.title(['n=' + str(n)], fontsize=14)

plt.show()


# As n gets large, the amount of w estimates that accurately fit witin our bounds decreases. The MLE of the uniform distribution can be either 1 or zero. As n increases our estimation for the model becomes less general and the variance of our MLE parameter decreases. 

# # 2.I

# In[174]:


import numpy as np
import matplotlib.pyplot as plt

sample_size = [5,25,125]
w0_true = 20
w1_true = 50
N = 10001
w0 = np.linspace(10,30, num=N)
w1 = np.linspace(40, 60, num=N)
Xs = np.random.rand(n,2)
Ys = np.array([x[0]*w0_true + x[1]*w1_true + np.random.normal(0, 1) for x in Xs])

for k in range(4):
    n = sample_size[k]

    # generate data 
    # np.linspace, np.random.normal and np.random.uniform might be useful functions
    
    
    
    
    
    # compute likelihood
    
    N = 1001 
    # W0s = 
    # W1s = 
    likelihood = np.ones([N,N]) # likelihood as a function of w_1 and w_0
                        
    for i1 in range(N):
        # w_1 = W1s[i1]
        for i2 in range(N):
            # w_2 = W2s[i2]
            for i in range(n):
                # compute the likelihood here

    # plotting the likelihood
    plt.figure()                          
    # for 2D likelihood using imshow
    plt.imshow(likelihood, cmap='hot', aspect='auto',extent=[0,4,0,4])
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.show()
    print(n)
   


# # Question 4

# # 4.E

# In[173]:


# assign problem parameters
w0=1
w1=1
N = [10, 100, 1000]
# generate data
# np.random might be useful
alpha = []
z = []
D = 10
for i in range(D):
    alpha.append(np.random.uniform(-1, 1))
    z.append(np.random.normal(0, 1))

y = np.array(alpha)*w1 + w0 + z

err = np.zeros(D)
for d in range(D):
    Xs = np.array([[x**i for i in range(d)] for x in alpha])
    w = np.linalg.lstsq(Xs, y)[0]
    err[d] = np.sum(np.square(Xs.dot(w) - y)) / len(y)
    
    
plt.figure()
# plotting likelihood for different n
plt.plot(D, err)
plt.xlabel('D', fontsize=10)
plt.ylabel('Err', fontsize=10)
plt.title(['n=' + str(n)], fontsize=14)
    
    

# fit data with different models
# np.polyfit and np.polyval might be useful


# plotting figures
# sample code

# plt.figure()
# plt.subplot(121)
# plt.semilogy(np.arange(1, deg+1),error[:,-1])
# plt.xlabel('degree of polynomial')
# plt.ylabel('log of error')
# plt.subplot(122)
# plt.semilogy(np.arange(n_s, n_s+step), error[-1,:])
# plt.xlabel('number of samples')
# plt.ylabel('log of error')
# plt.show()


# # Question 5
# 
# 

# In[75]:


class HW3_Sol(object):


    def __init__(self):
        pass

    def load_data(self):
        self.x_train = pickle.load(open('x_train.p','rb'), encoding='latin1')
        self.y_train = pickle.load(open('y_train.p','rb'), encoding='latin1')
        self.x_test = pickle.load(open('x_test.p','rb'), encoding='latin1')
        self.y_test = pickle.load(open('y_test.p','rb'), encoding='latin1')


hw3_sol = HW3_Sol()
hw3_sol.load_data()

#please visualize the 0th, 10th and 20th images in the training dataset. 
# Also find out what’s their corresponding control vectors.

#Data 
    # Tuples of n values 
    # First index is sample of n 
    # Second index is row of sample 
    # Third index is column of a row 
    # Last index is an entry in RGB pixel
    #Shape of x_train (91, 30, 30, 3)
    


#5.a
def visualize_control_vectors(index):
    sample = hw3_sol.x_train[index]
    y_sample = hw3_sol.y_train[index]
    plt.imshow(sample)
    plt.xlabel(str(index) + "th image. Control Vector: " + str(y_sample))
    plt.show()
    
visualize_control_vectors(0)
visualize_control_vectors(10)
visualize_control_vectors(20)


# In[156]:


#5.b
X = np.vstack([sample.flatten() for sample in hw3_sol.x_train])
U = hw3_sol.y_train

w = np.linalg.lstsq(X, U, rcond=None)
print("PI = ")
print(w[0])





# 5.b
# 
# The weights are on the order of e^(-5) to e^(-7). However, some of the weights are very large and some of the weights are very small in relation to eachother. There is a large difference in weight values due to the relatively large differences in the pixel values of our data. We can amend this through the implementation of standardization and ridge regression. 

# In[98]:


#5.c
def lstsqL(A, b, lambda_):
     return np.linalg.solve((A.T @ A) + np.eye(len(A[0]))*lambda_, A.T @ b)
lambda_array = [0.1, 1.0, 10, 100, 1000]

for l in lambda_array:
    w_hat = lstsqL(X, U, l)
    err = np.average(np.square(X.dot(w_hat) - U)) 
    print(str(err) + "      Traning Error for Lambda: " + str(l))

print()
print("This is different than Ehimares, Confirm")



# In[154]:


#5.d


X_S = (X/255)*2 - 1 # BUT THESE DATA POINTS ARE NOT ALL BETWEEN 0 and 1 ?!
print("Errors for standardized X")
print()
for l in lambda_array:
    w_hat = lstsqL(X_S, U, l)
    err = np.average(np.square(X_S.dot(w_hat) - U)) 
    print(str(err) + "      Traning Error for Lambda: " + str(l))


# In[155]:


#5.e
Xtest = np.vstack([sample.flatten() for sample in hw3_sol.x_test])
Utest = hw3_sol.y_test

for l in lambda_array:
    w_hat = lstsqL(Xtest, Utest, l)
    err = np.average(np.square(Xtest.dot(w_hat) - Utest)) 
    print(str(err) + "      Test Error for Lambda: " + str(l))
print()
print()
X_S_test = (Xtest/255)*2 - 1
for l in lambda_array:
    w_hat = lstsqL(X_S_test, Utest, l)
    err = np.average(np.square(X_S_test.dot(w_hat) - Utest)) 
    print(str(err) + "      Standardized Test Error for Lambda: " + str(l))


# 5.e. 
# 
# 
# Minimum error in test data OLS Ridge Regression with lambda = 100 for both standardized and unstandardized test data. 
# 
# Lambda affect on performance in terms of bias:
# 
# As lambda increases our bias increases. 
# 
# Lambda affect on performance in terms of variance:
# 
# As lambda increases our variance decreases. 
# 

# In[157]:


#5.f

#With training data


without = np.linalg.svd((X.T @ X) + 100*np.eye(len(X[0])), compute_uv = False)
k = np.max(without) / np.min(without)
print(k)

with_ = np.linalg.svd((X_S.T @ X_S) + 100*np.eye(len(X_S[0])), compute_uv = False)
k = np.max(with_) / np.min(with_)
print(k)


