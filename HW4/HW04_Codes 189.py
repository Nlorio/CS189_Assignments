
# coding: utf-8

# # Question 2

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
np.random.seed(0)

mu = [15, 5]
sigma = [[20, 0], [0, 10]]

samples = np.random.multivariate_normal(mu, sigma, size=100)
plt.scatter(samples[:, 0], samples[:, 1])
plt.show()
mean = np.average(samples, axis = 0)
covariance = np.sum(((sample - mean).reshape(2,1)).dot((np.transpose(sample - mean)).reshape(1,2)) for sample in samples)/len(samples)

    
print("Mean_hat: " + str(mean))
print("Covariance_hat: " + str(covariance))
#print((np.transpose(sample - mean)).dot(sample - mean) for sample in samples)


# In[55]:


sigma = [[20, 14], [14, 10]]
samples = np.random.multivariate_normal(mu, sigma, size=100)
plt.scatter(samples[:, 0], samples[:, 1])
plt.show()
mean = np.average(samples, axis = 0)
covariance = np.sum(((sample - mean).reshape(2,1)).dot((np.transpose(sample - mean)).reshape(1,2)) for sample in samples)/len(samples)
print("Mean_hat: " + str(mean))
print("Covariance_hat: " + str(covariance))


# In[54]:


sigma = [[20, -14], [-14, 10]]
samples = np.random.multivariate_normal(mu, sigma, size=100)
plt.scatter(samples[:, 0], samples[:, 1])
plt.show()
mean = np.average(samples, axis = 0)
covariance = np.sum(((sample - mean).reshape(2,1)).dot((np.transpose(sample - mean)).reshape(1,2)) for sample in samples)/len(samples)
print("Mean_hat: " + str(mean))
print("Covariance_hat: " + str(covariance))


# In[12]:


def generate_data(n):
    """
    This function generates data of size n.
    """
    
    X = np.random.normal(0, 5, (n, 2))
    z = np.random.normal(0, 1, (n, 1))
    
    y = np.sum(X, axis = 1).reshape(n, 1) + z
        
    return (X,y)

def tikhonov_regression(X,Y,Sigma):
    """
    This function computes w based on the formula of tikhonov_regression.
    """
    w = np.linalg.inv(X.T.dot(X) + np.linalg.inv(Sigma)).dot(X.T).dot(Y)
    return w

def compute_mean_var(X,y,Sigma):
    """
    This function computes the mean and variance of the posterior
    """
    mux, muy = tikhonov_regression(X, y, Sigma)
    m = np.linalg.inv(X.T.dot(X) + np.linalg.inv(Sigma))
    sigmax, sigmay, sigmaxy = np.sqrt(m[0][0]), np.sqrt(m[1][1]), m[0][1]
    
    return mux,muy,sigmax,sigmay,sigmaxy

Sigmas = [np.array([[1,0],[0,1]]), np.array([[1,0.25],[0.25,1]]),
          np.array([[1,0.9],[0.9,1]]), np.array([[1,-0.25],[-0.25,1]]),
          np.array([[1,-0.9],[-0.9,1]]), np.array([[0.1,0],[0,0.1]])]
names = [str(i) for i in range(1,6+1)]

for num_data in [5,50,500]:
    X,Y = generate_data(num_data)
    for i,Sigma in enumerate(Sigmas):

        mux,muy,sigmax,sigmay,sigmaxy = compute_mean_var(X, Y, Sigma)

        x = np.arange(0.5, 1.5, 0.01)
        y = np.arange(0.5, 1.5, 0.01)
        X_grid, Y_grid = np.meshgrid(x, y)

        Z = matplotlib.mlab.bivariate_normal(X_grid,Y_grid, sigmax, sigmay, mux, muy, sigmaxy)

        # plot
        plt.figure(figsize=(10,10))
        CS = plt.contour(X_grid, Y_grid, Z,
                         levels = np.concatenate([np.arange(0,0.05,0.01),np.arange(0.05,1,0.05)]))
        plt.clabel(CS, inline=1, fontsize=10)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Sigma'+ names[i] + ' with num_data = {}'.format(num_data))
        plt.savefig('Sigma'+ names[i] + '_num_data_{}.png'.format(num_data))


# As the number of data points increases the covariance decreases. This can be observed from our outputs for each of the sigmas 1 through 6. 
# 
# 
# # 3. E

# In[41]:


np.random.seed(0)
w = [1.0,1.0]
n_test = 500
n_trains = np.arange(5,205,5)
n_trails = 100

Sigmas = [np.array([[1,0],[0,1]]), np.array([[1,0.25],[0.25,1]]),
          np.array([[1,0.9],[0.9,1]]), np.array([[1,-0.25],[-0.25,1]]),
          np.array([[1,-0.9],[-0.9,1]]), np.array([[0.1,0],[0,0.1]])]
names = ['Sigma{}'.format(i+1) for i in range(6)]

def compute_mse(X,Y, w): # Empirical 
    """
    This function computes MSE given data and estimated w.
    """
   
    mse = np.sum(np.square(Y - X.dot(w)))/len(Y)
    return mse

def compute_theoretical_mse(w):
    """
    This function computes theoretical MSE given estimated w.
    """
    #TODO implement this
    theoretical_mse = 5*(w[0] - 1)**2 + 5*(w[1] - 1)**2 + 1
    return theoretical_mse

# Generate Test Data.
X_test, y_test = generate_data(n_test)

mses = np.zeros((len(Sigmas), len(n_trains), n_trails))

theoretical_mses = np.zeros((len(Sigmas), len(n_trains), n_trails))

for seed in range(n_trails):
    np.random.seed(seed)
    for i,Sigma in enumerate(Sigmas):
        for j,n_train in enumerate(n_trains):
            #TODO implement the mses and theoretical_mses
            x_train, y_train = generate_data(n_train)
            w = tikhonov_regression(x_train, y_train, Sigma)
            empirical_mse = compute_mse(X_test, y_test, w)
            theoretical_mse = compute_theoretical_mse(w)
            mses[i, j, seed] = empirical_mse
            theoretical_mses[i, j, seed] = theoretical_mse
            
            

# Plot
plt.figure()
for i,_ in enumerate(Sigmas):
    plt.plot(n_trains, np.mean(mses[i],axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('MSE on Test Data')
plt.legend()
plt.savefig('MSE.png')

plt.figure()
for i,_ in enumerate(Sigmas):
    plt.plot(n_trains, np.mean(theoretical_mses[i],axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('Theoretical MSE on Test Data')
plt.legend()
plt.savefig('theoretical_MSE.png')


plt.figure()
for i,_ in enumerate(Sigmas):
    plt.loglog(n_trains, np.mean(theoretical_mses[i]-1,axis = -1),label = names[i])
plt.xlabel('Number of data')
plt.ylabel('Log Theoretical MSE on Test Data')
plt.legend()
plt.savefig('log_theoretical_MSE.png')


# As the amount of training data increases our MSE, both theoretical and empirical, approach the same value. The goodness of our priors appears to be come less important as the amount of training data increases. 

# # 5 Kernel Ridge Regression

# In[98]:



import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import comb
import itertools

LAMBDA = 0.001

# choose the data you want to load
data = np.load('circle.npz')
# data = np.load('heart.npz')
# data = np.load('asymmetric.npz')

SPLIT = 0.8
X = data["x"]
y = data["y"]
X /= np.max(X)  # normalize the data

n_train = int(X.shape[0] * SPLIT)
X_train = X[:n_train:, :]
X_valid = X[n_train:, :]
y_train = y[:n_train]
y_valid = y[n_train:]

LAMBDA = 0.001


def lstsq(A, b, lambda_=0):
    return np.linalg.solve(A.T @ A + lambda_ * np.eye(A.shape[1]), A.T @ b)


def heatmap(f, X, y, clip=5):
 # example: heatmap(lambda x, y: x * x + y * y)
 # clip: clip the function range to [-clip, clip] to generate a clean plot
 # set it to zero to disable this function
    xx = yy = np.linspace(np.min(X), np.max(X), 72)
    x0, y0 = np.meshgrid(xx, yy)
    x0, y0 = x0.ravel(), y0.ravel()
    z0 = f(x0, y0)
    
    if clip:
        z0[z0 > clip] = clip
        z0[z0 < -clip] = -clip
        
    plt.hexbin(x0, y0, C=z0, gridsize=50, cmap=cm.jet, bins=None)
    plt.colorbar()
    cs = plt.contour(xx, yy, z0.reshape(xx.size, yy.size), [-2, -1, -0.5, 0, 0.5, 1,2], cmap=cm.jet)
    plt.clabel(cs, inline=1, fontsize=10)
    
    pos = y[:] == +1.0
    neg = y[:] == -1.0
    plt.scatter(X[pos, 0], X[pos, 1], c='red', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='blue', marker='v')
    plt.show()




# In[92]:


#5.A

# choose the data you want to load
data = np.load('circle.npz')
SPLIT = 0.8
Xc = data["x"]
yc = data["y"]
Xc /= np.max(Xc) # normalize the data

n_train_c = int(Xc.shape[0] * SPLIT)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xc[:,0], Xc[:,1], yc, c=yc)


# In[93]:


data = np.load('heart.npz')
SPLIT = 0.8
Xh = data["x"]
yh = data["y"]
Xh /= np.max(Xh) # normalize the data
n_train_h = int(Xh.shape[0] * SPLIT)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xh[:,0], Xh[:,1], yh, c=yh)


# In[94]:


data = np.load('asymmetric.npz')
SPLIT = 0.8
Xa = data["x"]
ya = data["y"]
Xa /= np.max(Xa) # normalize the data
n_train_a = int(Xa.shape[0] * SPLIT)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xa[:,0], Xa[:,1], ya, c=ya)


# # 5.b

# In[95]:


def partitions(n, b):
    masks = np.identity(b, dtype=int)
    for c in itertools.combinations_with_replacement(masks, n):
        yield sum(c)
        
def compute_multivariates(features, degree):
    if degree == 0:
        return [1]
    powers = partitions(degree, len(features))
    results = []
    for deg in powers:
        total = 1
        for i in range(len(deg)):
            total = total*(features[i]**deg[i])
        results.append(total)
    results.extend(compute_multivariates(features, degree-1))
    return np.array(results)

def lstsqL(A, b, lambda_):
    return np.linalg.solve((A.T @ A) + np.eye(len(A[0]))*lambda_, A.T @ b)

l = 0.001


all_multivariates_Xc = np.array([compute_multivariates(x, 16) for x in Xc])
all_multivariates_Xh = np.array([compute_multivariates(x, 16) for x in Xh])
all_multivariates_Xa = np.array([compute_multivariates(x, 16) for x in Xa]
                               )
Xc_train_mult = all_multivariates_Xc[:n_train_c:, :]
Xc_valid_mult = all_multivariates_Xc[n_train_c:, :]

Xh_train_mult = all_multivariates_Xh[:n_train_h:, :]
Xh_valid_mult = all_multivariates_Xh[n_train_h:, :]

Xa_train_mult = all_multivariates_Xa[:n_train_a:, :]
Xa_valid_mult = all_multivariates_Xa[n_train_a:, :]

yc_train = yc[:n_train_c]
yc_valid = yc[n_train_c:]

yh_train = yh[:n_train_h]
yh_valid = yh[n_train_h:]

ya_train = ya[:n_train_a]
ya_valid = ya[n_train_a:]


Xc_train_reg = Xc[:n_train_c:, :]
Xc_valid_reg = Xc[n_train_c:, :]

Xh_train_reg = Xh[:n_train_h:, :]
Xh_valid_reg = Xh[n_train_h:, :]

Xa_train_reg = Xa[:n_train_a:, :]
Xa_valid_reg = Xa[n_train_a:, :]






# In[99]:


def f(x, y, p):
    data = np.array([x, y]).T
    return np.array([compute_multivariates(x_, p) for x_ in data])

for p in range(1, 17):

    start_of_features = comb(2 + p, 2, exact=True)

    Xc_train = Xc_train_mult[:, -start_of_features:]
    Xc_valid = Xc_valid_mult[:, -start_of_features:]

    w = lstsqL(Xc_train, yc_train, l)

    err_train = np.sum(np.square((yc_train - Xc_train.dot(w)))) / len(yc_train)
    err_test = np.sum(np.square((yc_valid - Xc_valid.dot(w)))) / len(yc_valid)

    print('p:', p, ', err_train:', err_train, ', err_test:', err_test)

    if p%2 == 0:

        heatmap(lambda x, y: f(x, y, p).dot(w), Xc, yc)


# In[100]:


for p in range(1, 17):

    start_of_features = comb(2 + p, 2, exact=True)

    Xh_train = Xh_train_mult[:, -start_of_features:]
    Xh_valid = Xh_valid_mult[:, -start_of_features:]

    w = lstsqL(Xh_train, yh_train, l)

    err_train = np.sum(np.square((yh_train - Xh_train.dot(w)))) / len(yh_train)
    err_test = np.sum(np.square((yh_valid - Xh_valid.dot(w)))) / len(yh_valid)

    print('p:', p, ', err_train:', err_train, ', err_test:', err_test)

    if p%2 == 0:

        heatmap(lambda x, y: f(x, y, p).dot(w), Xh, yh)


# In[101]:


for p in range(1, 17):

    start_of_features = comb(2 + p, 2, exact=True)

    Xa_train = Xa_train_mult[:, -start_of_features:]
    Xa_valid = Xa_valid_mult[:, -start_of_features:]

    w = lstsqL(Xa_train, ya_train, l)

    err_train = np.sum(np.square((ya_train - Xa_train.dot(w)))) / len(ya_train)
    err_test = np.sum(np.square((ya_valid - Xa_valid.dot(w)))) / len(ya_valid)

    print('p:', p, ', err_train:', err_train, ', err_test:', err_test)

    if p%2 == 0:

        heatmap(lambda x, y: f(x, y, p).dot(w), Xa, ya)


# # 5.C

# In[102]:


def kernelRidge_c(A, b, lambda_, d):
    K = np.power(Xc_train_reg @ Xc_train_reg.T + 1, d)
    return A.T @ (np.linalg.inv(K+lambda_*np.eye(len(K[0])))) @ (b)


# In[107]:


for p in range(1, 17):

    start_of_features = comb(2 + p, 2, exact=True)

    Xc_train = Xc_train_mult[:, -start_of_features:]
    Xc_valid = Xc_valid_mult[:, -start_of_features:]

    w = kernelRidge_c(Xc_train, yc_train, l, p)

    err_train = np.sum(np.square((yc_train - Xc_train.dot(w)))) / len(yc_train)
    err_test = np.sum(np.square((yc_valid - Xc_valid.dot(w)))) / len(yc_valid)

    print('p:', p, ', err_train:', err_train, ', err_test:', err_test)

    if p%2 == 0:

        heatmap(lambda x, y: f(x, y, p).dot(w), Xc, yc)


# In[111]:


def kernelRidge_h(A, b, lambda_, d):
    k = np.power(Xh_train_reg @ Xh_train_reg.T + 1, d)
    return A.T @ (np.linalg.inv(k+lambda_*np.eye(len(k[0])))) @ (b)

for p in range(1, 17):

    start_of_features = comb(2 + p, 2, exact=True)

    Xh_train = Xh_train_mult[:, -start_of_features:]
    Xh_valid = Xh_valid_mult[:, -start_of_features:]

    w = kernelRidge_h(Xh_train, yh_train, l, p)

    err_train = np.sum(np.square((yh_train - Xh_train.dot(w)))) / len(yh_train)
    err_test = np.sum(np.square((yh_valid - Xh_valid.dot(w)))) / len(yh_valid)

    print('p:', p, ', err_train:', err_train, ', err_test:', err_test)

    if p%2 == 0:

        heatmap(lambda x, y: f(x, y, p).dot(w), Xh, yh)


# In[112]:


def kernelRidge_a(A, b, lambda_, d):
    k = np.power(Xa_train_reg @ Xa_train_reg.T + 1, d)
    return A.T @ (np.linalg.inv(k+lambda_*np.eye(len(k[0])))) @ (b)

for p in range(1, 17):

    start_of_features = comb(2 + p, 2, exact=True)

    Xa_train = Xa_train_mult[:, -start_of_features:]
    Xa_valid = Xa_valid_mult[:, -start_of_features:]

    w = kernelRidge_a(Xa_train, ya_train, l, p)

    err_train = np.sum(np.square((ya_train - Xa_train.dot(w)))) / len(ya_train)
    err_test = np.sum(np.square((ya_valid - Xa_valid.dot(w)))) / len(ya_valid)

    print('p:', p, ', err_train:', err_train, ', err_test:', err_test)

    if p%2 == 0:

        heatmap(lambda x, y: f(x, y, p).dot(w), Xa, ya)


# In[115]:


n_train_c_new = int(Xc.shape[0] * 0.85)
Xc_train_reg_n = Xc[:n_train_c_new:, :]
Xc_valid_reg_n = Xc[n_train_c_new:, :]

def kernelRidge_c1(A, b, lambda_, d):
    K = np.power(Xc_train_reg_n @ Xc_train_reg_n.T + 1, d)
    return A.T @ (np.linalg.inv(K+lambda_*np.eye(len(K[0])))) @ (b)


# In[116]:


Xc_train_mult_1 = all_multivariates_Xc[:n_train_c_new:, :]
Xc_valid_mult_1 = all_multivariates_Xc[n_train_c_new:, :]

yc_train_1 = yc[:n_train_c_new]
yc_valid_1 = yc[n_train_c_new:]
for p in range(1, 24):

    start_of_features = comb(2 + p, 2, exact=True)

    Xc_train = Xc_train_mult_1[:, -start_of_features:]
    Xc_valid = Xc_valid_mult_1[:, -start_of_features:]

    w = kernelRidge_c1(Xc_train, yc_train_1, l, p)

    err_train = np.sum(np.square((yc_train_1 - Xc_train.dot(w)))) / len(yc_train)
    err_test = np.sum(np.square((yc_valid_1 - Xc_valid.dot(w)))) / len(yc_valid)

    print('p:', p, ', err_train:', err_train, ', err_test:', err_test)


# In[120]:


Xa_train_mult = all_multivariates_Xa[:n_train_a:, :]
Xa_valid_mult = all_multivariates_Xa[n_train_a:, :]
ya_train = ya[:n_train_a]
ya_valid = ya[n_train_a:]

n_trains = np.arange(10, n_train_a, 1000)
for p in (5, 6):

    start_of_features = comb(2 + p, 2, exact=True)
    Xa_valid = Xa_valid_mult[:, -start_of_features:]

    for l in (0.0001,0.001,0.01):
        err_valid = []

        for n_train in n_trains:

            # 100 trials
            err_test = 0
            for seed in range(100):

                np.random.seed(seed)
                samples = np.random.randint(0, n_train_a, n_train)
                Xa_train = Xa_train_mult[samples, -start_of_features:]

                #err_test = 0
                w = lstsqL(Xa_train, ya_train[samples], l)

                err_test += np.sum(np.square((ya_valid - Xa_valid.dot(w)))) / len(ya_valid)


                err_valid.append(err_test/100)

    plt.title("lambda = %f, p = %d" % (l, p))
    plt.plot(n_trains, err_valid)
    plt.show()


# # 5. E
# 
# # 5. F
# 
# Polynomial ridge regression with the implementation of the kernel trick is more efficient. If the amount of training data we have is significantly less than the hnumber of features we have than we would opt to implement kernel. Otherwise, the non-kernalized version of polynomial ridge regression is more efficient. 
# 
# 
# # 5. G
