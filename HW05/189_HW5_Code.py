
# coding: utf-8

# # Question 2 E 

# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread,imsave

imFile = 'stpeters_probe_small.png'
compositeFile = 'tennis.png'
targetFile = 'interior.jpg'

# This loads and returns all of the images needed for the problem
# data - the image of the spherical mirror
# tennis - the image of the tennis ball that we will relight
# target - the image that we will paste the tennis ball onto
def loadImages():
    imFile = 'stpeters_probe_small.png'
    compositeFile = 'tennis.png'
    targetFile = 'interior.jpg'
    
    data = imread(imFile).astype('float')*1.5
    tennis = imread(compositeFile).astype('float')
    target = imread(targetFile).astype('float')/255

    return data, tennis, target
    

# This function takes as input a square image of size m x m x c
# where c is the number of color channels in the image.  We
# assume that the image contains a scphere and that the edges
# of the sphere touch the edge of the image.
# The output is a tuple (ns, vs) where ns is an n x 3 matrix
# where each row is a unit vector of the direction of incoming light
# vs is an n x c vector where the ith row corresponds with the
# image intensity of incoming light from the corresponding row in ns
def extractNormals(img):

    # Assumes the image is square
    d = img.shape[0]
    r = d / 2
    ns = []
    vs = []
    for i in range(d):
        for j in range(d):

            # Determine if the pixel is on the sphere
            x = j - r
            y = i - r
            if x*x + y*y > r*r-100:
                continue

            # Figure out the normal vector at the point
            # We assume that the image is an orthographic projection
            z = np.sqrt(r*r-x*x-y*y)
            n = np.asarray([x,y,z])
            n = n / np.sqrt(np.sum(np.square(n)))
            view = np.asarray([0,0,-1])
            n = 2*n*(np.sum(n*view))-view
            ns.append(n)
            vs.append(img[i,j])

    return np.asarray(ns), np.asarray(vs)

# This function renders a diffuse sphere of radius r
# using the spherical harmonic coefficients given in
# the input coeff where coeff is a 9 x c matrix
# with c being the number of color channels
# The output is an 2r x 2r x c image of a diffuse sphere
# and the value of -1 on the image where there is no sphere
def renderSphere(r,coeff):

    d = 2*r
    img = -np.ones((d,d,3))
    ns = []
    ps = []

    for i in range(d):
        for j in range(d):

            # Determine if the pixel is on the sphere
            x = j - r
            y = i - r
            if x*x + y*y > r*r:
                continue

            # Figure out the normal vector at the point
            # We assume that the image is an orthographic projection
            z = np.sqrt(r*r-x*x-y*y)
            n = np.asarray([x,y,z])
            n = n / np.sqrt(np.sum(np.square(n)))
            ns.append(n)
            ps.append((i,j))

    ns = np.asarray(ns)
    B = computeBasis(ns)
    vs = B.dot(coeff)

    for p,v in zip(ps,vs):
        img[p[0],p[1]] = np.clip(v,0,255)

    return img

# relights the sphere in img, which is assumed to be a square image
# coeff is the matrix of spherical harmonic coefficients
def relightSphere(img, coeff):
    img = renderSphere(int(img.shape[0]/2),coeff)/255*img/255
    return img

# Copies the image of source onto target
# pixels with values of -1 in source will not be copied
def compositeImages(source, target):
    
    # Assumes that all pixels not equal to 0 should be copied
    out = target.copy()
    cx = int(target.shape[1]/2)
    cy = int(target.shape[0]/2)
    sx = cx - int(source.shape[1]/2)
    sy = cy - int(source.shape[0]/2)

    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            if np.sum(source[i,j]) >= 0:
                out[sy+i,sx+j] = source[i,j]

    return out

# Fill in this function to compute the basis functions
# This function is used in renderSphere()
def computeBasis(ns):
    # Returns the first 9 spherical harmonic basis functions

    #################################################
    B = np.ones((len(ns), 9))
    for i in range(len(ns)):
        x,y,z = ns[i] / np.linalg.norm(ns[i])
        B[i] = np.array([1,
                         y,
                         x,
                         z,
                         x*y,
                         y*z,
                         3*z**2 - 1,
                         x*z,
                         x**2 - y**2])
    #################################################
    return B
    
#if __name__ == '__main__':

data,tennis,target = loadImages()
ns, vs = extractNormals(data)
B = computeBasis(ns)


# reduce the number of samples because computing the SVD on
# the entire data set takes too long
Bp = B[::50]
vsp = vs[::50]

#################################################
# TODO: Solve for the coefficients using least squares
# or total least squares here
print("Starting Least Squares")
w = np.linalg.lstsq(Bp, vsp)

#Total Least Sqaures
# w = np.solve(np.transpose(Bp) @ Bp - sigma2 @ np.ones(len(Bp[0])) = np.transpose(Bp) @ svp )


##################################################
coeff = w[0]

#coeff[0,:] = 255

img = relightSphere(tennis,coeff)

output = compositeImages(img,target)

print('Coefficients:\n'+str(coeff))

plt.figure(1)
plt.imshow(output)
plt.show()

imsave('output.png',output)


# # Question 2 F 

# In[3]:


# TODO: Solve for the coefficients using least squares
# or total least squares here
print("Starting Total Least Squares")

#Total Least Squares
S = np.linalg.svd(np.hstack((Bp, vsp)), compute_uv = False)[-1]**2
w = np.linalg.inv(np.transpose(Bp).dot(Bp) -  S*np.eye(9)).dot(np.transpose(Bp).dot(vsp))

##################################################
coeff = w

#coeff[0,:] = 255

img = relightSphere(tennis,coeff)

output = compositeImages(img,target)

print('Coefficients:\n'+str(coeff))

plt.figure(1)
plt.imshow(output)
plt.show()

imsave('output.png',output)


# # Question 2 G 

# In[9]:


# TODO: Solve for the coefficients using least squares
# or total least squares here
print("Starting Total Least Squares")

#Total Least Squares
vsp_adj = vsp/np.max(vsp)
S = np.linalg.svd(np.hstack((Bp, vsp_adj)), compute_uv = False)[-1]**2
w = np.linalg.inv(np.transpose(Bp).dot(Bp) -  S*np.eye(9)).dot(np.transpose(Bp).dot(vsp_adj))

##################################################
coeff = w

#coeff[0,:] = 255

img = relightSphere(tennis,coeff) * np.max(vsp)

output = compositeImages(img,target)

print('Coefficients:\n'+str(coeff))

plt.figure(1)
plt.imshow(output)
plt.show()

imsave('output.png',output)


# # Question 3 H

# In[73]:


import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import sklearn.linear_model
from sklearn.model_selection import train_test_split


######## PROJECTION FUNCTIONS ##########

## Random Projections ##
def random_matrix(d, k):
    '''
    d = original dimension
    k = projected dimension
    '''
    return 1./np.sqrt(k)*np.random.normal(0, 1, (d, k))

def random_proj(X, k):
    _, d= X.shape
    return X.dot(random_matrix(d, k))

## PCA and projections ##
def my_pca(X, k):
    '''
    compute PCA components
    X = data matrix (each row as a sample)
    k = #principal components
    '''
    n, d = X.shape
    assert(d>=k)
    _, _, Vh = np.linalg.svd(X)    
    V = Vh.T
    return V[:, :k]

def pca_proj(X, k):
    
    '''
    compute projection of matrix X
    along its first k principal components
    '''
    P = my_pca(X, k)
     # P = P.dot(P.T)
    return X.dot(P)
    
    



######### LINEAR MODEL FITTING ############

def rand_proj_accuracy_split(X, y, k):
    '''
    Fitting a k dimensional feature set obtained
    from random projection of X, versus y
    for binary classification for y in {-1, 1}
    '''
    
    # test train split
    _, d = X.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # random projection
    J = np.random.normal(0., 1., (d, k))
    rand_proj_X = X_train.dot(J)
    
    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(rand_proj_X, y_train)
    
    # predict y
    y_pred=line.predict(X_test.dot(J))
    
    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)

def pca_proj_accuracy(X, y, k):
    '''
    Fitting a k dimensional feature set obtained
    from PCA projection of X, versus y
    for binary classification for y in {-1, 1}
    '''

    # test-train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # pca projection
    P = my_pca(X_train, k)
    P = P.dot(P.T)
    pca_proj_X = X_train.dot(P)
                
    # fit a linear model
    line = sklearn.linear_model.LinearRegression(fit_intercept=False)
    line.fit(pca_proj_X, y_train)
    
     # predict y
    y_pred=line.predict(X_test.dot(P))
    

    # return the test error
    return 1-np.mean(np.sign(y_pred)!= y_test)


######## LOADING THE DATASETS #########

# to load the data:
# data = np.load('data/data1.npz')
# X = data['X']
# y = data['y']
# n, d = X.shape

data1 = np.load('data1.npz')
data2 = np.load('data2.npz')
data3 = np.load('data3.npz')

X1 = data1['X']
y1 = data1['y']
n1, d1 = X1.shape

X2 = data2['X']
y2 = data2['y']
n2, d2 = X2.shape

X3 = data3['X']
y3 = data3['y']
n3, d3 = X3.shape

# n_trials = 10  # to average for accuracies over random projections

######### YOUR CODE GOES HERE ##########

# Using PCA and Random Projection for:
# Visualizing the datasets 

#k = 2
def visualize(X, k):
    pca_projection =pca_proj(X, k)
    x = np.zeros(len(pca_projection))
    y = np.zeros(len(pca_projection))
    for i in range(len(pca_projection)):
        x[i] = pca_projection[i][0]
        y[i] = pca_projection[i][0]
    return x, y    
        
def visualize_random(X, k):
    pca_rand_proj = random_proj(X, k)
    x = np.zeros(len(pca_rand_proj))
    y = np.zeros(len(pca_rand_proj))
    for i in range(len(pca_rand_proj)):
        x[i] = pca_rand_proj[i][0]
        y[i] = pca_rand_proj[i][0]
    return x, y 
    
    
v1_x, v1_y = visualize(X1, 2)
v1r_x, v1r_y = visualize_random(X1, 2)

v2_x, v2_y = visualize(X2, 2)
v2r_x, v2r_y = visualize_random(X2, 2)

v3_x, v3_y = visualize(X3, 2)
v3r_x, v3r_y = visualize_random(X3, 2)


# Computing the accuracies over different datasets.


def compute_accuracy(X, y, k):
    accuracy_pca = pca_proj_accuracy(X, y, k)  
    r = np.zeros(10)
    for i in range(10):
        r[i] = rand_proj_accuracy_split(X, y, k)
    accuracy_rand_pca = np.average(r)
      
    return accuracy_pca, accuracy_rand_pca
      
def plot_accuracy(X, y):
    d = X.shape[1]
    A = np.zeros(d)
    A_rand = np.zeros(d)
    for i in range(d):
        accuracies = compute_accuracy(X, y, i+1)
        A[i] = accuracies[0]
        A_rand[i] = accuracies[1]
    #print(A)
    plt.plot(range(1, d+1), A)
    plt.plot(range(1, d+1), A_rand, color='red')
    plt.show()
    return





# Don't forget to average the accuracy for multiple
# random projections to get a smooth curve.





# And computing the SVD of the feature matrix

def compute_plot_svd(X):
    svd = np.linalg.svd(X, compute_uv = False)
    print("Singular Value Plot")
    plt.bar(range(1, len(svd)+1), svd)
    plt.show()
######## YOU CAN PLOT THE RESULTS HERE ########

# plt.plot, plt.scatter would be useful for plotting


#PLOT PART H
def plot_partH():
    plt.scatter(v1_x, v1_y)
    print("Data1 Projection")
    plt.show()
    plt.scatter(v1r_x, v1r_y)
    print("Data1 Random Projection")
    plt.show()

    plt.scatter(v2_x, v2_y, color='red')
    print("Data2 Projection")
    plt.show()
    plt.scatter(v2r_x, v2r_y, color = 'red')
    print("Data2 Random Projection")
    plt.show()

    plt.scatter(v3_x, v3_y, color = 'orange')
    print("Data3 Projection")
    plt.show()
    plt.scatter(v3r_x, v3r_y, color = 'orange')
    print("Data3 Random Projection")
    plt.show()
    return

#PLOT PART I 
def plot_partI():
    
    print("Red is for Random_Projection Accuracy Errors")
    print("Blue is for PCA Accuracy Errors ")
    plot_accuracy(X1, y1) 
    plot_accuracy(X2, y2)
    plot_accuracy(X3, y3)
    return

def plot_partJ():
    print('X1')
    compute_plot_svd(X1)
    print('X2')
    compute_plot_svd(X2)
    print('X3')
    compute_plot_svd(X3)

plot_partH()
plot_partI()
plot_partJ()


# ## 3 H Description
# 
# The PCA and random projections have very similar results.Dont observe a difference. There is a strong positive linear trend in all three data sets. 

# # 3 I Comment
# 
# As we increase the number of principal components the accuracy of our two methods of projections converge. 

# # 3 J Observations & Explanation
# 
# The size of the first singular values, and thus the first principal component, dominates in each of the data sets. The other singular values are smaller and similar in magnitude.
# 
# For regular the greatest reduction in error increases after the selectin of the first principal component and does not improve much thereafter. Knowing this it makes sense that the random takes longer to reduce errors. 
