
# coding: utf-8

# ## Question 4

# In[17]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial


# Gradient descent optimization
# The learning rate is specified by eta
class GDOptimizer(object):
    def __init__(self, eta):
        self.eta = eta

    def initialize(self, layers):
        pass

    # This function performs one gradient descent step
    # layers is a list of dense layers in the network
    # g is a list of gradients going into each layer before the nonlinear activation
    # a is a list of of the activations of each node in the previous layer going
    def update(self, layers, g, a):
        m = a[0].shape[1]
        for layer, curGrad, curA in zip(layers, g, a):
            layer.updateWeights(-self.eta / m * np.dot(curGrad, curA.T))
            layer.updateBias(-self.eta / m * np.sum(curGrad, 1).reshape(layer.b.shape))


# Cost function used to compute prediction errors
class QuadraticCost(object):

    # Compute the squared error between the prediction yp and the observation y
    # This method should compute the cost per element such that the output is the
    # same shape as y and yp
    @staticmethod
    def fx(y, yp):
        return 0.5 * np.square(yp - y)

    # Derivative of the cost function with respect to yp
    @staticmethod
    def dx(y, yp):
        return y - yp


# Sigmoid function fully implemented as an example
class SigmoidActivation(object):
    @staticmethod
    def fx(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def dx(z):
        return SigmoidActivation.fx(z) * (1 - SigmoidActivation.fx(z))


# Hyperbolic tangent function
class TanhActivation(object):

    # Compute tanh for each element in the input z
    @staticmethod
    def fx(z):
        return np.tanh(z)

    # Compute the derivative of the tanh function with respect to z
    @staticmethod
    def dx(z):
        return 1 - np.square(np.tanh(z))


# Rectified linear unit
class ReLUActivation(object):
    @staticmethod
    def fx(z):
        return np.maximum(0, z)

    @staticmethod
    def dx(z):
        return (z > 0).astype('float')


# Linear activation
class LinearActivation(object):
    @staticmethod
    def fx(z):
        return z

    @staticmethod
    def dx(z):
        return np.ones(z.shape)


# This class represents a single hidden or output layer in the neural network
class DenseLayer(object):

    # numNodes: number of hidden units in the layer
    # activation: the activation function to use in this layer
    def __init__(self, numNodes, activation):
        self.numNodes = numNodes
        self.activation = activation

    def getNumNodes(self):
        return self.numNodes

    # Initialize the weight matrix of this layer based on the size of the matrix W
    def initialize(self, fanIn, scale=1.0):
        s = scale * np.sqrt(6.0 / (self.numNodes + fanIn))
        self.W = np.random.normal(0, s, (self.numNodes, fanIn))
        self.b = np.random.uniform(-1, 1, (self.numNodes, 1))

    # Apply the activation function of the layer on the input z
    def a(self, z):
        return self.activation.fx(z)

    # Compute the linear part of the layer
    # The input a is an n x k matrix where n is the number of samples
    # and k is the dimension of the previous layer (or the input to the network)
    def z(self, a):
        return self.W.dot(a) + self.b  # Note, this is implemented where we assume a is k x n

    # Compute the derivative of the layer's activation function with respect to z
    # where z is the output of the above function.
    # This derivative does not contain the derivative of the matrix multiplication
    # in the layer.  That part is computed below in the model class.
    def dx(self, z):
        return self.activation.dx(z)

    # Update the weights of the layer by adding dW to the weights
    def updateWeights(self, dW):
        self.W = self.W + dW

    # Update the bias of the layer by adding db to the bias
    def updateBias(self, db):
        self.b = self.b + db


# This class handles stacking layers together to form the completed neural network
class Model(object):

    # inputSize: the dimension of the inputs that go into the network
    def __init__(self, inputSize):
        self.layers = []
        self.inputSize = inputSize

    # Add a layer to the end of the network
    def addLayer(self, layer):
        self.layers.append(layer)

    # Get the output size of the layer at the given index
    def getLayerSize(self, index):
        if index >= len(self.layers):
            return self.layers[-1].getNumNodes()
        elif index < 0:
            return self.inputSize
        else:
            return self.layers[index].getNumNodes()

    # Initialize the weights of all of the layers in the network and set the cost
    # function to use for optimization
    def initialize(self, cost, initializeLayers=True):
        self.cost = cost
        if initializeLayers:
            for i in range(0, len(self.layers)):
                if i == len(self.layers) - 1:
                    self.layers[i].initialize(self.getLayerSize(i - 1))
                else:
                    self.layers[i].initialize(self.getLayerSize(i - 1))

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    # This function returns
    # yp - the output of the network
    # a - a list of inputs for each layer of the newtork where
    #     a[i] is the input to layer i
    # z - a list of values for each layer after evaluating layer.z(a) but
    #     before evaluating the nonlinear function for the layer
    def evaluate(self, x):
        curA = x.T
        a = [curA]
        z = []
        for layer in self.layers:
            z.append(layer.z(curA))
            curA = layer.a(z[-1])
            a.append(curA)
        yp = a.pop()
        return yp, a, z

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    def predict(self, a):
        a, _, _ = self.evaluate(a)
        return a.T

    # Train the network given the inputs x and the corresponding observations y
    # The network should be trained for numEpochs iterations using the supplied
    # optimizer
    def train(self, x, y, numEpochs, optimizer):

        # Initialize some stuff
        hist = []
        optimizer.initialize(self.layers)

        # Run for the specified number of epochs
        for epoch in range(0, numEpochs):

            # Feed forward
            # Save the output of each layer in the list a
            # After the network has been evaluated, a should contain the
            # input x and the output of each layer except for the last layer
            yp, a, z = self.evaluate(x)

            # Compute the error
            C = self.cost.fx(yp, y.T)
            d = self.cost.dx(yp, y.T)
            grad = []

            # Backpropogate the error
            idx = len(self.layers)
            for layer, curZ in zip(reversed(self.layers), reversed(z)):
                idx = idx - 1
                # Here, we compute dMSE/dz_i because in the update
                # function for the optimizer, we do not give it
                # the z values we compute from evaluating the network
                grad.insert(0, np.multiply(d, layer.dx(curZ)))
                d = np.dot(layer.W.T, grad[0])

            # Update the errors
            optimizer.update(self.layers, grad, a)

            # Compute the error at the end of the epoch
            yh = self.predict(x)
            C = self.cost.fx(yh, y)
            C = np.mean(C)
            hist.append(C)
        return hist

    def trainBatch(self, x, y, batchSize, numEpochs, optimizer):

        # Copy the data so that we don't affect the original one when shuffling
        x = x.copy()
        y = y.copy()
        hist = []
        n = x.shape[0]

        for epoch in np.arange(0, numEpochs):

            # Shuffle the data
            r = np.arange(0, x.shape[0])
            x = x[r, :]
            y = y[r, :]
            e = []

            # Split the data in chunks and run SGD
            for i in range(0, n, batchSize):
                end = min(i + batchSize, n)
                batchX = x[i:end, :]
                batchY = y[i:end, :]
                e += self.train(batchX, batchY, 1, optimizer)
            hist.append(np.mean(e))

        return hist


########################################################################
######### Part b #######################################################
########################################################################


########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################
def compute_gradient_of_likelihood(single_obj_loc, sensor_loc, single_distance, noise=1):
    """
    Compute the gradient of the loglikelihood function for part a.

    Input:
    single_obj_loc: 1 * d numpy array.
    Location of the single object.

    sensor_loc: k * d numpy array.
    Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: d-dimensional numpy array.

    """
    loc_difference = single_obj_loc - sensor_loc  # k * d.
    phi = np.linalg.norm(loc_difference, axis=1)  # k.
    weight = (phi - single_distance) / phi  # k.

    grad = -np.sum(np.expand_dims(weight, 1) * loc_difference, axis=0) / noise**2  # d
    return grad


########################################################################
######### Part c #################################################
########################################################################
def log_likelihood(obj_loc, sensor_loc, distance, noise=1):
    """
    This function computes the log likelihood (as expressed in Part a).
    Input:
    obj_loc: shape [1,2]
    sensor_loc: shape [7,2]
    distance: shape [7]
    Output:
    The log likelihood function value.
    """
    diff_distance = np.sqrt(np.sum((sensor_loc - obj_loc)**2, axis=1)) - distance
    func_value = -sum((diff_distance)**2) / (2 * noise**2)
    return func_value


########################################################################
######### Part e, f, g #################################################
########################################################################


########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################
def compute_grad_likelihood_part_e(sensor_loc, obj_loc, distance, noise=1):
    """
    Compute the gradient of the loglikelihood function for part d.

    Input:
    sensor_loc: k * d numpy array.
    Location of sensors.

    obj_loc: n * d numpy array.
    Location of the objects.

    distance: n * k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: k * d numpy array.
    """
    grad = np.zeros(sensor_loc.shape)
    for i, single_sensor_loc in enumerate(sensor_loc):
        single_distance = distance[:, i]
        grad[i] = compute_gradient_of_likelihood(single_sensor_loc, obj_loc, single_distance,
                                                 noise)

    return grad


def find_mle_by_grad_descent_part_e(initial_sensor_loc,
                                    obj_loc,
                                    distance,
                                    noise=1,
                                    lr=0.001,
                                    num_iters=1000):
    """
    Compute the gradient of the loglikelihood function for part a.

    Input:
    initial_sensor_loc: k * d numpy array.
    Initialized Location of the sensors.

    obj_loc: n * d numpy array. Location of the n objects.

    distance: n * k dimensional numpy array.
    Observed distance of the n object.

    Output:
    sensor_loc: k * d numpy array. The mle for the location of the object.

    """
    sensor_loc = initial_sensor_loc
    for t in range(num_iters):
        sensor_loc += lr * compute_grad_likelihood_part_e(sensor_loc, obj_loc, distance, noise)

    return sensor_loc


########################################################################

#########  Estimate distance given estimated sensor locations. #########
########################################################################


def compute_distance_with_sensor_and_obj_loc(sensor_loc, obj_loc):
    """
    stimate distance given estimated sensor locations.

    Input:
    sensor_loc: k * d numpy array.
    Location of the sensors.

    obj_loc: n * d numpy array. Location of the n objects.

    Output:
    distance: n * k dimensional numpy array.
    """
    estimated_distance = scipy.spatial.distance.cdist(obj_loc, sensor_loc, metric='euclidean')
    return estimated_distance


########################################################################
#########  Data Generating Functions ###################################
########################################################################


def generate_sensors(num_sensors=7, spatial_dim=2):
    """
    Generate sensor locations.
    Input:
    num_sensors: The number of sensors.
    spatial_dim: The spatial dimension.
    Output:
    sensor_loc: num_sensors * spatial_dim numpy array.
    """
    sensor_loc = 100 * np.random.randn(num_sensors, spatial_dim)
    return sensor_loc


def generate_dataset(sensor_loc,
                     num_sensors=7,
                     spatial_dim=2,
                     num_data=1,
                     original_dist=True,
                     noise=1):
    """
    Generate the locations of n points.

    Input:
    sensor_loc: num_sensors * spatial_dim numpy array. Location of sensor.
    num_sensors: The number of sensors.
    spatial_dim: The spatial dimension.
    num_data: The number of points.
    original_dist: Whether the data are generated from the original
    distribution.

    Output:
    obj_loc: num_data * spatial_dim numpy array. The location of the num_data objects.
    distance: num_data * num_sensors numpy array. The distance between object and
    the num_sensors sensors.
    """
    assert num_sensors, spatial_dim == sensor_loc.shape

    obj_loc = 100 * np.random.randn(num_data, spatial_dim)
    if not original_dist:
        obj_loc += 1000

    distance = scipy.spatial.distance.cdist(obj_loc, sensor_loc, metric='euclidean')
    distance += np.random.randn(num_data, num_sensors) * noise
    return distance, obj_loc


# # 4.a

# In[24]:


import numpy as np
import scipy.spatial


#####################################################################
## Models used for predictions.
#####################################################################
def compute_update(single_obj_loc, sensor_loc, single_distance):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:
    single_obj_loc: 1 * d numpy array.
    Location of the single object.

    sensor_loc: k * d numpy array.
    Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    grad: d-dimensional numpy array.

    """
    loc_difference = single_obj_loc - sensor_loc  # k * d.
    phi = np.linalg.norm(loc_difference, axis=1)  # k.
    grad = loc_difference / np.expand_dims(phi, 1)  # k * 2.
    update = np.linalg.solve(grad.T.dot(grad), grad.T.dot(single_distance - phi))

    return update


def get_object_location(sensor_loc, single_distance, num_iters=20, num_repeats=10):
    """
    Compute the gradient of the log-likelihood function for part a.

    Input:

    sensor_loc: k * d numpy array. Location of sensor.

    single_distance: k dimensional numpy array.
    Observed distance of the object.

    Output:
    obj_loc: 1 * d numpy array. The mle for the location of the object.

    """
    obj_locs = np.zeros((num_repeats, 1, 2))
    distances = np.zeros(num_repeats)
    for i in range(num_repeats):
        obj_loc = np.random.randn(1, 2) * 100
        for t in range(num_iters):
            obj_loc += compute_update(obj_loc, sensor_loc, single_distance)

        distances[i] = np.sum((single_distance - np.linalg.norm(obj_loc - sensor_loc, axis=1))**2)
        obj_locs[i] = obj_loc

    obj_loc = obj_locs[np.argmin(distances)]

    return obj_loc[0]


def generative_model(X, Y, Xs_test, Ys_test):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    initial_sensor_loc = np.random.randn(7, 2) * 100
    estimated_sensor_loc = find_mle_by_grad_descent_part_e(
        initial_sensor_loc, Y, X, lr=0.001, num_iters=1000)

    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array(
            [get_object_location(estimated_sensor_loc, X_test_single) for X_test_single in X_test])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


def oracle_model(X, Y, Xs_test, Ys_test, sensor_loc):
    """
    This function implements the generative model.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    sensor_loc: location of the sensors.
    Output:
    mse: Mean square error on test data.
    """
    mses = []
    for i, X_test in enumerate(Xs_test):
        Y_test = Ys_test[i]
        Y_pred = np.array([
            get_object_location(sensor_loc, X_test_single)
            for X_test_single in X_test
        ])
        mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
        mses.append(mse)
    return mses


# def assemble_feature(x, D):
#     n_feature = x.shape[1]
#     Q = [(np.ones(x.shape[0]), 0, 0)]
#     i = 0
#     while Q[i][1] < D:
#         cx, degree, last_index = Q[i]
#         for j in range(last_index, n_feature):
#             Q.append((cx * x[:, j], degree + 1, j))
#         i += 1
#     return np.column_stack([q[0] for q in Q])

from sklearn.preprocessing import PolynomialFeatures

def linear_regression(X, Y, Xs_test, Ys_test):
    """
    This function performs linear regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    poly = PolynomialFeatures(1)
    Xtrain = poly.fit_transform(X)
    Ytrain = Y
    w = np.linalg.lstsq(Xtrain, Ytrain)[0]
    
    n = len(Ys_test)
    mses = []
    for i in range(n):
        Xt = poly.fit_transform(Xs_test[i])
        Yt = Ys_test[i]
        Y_pred = Xt.dot(w)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Yt)**2, axis=1)))
        mses.append(mse)
    return mses




def poly_regression_second(X, Y, Xs_test, Ys_test):
    """
    This function performs second order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    poly = PolynomialFeatures(2)
    Xtrain = poly.fit_transform(X)
    Ytrain = Y
    w = np.linalg.lstsq(Xtrain, Ytrain)[0]
    
    n = len(Ys_test)
    mses = []
    for i in range(n):
        Xt = poly.fit_transform(Xs_test[i])
        Yt = Ys_test[i]
        Y_pred = Xt.dot(w)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Yt)**2, axis=1)))
        mses.append(mse)
    return mses


def poly_regression_cubic(X, Y, Xs_test, Ys_test):
    """
    This function performs third order polynomial regression.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    poly = PolynomialFeatures(3)
    Xtrain = poly.fit_transform(X)
    Ytrain = Y
    w = np.linalg.lstsq(Xtrain, Ytrain)[0]
    
    n = len(Ys_test)
    mses = []
    for i in range(n):
        Xt = poly.fit_transform(Xs_test[i])
        Yt = Ys_test[i]
        Y_pred = Xt.dot(w)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Yt)**2, axis=1)))
        mses.append(mse)
    return mses


def neural_network(X, Y, Xs_test, Ys_test):
    """
    This function performs neural network prediction.
    Input:
    X: independent variables in training data.
    Y: dependent variables in training data.
    Xs_test: independent variables in test data.
    Ys_test: dependent variables in test data.
    Output:
    mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    Xtrain = X
    Ytrain = Y
    model = Model(Xtrain.shape[1])
    model.addLayer(DenseLayer(100, ReLUActivation()))
    model.addLayer(DenseLayer(100, ReLUActivation()))
    model.addLayer(DenseLayer(Ytrain.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())
    model.train(Xtrain,Ytrain,500,GDOptimizer(eta=0.0000001)) # Learning rate of 0.02
    
    n = len(Ys_test)
    mses = []
    for i in range(n):
        Xt = Xs_test[i]
        Yt = Ys_test[i]
        Y_pred = model.predict(Xt)
        mse = np.mean(np.sqrt(np.sum((Y_pred - Yt)**2, axis=1)))
        mses.append(mse)
        
    return mse


# # 4.b

# In[19]:


def main():
    np.random.seed(0)
    sensor_loc = generate_sensors()
    regular_loc, _ = generate_dataset(
        sensor_loc,
        num_sensors=sensor_loc.shape[0],
        spatial_dim=2,
        num_data=20,
        original_dist=True,
        noise=1)
    shifted_loc, _ = generate_dataset(
        sensor_loc,
        num_sensors=sensor_loc.shape[0],
        spatial_dim=2,
        num_data=20,
        original_dist=False,
        noise=1)

    plt.scatter(sensor_loc[:, 0], sensor_loc[:, 1], label="sensors")
    plt.scatter(regular_loc[:, 0], regular_loc[:, 1], label="regular points")
    plt.scatter(shifted_loc[:, 0], shifted_loc[:, 1], label="shifted points")
    plt.legend()
    plt.savefig("dataset.png")
    plt.show()


if __name__ == "__main__":
    main()


# # 4.c

# In[26]:


def main():
    #############################################################################
    #######################PLOT PART 1###########################################
    #############################################################################
    np.random.seed(0)

    ns = np.arange(10, 310, 20)
    replicates = 5
    num_methods = 6
    num_sets = 3
    mses = np.zeros((len(ns), replicates, num_methods, num_sets))

    def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
        return generate_dataset(
            sensor_loc,
            num_sensors=k,
            spatial_dim=d,
            num_data=n,
            original_dist=original_dist,
            noise=noise)

    for s in range(replicates):
        sensor_loc = generate_sensors()
        X_test, Y_test = generate_data(sensor_loc, n=1000)
        X_test2, Y_test2 = generate_data(
            sensor_loc, n=1000, original_dist=False)
        for t, n in enumerate(ns):
            X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
            Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]
            ### Linear regression:
            mse = linear_regression(X, Y, Xs_test, Ys_test)
            mses[t, s, 0] = mse

            ### Second-order Polynomial regression:
            mse = poly_regression_second(X, Y, Xs_test, Ys_test)
            mses[t, s, 1] = mse

            ### 3rd-order Polynomial regression:
            mse = poly_regression_cubic(X, Y, Xs_test, Ys_test)
            mses[t, s, 2] = mse

            ### Neural Network:
            mse = neural_network(X, Y, Xs_test, Ys_test)
            mses[t, s, 3] = mse

            ### Generative model:
            mse = generative_model(X, Y, Xs_test, Ys_test)
            mses[t, s, 4] = mse

            ### Oracle model:
            mse = oracle_model(X, Y, Xs_test, Ys_test, sensor_loc)
            mses[t, s, 5] = mse

            print('{}th Experiment with {} samples done...'.format(s, n))

    ### Plot MSE for each model.
    plt.figure()
    regressors = [
        'Linear Regression', '2nd-order Polynomial Regression',
        '3rd-order Polynomial Regression', 'Neural Network',
        'Generative Model', 'Oracle Model'
    ]
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 0], axis=1), label=regressors[a])

    plt.title('Error on training data for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('train_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 1], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from the same distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_same_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 2], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from a different distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_different_mse.png')
    plt.show()


if __name__ == '__main__':
    import datetime
    print("C start time:", datetime.datetime.now())
    main()
    print("C end time:", datetime.datetime.now())


# **Briefly  describe  your  observations.   What  are  the  strengths  and  weaknesses  of  each
# model?**
# 
# 

# # 4.d

# In[20]:


activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
lr = dict(ReLU=0.0000007,tanh=0.00001,linear=0.005)

def neural_network(X, Y, X_test, Y_test, num_neurons, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_neurons: number of neurons in each layer
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    model = Model(X.shape[1])
    model.addLayer(DenseLayer(num_neurons, activations[activation]))
    model.addLayer(DenseLayer(num_neurons, activations[activation]))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())
    model.train(X,Y,500,GDOptimizer(eta=lr[activation]))
    Y_pred = model.predict(X_test)
    mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
    print(mse)
    return mse


#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_neuronss = np.arange(100, 550, 50)
mses = np.zeros((len(num_neuronss), 2))

# for s in range(replicates):

sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_neurons in enumerate(num_neuronss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_neurons, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} neurons done...'.format(num_neurons))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_neuronss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of neurons')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('num_neurons.png')


# # 4.e

# In[21]:


activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
lr = dict(ReLU=0.0000007,tanh=0.00001,linear=0.005)

def neural_network(X, Y, X_test, Y_test, num_layers, activation):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_layers: number of layers in neural network
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    # Have K hidden layers and L neurons per layer.
    # Want total # weights to be 10000
    # X.shape[1]*L + (k-1)*(L*L) + L*Y.shape[1] = 10000
    # => (k-1)*L^2 + L*(X.shape + Y.shape[1]) - 10000 = 0
    # Let x = X.shape[1], y = Y.shape[1].
    # Quadratic formula:
    # L = (-(x + y) +- sqrt((x+y)**2 - 4*(k-1)*(-10000))) / 2*(k-1)
    # Since expression on the right is positive and >= (x+1) for k >= 0,
    # only positive makes sense. 
    k = num_layers
    x = X.shape[1]
    y = Y.shape[1]
    L = (-(x + y) + np.sqrt((x + y)**2 + 40000*(k-1)))/2*(k-1)
    L = int(L)
    
    model = Model(X.shape[1])
    for i in range(num_layers):
        model.addLayer(DenseLayer(L, activations[activation]))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())
    model.train(X,Y,500,GDOptimizer(eta=lr[activation])) # Learning rate of 0.02
    
    Y_pred = model.predict(X_test)
    mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
    print(mse)
    return mse


#############################################################################
#######################PLOT PART 2###########################################
#############################################################################
def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
    return generate_dataset(
        sensor_loc,
        num_sensors=k,
        spatial_dim=d,
        num_data=n,
        original_dist=original_dist,
        noise=noise)


np.random.seed(0)
n = 200
num_layerss = [1, 2, 3, 4]
mses = np.zeros((len(num_layerss), 2))

# for s in range(replicates):
sensor_loc = generate_sensors()
X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
X_test, Y_test = generate_data(sensor_loc, n=1000)
for t, num_layers in enumerate(num_layerss):
    ### Neural Network:
    mse = neural_network(X, Y, X_test, Y_test, num_layers, "ReLU")
    mses[t, 0] = mse

    mse = neural_network(X, Y, X_test, Y_test, num_layers, "tanh")
    mses[t, 1] = mse

    print('Experiment with {} layers done...'.format(num_layers))

### Plot MSE for each model.
plt.figure()
activation_names = ['ReLU', 'Tanh']
for a in range(2):
    plt.plot(num_layerss, mses[:, a], label=activation_names[a])

plt.title('Error on validation data verses number of neurons')
plt.xlabel('Number of layers')
plt.ylabel('Average Error')
plt.legend(loc='best')
plt.yscale('log')
plt.savefig('num_layers.png')


# # 4.f

# In[27]:


activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
lr = dict(ReLU=0.0000007,tanh=0.00001,linear=0.005)

def neural_network(X, Y, Xs_test, Ys_test):
    """
    This function performs neural network prediction.
    Input:
        X: independent variables in training data.
        Y: dependent variables in training data.
        X_test: independent variables in test data.
        Y_test: dependent variables in test data.
        num_layers: number of layers in neural network
        activation: type of activation, ReLU or tanh
    Output:
        mse: Mean square error on test data.
    """
    ## YOUR CODE HERE
    #################
    
    num_layers = 4
    activation = "ReLU"
    
    k = num_layers
    x = X.shape[1]
    y = Y.shape[1]
    L = (-(x + y) + np.sqrt((x + y)**2 + 40000*(k-1)))/2*(k-1)
    L = int(L)
    
    model = Model(X.shape[1])
    for i in range(num_layers):
        model.addLayer(DenseLayer(L, activations[activation]))
    model.addLayer(DenseLayer(Y.shape[1],LinearActivation()))
    model.initialize(QuadraticCost())
    model.train(X,Y,500,GDOptimizer(eta=lr[activation])) # Learning rate of 0.02
    
    Y_pred = model.predict(X_test)
    mse = np.mean(np.sqrt(np.sum((Y_pred - Y_test)**2, axis=1)))
    print(mse)
    return mse


def main():
    #############################################################################
    #######################PLOT PART 1###########################################
    #############################################################################
    np.random.seed(0)

    ns = np.arange(10, 310, 20)
    replicates = 5
    num_methods = 6
    num_sets = 3
    mses = np.zeros((len(ns), replicates, num_methods, num_sets))

    def generate_data(sensor_loc, k=7, d=2, n=1, original_dist=True, noise=1):
        return generate_dataset(
            sensor_loc,
            num_sensors=k,
            spatial_dim=d,
            num_data=n,
            original_dist=original_dist,
            noise=noise)

    for s in range(replicates):
        sensor_loc = generate_sensors()
        X_test, Y_test = generate_data(sensor_loc, n=1000)
        X_test2, Y_test2 = generate_data(
            sensor_loc, n=1000, original_dist=False)
        for t, n in enumerate(ns):
            X, Y = generate_data(sensor_loc, n=n)  # X [n * 2] Y [n * 7]
            Xs_test, Ys_test = [X, X_test, X_test2], [Y, Y_test, Y_test2]
            ### Linear regression:
            mse = linear_regression(X, Y, Xs_test, Ys_test)
            mses[t, s, 0] = mse

            ### Second-order Polynomial regression:
            mse = poly_regression_second(X, Y, Xs_test, Ys_test)
            mses[t, s, 1] = mse

            ### 3rd-order Polynomial regression:
            mse = poly_regression_cubic(X, Y, Xs_test, Ys_test)
            mses[t, s, 2] = mse

            ### Neural Network:
            mse = neural_network(X, Y, Xs_test, Ys_test)
            mses[t, s, 3] = mse

            ### Generative model:
            mse = generative_model(X, Y, Xs_test, Ys_test)
            mses[t, s, 4] = mse

            ### Oracle model:
            mse = oracle_model(X, Y, Xs_test, Ys_test, sensor_loc)
            mses[t, s, 5] = mse

            print('{}th Experiment with {} samples done...'.format(s, n))

    ### Plot MSE for each model.
    plt.figure()
    regressors = [
        'Linear Regression', '2nd-order Polynomial Regression',
        '3rd-order Polynomial Regression', 'Neural Network',
        'Generative Model', 'Oracle Model'
    ]
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 0], axis=1), label=regressors[a])

    plt.title('Error on training data for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('train_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 1], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from the same distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_same_mse.png')
    plt.show()

    plt.figure()
    for a in range(6):
        plt.plot(ns, np.mean(mses[:, :, a, 2], axis=1), label=regressors[a])

    plt.title(
        'Error on test data from a different distribution for Various models')
    plt.xlabel('Number of training data')
    plt.ylabel('Average Error')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.savefig('val_different_mse.png')
    plt.show()


if __name__ == '__main__':
    main()

