
# coding: utf-8

# # Question 2 B

# In[39]:


""" Tools for calculating Gradient Descent for ||Ax-b||. """
import matplotlib.pyplot as plt
import numpy as np




def compute_gradient(A, b, x):
    """Computes the gradient of ||Ax-b|| with respect to x."""
    return np.dot(A.T, (np.dot(A, x) - b)) / np.linalg.norm(np.dot(A, x) - b)


def compute_update(A, b, x, step_count, step_size):
    """Computes the new point after the update at x."""
    return x - step_size(step_count) * compute_gradient(A, b, x)


def compute_updates(A, b, p, total_step_count, step_size):
    """Computes several updates towards the minimum of ||Ax-b|| from p.

    Params:
        b: in the equation ||Ax-b||
        p: initialization point
        total_step_count: number of iterations to calculate
        step_size: function for determining the step size at step i
    """
    positions = [np.array(p)]
    n = 0 
    for k in range(total_step_count):
        n += 1
        positions.append(compute_update(A, b, positions[-1], k, step_size))
    print(n)
    return np.array(positions)




    
################################################################################
# TODO(student): Input Variables
A = np.array([[1, 0], [0, 1]])  # do not change this until the last part
b = np.array([4.5, 6])  # b in the equation ||Ax-b||
initial_position = np.array([0, 0])  # position at iteration 0
total_step_count = 15 # number of GD steps to take
step_size = lambda i: 1  # step size at iteration i
################################################################################

# computes desired number of steps of gradient descent
positions = compute_updates(A, b, initial_position, total_step_count, step_size)

# print out the values of the x_i
print(positions)
print(np.dot(np.linalg.inv(A), b))

# plot the values of the x_i
plt.scatter(positions[:, 0], positions[:, 1], c='blue')
plt.scatter(np.dot(np.linalg.inv(A), b)[0],
            np.dot(np.linalg.inv(A), b)[1], c='red')
plt.plot()
plt.show()


# ALthough the function is convex, the gradient descent method will not find the optimal solution. 
# 
# **It will never get within 0.01 of the optimal solution. ** The gradient descent method clearly begins oscillating between [4.8 6.4] and [4.2 5.6]. The error in these step are [0.3, 0.4], which does not satisfy epsilon. Does not converge to the optimal solution.
# 
# The step size is constant and is too large.
# 
# 
# **For General b**
# 
# 
# 
# 

# In[54]:


################################################################################
# TODO(student): Input Variables
A = np.array([[1, 0], [0, 1]])  # do not change this until the last part
b = np.array([8, 8])  # b in the equation ||Ax-b||
initial_position = np.array([0, 0])  # position at iteration 0
total_step_count = 15  # number of GD steps to take
step_size = lambda i: 1  # step size at iteration i
################################################################################

# computes desired number of steps of gradient descent
positions = compute_updates(A, b, initial_position, total_step_count, step_size)

# print out the values of the x_i
print(positions)
print(np.dot(np.linalg.inv(A), b))

# plot the values of the x_i
plt.scatter(positions[:, 0], positions[:, 1], c='blue')
plt.scatter(np.dot(np.linalg.inv(A), b)[0],
            np.dot(np.linalg.inv(A), b)[1], c='red')
plt.plot()
plt.show()


# The same can be said for general b. Gradient descent will not always reach the optimal solution. 
# 
# The step size is constant and is too large. Therefore it will not always converge. 
# 

# # Question 2 C

# In[60]:


################################################################################
# TODO(student): Input Variables
A = np.array([[1, 0], [0, 1]])  # do not change this until the last part
b = np.array([4.5, 6])  # b in the equation ||Ax-b||
initial_position = np.array([0, 0])  # position at iteration 0
total_step_count = 5000  # number of GD steps to take
step_size = lambda i: (5/6)**i # step size at iteration i
################################################################################

# computes desired number of steps of gradient descent
positions = compute_updates(A, b, initial_position, total_step_count, step_size)

# print out the values of the x_i
print(positions)
print(np.dot(np.linalg.inv(A), b))

# plot the values of the x_i
plt.scatter(positions[:, 0], positions[:, 1], c='blue')
plt.scatter(np.dot(np.linalg.inv(A), b)[0],
            np.dot(np.linalg.inv(A), b)[1], c='red')
plt.plot()
plt.show()


# computes desired number of steps of gradient descent
b = [3, 3]
positions = compute_updates(A, b, initial_position, total_step_count, step_size)

# print out the values of the x_i
print(positions)
print(np.dot(np.linalg.inv(A), b))

# plot the values of the x_i
plt.scatter(positions[:, 0], positions[:, 1], c='blue')
plt.scatter(np.dot(np.linalg.inv(A), b)[0],
            np.dot(np.linalg.inv(A), b)[1], c='red')
plt.plot()
plt.show()


# No the gradient descent will not find the optimal solution. This is because our step size gets exponentially smaller as we iterate. It does not converge. The plot does a good job illustrating this point. 
# 
# **For General b**
# 
# Conceptually, if our optimal solution x* is close enough to our intial x than our gradient descent method will converge to the optimal solution. 
# 
# 
# 
# 

# # Question 2 D

# In[57]:


################################################################################
# TODO(student): Input Variables
A = np.array([[1, 0], [0, 1]])  # do not change this until the last part
b = np.array([4.5, 6])  # b in the equation ||Ax-b||
initial_position = np.array([0, 0])  # position at iteration 0
total_step_count = 5000  # number of GD steps to take
step_size = lambda i: 1/(i + 1) # step size at iteration i
################################################################################

# computes desired number of steps of gradient descent
positions = compute_updates(A, b, initial_position, total_step_count, step_size)

# print out the values of the x_i
print(positions)
print(np.dot(np.linalg.inv(A), b))

# plot the values of the x_i
plt.scatter(positions[:, 0], positions[:, 1], c='blue')
plt.scatter(np.dot(np.linalg.inv(A), b)[0],
            np.dot(np.linalg.inv(A), b)[1], c='red')
plt.plot()
plt.show()


# Our gradient descent method converges to the optimal solution. The expression for the number of iterations required for convergence is in the hand written part of this HW submission. 
# 
# This step size makes our gradient descent method work for general b. 

# # Question 2 E 

# In[71]:


################################################################################
# TODO(student): Input Variables
A = np.array([[10, 0], [0, 1]])  # do not change this until the last part
b = np.array([4.5, 6])  # b in the equation ||Ax-b||
initial_position = np.array([0, 0])  # position at iteration 0
total_step_count = 5000  # number of GD steps to take
steps = [lambda i : 1, lambda i: (5/6)**i, lambda i: 1/(i + 1)] # step size at iteration i
################################################################################

# plot the values of the x_i
for i in range(3):
    print(i)
    step_size = steps[i]
    # computes desired number of steps of gradient descent
    positions = compute_updates(A, b, initial_position, total_step_count, step_size)
    # print out the values of the x_i
    print(positions)
    print(np.dot(np.linalg.inv(A), b))
    plt.scatter(positions[:, 0], positions[:, 1], c='blue')
    plt.scatter(np.dot(np.linalg.inv(A), b)[0],
                np.dot(np.linalg.inv(A), b)[1], c='red')
    print(A, "Step Size Method: " + str(i))
    plt.plot()
    plt.show()





################################################################################
# TODO(student): Input Variables
A = np.array([[15, 8], [6, 5]])  # do not change this until the last part
################################################################################

# plot the values of the x_i
for i in range(3):
    print(i)
    step_size = steps[i]
    # computes desired number of steps of gradient descent
    positions = compute_updates(A, b, initial_position, total_step_count, step_size)
    # print out the values of the x_i
    print(positions)
    print(np.dot(np.linalg.inv(A), b))
    plt.scatter(positions[:, 0], positions[:, 1], c='blue')
    plt.scatter(np.dot(np.linalg.inv(A), b)[0],
                np.dot(np.linalg.inv(A), b)[1], c='red')
    print(A, "Step Size Method: " + str(i))
    plt.plot()
    plt.show()


# The step size function that we utilized for Q2.d should converge to the optimal solution regardless of the matrix A. 
# 
# Changing A only changes what we are minimizing. 
# 
# 

# # Question 5 

# In[98]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
    #
    def update(self, layers, g, a):
        m = a[0].shape[1]
        for layer, curGrad, curA in zip(layers, g, a):
            # TODO: PART F #########################################################################
            # Compute the gradients for layer.W and layer.b using the gradient for the output of the
            # layer curA and the gradient of the output curGrad
            # Use the gradients to update the weight and the bias for the layer
            #
            # Normalize the learning rate by m (defined above), the number of training examples input
            # (in parallel) to the network.
            #
            # It may help to think about how you would calculate the update if we input just one
            # training example at a time; then compute a mean over these individual update values.
            # ######################################################################################
            Wgrad_layer = curGrad * curA 
            bgrad_layer = curGrad
            
            layer.W = layer.W - self.eta/m * np.mean(Wgrad_layer)
            layer.b = layer.b - self.eta/m * np.mean(bgrad_layer)

# Cost function used to compute prediction errors
class QuadraticCost(object):

    # Compute the squared error between the prediction yp and the observation y
    # This method should compute the cost per element such that the output is the
    # same shape as y and yp
    @staticmethod
    def fx(y,yp):
        # TODO: PART B #########################################################################
        # Implement me
        return 0.5*np.square(y - yp)
        # ######################################################################################
        

    # Derivative of the cost function with respect to yp
    @staticmethod
    def dx(y,yp):
        # TODO: PART B #########################################################################
        # Implement me
        return yp - y
        # ######################################################################################
        

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
        # TODO: PART C #################################################################################
        return np.tanh(z)
        # return (np.exp(z) - np.exp(-z))/(np.exp(z)+np.exp(-z))
        # ######################################################################################
        

    # Compute the derivative of the tanh function with respect to z
    @staticmethod
    def dx(z):
        # TODO: PART C #########################################################################
        return 1 - TanhActivation.fx(z)**2
        #return 1 - (np.exp(z) - np.exp(-z))**2/(np.exp(z)+np.exp(-z))**2
        # ######################################################################################
        

# Rectified linear unit
class ReLUActivation(object):
    @staticmethod
    def fx(z):
        # TODO: PART C #########################################################################
        return np.maximum(z, 0)
    
        # ######################################################################################
        

    @staticmethod
    def dx(z):
        # TODO: PART C #########################################################################
        
        ret = np.copy(z)
        ret[ret <= 0] = 0
        ret[ret > 0] = 1
        return ret
        # ######################################################################################
        

# Linear activation
class LinearActivation(object):
    @staticmethod
    def fx(z):
        # TODO: PART C #########################################################################
        return z
        # ######################################################################################
        

    @staticmethod
    def dx(z):
        # TODO: PART C #########################################################################
        return np.ones(z.shape)
        # ######################################################################################
        

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
        self.W = np.random.normal(0, s,
                                   (self.numNodes,fanIn))
        self.b = np.random.uniform(-1,1,(self.numNodes,1))

    # Apply the activation function of the layer on the input z
    def a(self, z):
        return self.activation.fx(z)

    # Compute the linear part of the layer
    # The input a is an n x k matrix where n is the number of samples
    # and k is the dimension of the previous layer (or the input to the network)
    def z(self, a):
        return self.W.dot(a) + self.b # Note, this is implemented where we assume a is k x n

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
            for i in range(0,len(self.layers)):
                if i == len(self.layers) - 1:
                    self.layers[i].initialize(self.getLayerSize(i-1))
                else:
                    self.layers[i].initialize(self.getLayerSize(i-1))

    # Compute the output of the network given some input a
    # The matrix a has shape n x k where n is the number of samples and
    # k is the dimension
    # This function returns
    # yp - the output of the network
    # a - a list of inputs for each layer of the newtork where
    #     a[i] is the input to layer i
    #     (note this does not include the network output!)
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
        a,_,_ = self.evaluate(a)
        return a.T

    # Computes the gradients at each layer. y is the true labels, yp is the
    # predicted labels, and z is a list of the intermediate values in each
    # layer. Returns the gradients and the forward pass outputs (per layer).
    #
    # In particular, we compute dMSE/dz_i. The reasoning behind this is that
    # in the update function for the optimizer, we do not give it the z values
    # we compute from evaluating the network.
    def compute_grad(self, x, y):
        # Feed forward, computing outputs of each layer and
        # intermediate outputs before the non-linearities
        yp, a, z = self.evaluate(x)

        # d represents (dMSE / da_i) that you derive in part (e);
        #   it is inialized here to be (dMSE / dyp)
        d = self.cost.dx(y.T, yp)
        grad = []

        # Backpropogate the error
        for layer, curZ in zip(reversed(self.layers),reversed(z)):
            # TODO: PART D #########################################################################
            # Compute the gradient of the output of each layer with respect to the error
            # grad[i] should correspond with the gradient of the output of layer i
            #   before the activation is applied (dMSE / dz_i); be sure values are stored
            #   in the correct ordering!
            
            ## They compute the first d for you
            
            grad = [d*layer.dx(curZ)] + grad
            
            d = layer.W.T.dot(d*layer.dx(curZ))
                              
            #layer.activation(curZ)
            
            
            # ######################################################################################

        return grad, a

    # Computes the gradients at each layer. y is the true labels, yp is the
    # predicted labels, and z is a list of the intermediate values in each
    # layer. Uses numerical derivatives to solve rather than symbolic derivatives.
    # Returns the gradients and the forward pass outputs (per layer).
    #
    # In particular, we compute dMSE/dz_i. The reasoning behind this is that
    # in the update function for the optimizer, we do not give it the z values
    # we compute from evaluating the network.
    def numerical_grad(self, x, y, delta=1e-4):

        # computes the loss function output when starting from the ith layer
        # and inputting z_i
        def compute_cost_from_layer(layer_i, z_i):
            cost = self.layers[layer_i].a(z_i)
            for layer in self.layers[layer_i+1:]:
                cost = layer.a(layer.z(cost))
            return self.cost.fx(y.T, cost)

        # numerically computes the gradient of the error with respect to z_i
        def compute_grad_from_layer(layer_i, inp):
            mask = np.zeros(self.layers[layer_i].b.shape)
            grad_z = []
            # iterate to compute gradient of each variable in z_i, one at a time
            for i in range(mask.shape[0]):
                mask[i] = 1
                delta_p_output = compute_cost_from_layer(layer_i, inp+mask*delta)
                delta_n_output = compute_cost_from_layer(layer_i, inp-mask*delta)
                grad_z.append((delta_p_output - delta_n_output) / (2 * delta))
                mask[i] = 0;

            return np.vstack(grad_z)

        _, a, _ = self.evaluate(x)

        grad = []
        i = 0
        curA = x.T
        for layer in self.layers:
            curA = layer.z(curA)
            grad.append(compute_grad_from_layer(i, curA))
            curA = layer.a(curA)
            i += 1


        return grad, a

    # Train the network given the inputs x and the corresponding observations y
    # The network should be trained for numEpochs iterations using the supplied
    # optimizer
    def train(self, x, y, numEpochs, optimizer):

        # Initialize some stuff
        n = x.shape[0]
        x = x.copy()
        y = y.copy()
        hist = []
        optimizer.initialize(self.layers)

        # Run for the specified number of epochs
        for epoch in range(0,numEpochs):

            # Compute the gradients
            grad, a = self.compute_grad(x, y)

            # Update the network weights
            optimizer.update(self.layers, grad, a)

            # Compute the error at the end of the epoch
            yh = self.predict(x)
            C = self.cost.fx(y, yh)
            C = np.mean(C)
            hist.append(C)
        return hist

if __name__ == '__main__':
    # switch these statements to True to run the code for the corresponding parts
    # PART E
    DEBUG_MODEL = True
    # Part G
    BASE_MODEL = True
    # Part H
    DIFF_SIZES = True
    # Part I
    RIDGE = True
    # Part J
    SGD = False



    # Generate the training set
    np.random.seed(9001)
    x=np.random.uniform(-np.pi,np.pi,(1000,1))
    y=np.sin(x)
    xLin=np.linspace(-np.pi,np.pi,250).reshape((-1,1))
    yHats = {}

    activations = dict(ReLU=ReLUActivation,
                       tanh=TanhActivation,
                       linear=LinearActivation)
    lr = dict(ReLU=0.02,tanh=0.02,linear=0.005)
    names = ['ReLU','linear','tanh']

    #### PART F ####
    if DEBUG_MODEL:
        print('Debugging gradients..')
        # Build the model
        activation = activations["ReLU"]
        model = Model(x.shape[1])
        model.addLayer(DenseLayer(10,activation()))
        model.addLayer(DenseLayer(10,activation()))
        model.addLayer(DenseLayer(1,LinearActivation()))
        model.initialize(QuadraticCost())

        grad, _ = model.compute_grad(x, y)
        n_grad, _ = model.numerical_grad(x, y)
        for i in range(len(grad)):
            print('squared difference of layer %d:' % i, np.linalg.norm(grad[i] - n_grad[i]))


    #### PART G ####
    if BASE_MODEL:
        print('\n----------------------------------------\n')
        print('Standard fully connected network')
        for key in names:
            # Build the model
            activation = activations[key]
            model = Model(x.shape[1])
            model.addLayer(DenseLayer(100,activation()))
            model.addLayer(DenseLayer(100,activation()))
            model.addLayer(DenseLayer(1,LinearActivation()))
            model.initialize(QuadraticCost())

            # Train the model and display the results
            hist = model.train(x,y,500,GDOptimizer(eta=lr[key]))
            yHat = model.predict(x)
            yHats[key] = model.predict(xLin)
            error = np.mean(np.square(yHat - y))/2
            print(key+' MSE: '+str(error))
            plt.plot(hist)
            plt.title(key+' Learning curve')
            plt.show()

        # Plot the approximations
        font = {'family' : 'DejaVu Sans',
            'weight' : 'bold',
                'size'   : 12}
        matplotlib.rc('font', **font)
        y = np.sin(xLin)
        for key in activations:
            plt.plot(xLin,y)
            plt.plot(xLin,yHats[key])
            plt.title(key+' approximation')
            plt.savefig(key+'-approx.png')
            plt.show()

    # Train with different sized networks
    #### PART H ####
    if DIFF_SIZES:
        print('\n----------------------------------------\n')
        print('Training with various sized network')
        names = ['ReLU', 'tanh']
        sizes = [5,10,25,50]
        widths = [1,2,3]
        errors = {}
        y = np.sin(x)
        for key in names:
            error = []
            for width in widths:
                for size in sizes:
                    activation = activations[key]
                    model = Model(x.shape[1])
                    for _ in range(width):
                        model.addLayer(DenseLayer(size,activation()))
                    model.addLayer(DenseLayer(1,LinearActivation()))
                    model.initialize(QuadraticCost())
                    hist = model.train(x,y,500,GDOptimizer(eta=lr[key]))
                    yHat = model.predict(x)
                    yHats[key] = model.predict(xLin)
                    e = np.mean(np.square(yHat - y))/2
                    error.append(e)
            errors[key] = np.asarray(error).reshape((len(widths),len(sizes)))

        # Print the results
        for key in names:
            error = errors[key]
            print(key+' MSE Error')
            header = '{:^8}'
            for _ in range(len(sizes)):
                header += ' {:^8}'
            headerText = ['Layers'] + [str(s)+' nodes' for s in sizes]
            print(header.format(*headerText))
            for width,row in zip(widths,error):
                text = '{:>8}'
                for _ in range(len(row)):
                    text += ' {:<8}'
                rowText = [str(width)] + ['{0:.5f}'.format(r) for r in row]
                print(text.format(*rowText))

    # Perform ridge regression on the last layer of the network
    #### PART I ####
    if RIDGE:
        print('\n----------------------------------------\n')
        print('Running ridge regression on last layer')
        from sklearn.linear_model import Ridge
        errors = {}
        for key in names:
            error = []
            sizes = [5,10,25,50]
            widths = [1,2,3]
            for width in widths:
                for size in sizes:
                    activation = activations[key]
                    model = Model(x.shape[1])
                    for _ in range(width):
                        model.addLayer(DenseLayer(size,activation()))
                    model.initialize(QuadraticCost())
                    ridge = Ridge(alpha=0.1)
                    X = model.predict(x)
                    ridge.fit(X,y)
                    yHat = ridge.predict(X)
                    e = np.mean(np.square(yHat - y))/2
                    error.append(e)
            errors[key] = np.asarray(error).reshape((len(widths),len(sizes)))

        # Print the results
        for key in names:
            error = errors[key]
            print(key+' MSE Error')
            header = '{:^8}'
            for _ in range(len(sizes)):
                header += ' {:^8}'
            headerText = ['Layers'] + [str(s)+' nodes' for s in sizes]
            print(header.format(*headerText))
            for width,row in zip(widths,error):
                text = '{:>8}'
                for _ in range(len(row)):
                    text += ' {:<8}'
                rowText = [str(width)] + ['{0:.5f}'.format(r) for r in row]
                print(text.format(*rowText))

        # Plot the results
        for key in names:
            for width,row in zip(widths,error):
                layer = ' layers'
                if width == 1:
                    layer = ' layer'
                plt.semilogy(row,label=str(width)+layer)
            plt.title('MSE for ridge regression with '+key+' activation')
            plt.xticks(range(len(sizes)),sizes)
            plt.xlabel('Layer size')
            plt.ylabel('MSE')
            plt.legend()
            plt.savefig(key+'-ridge.png')
            plt.show()

    #### BONUS PART J ####
    if SGD:
        # Test for SGD... Implement!
        pass


# ## Question 5 Comment
# 
# Layer size is proportional to MSE. For both nonlinearities the MSE decreases with layer size. However there is not an appreciable difference as we increase the number of layers. 

# ## Question 4 

# In[ ]:


import numpy as np
import scipy.spatial
import matplotlib 
import matplotlib.pyplot as plt
########################################################################
#########  Data Generating Functions ###################################
########################################################################
VAR_MEASUREMENT_NOISE = 1

def generate_sensors(k = 7, d = 2):
    """
    Generate sensor locations. 
    Input:
    k: The number of sensors.
    d: The spatial dimension.
    Output:
    sensor_loc: k * d numpy array.
    """
    sensor_loc = 100*np.random.randn(k,d)
    return sensor_loc

def generate_data(sensor_loc, k = 7, d = 2, 
                 n = 1, original_dist = True, sigma_s = 100):
    """
    Generate the locations of n points and distance measurements.  

    Input:
    sensor_loc: k * d numpy array. Location of sensor. 
    k: The number of sensors.
    d: The spatial dimension.
    n: The number of points.
    original_dist: Whether the data are generated from the original 
    distribution. 
    sigma_s: the standard deviation of the distribution 
    that generate each object location.

    Output:
    obj_loc: n * d numpy array. The location of the n objects. 
    distance: n * k numpy array. The distance between object and 
    the k sensors. 
    """
    assert k, d == sensor_loc.shape

    obj_loc = sigma_s*np.random.randn(n, d)
    if not original_dist:
        obj_loc = sigma_s*np.random.randn(n, d)+([300,300])

    distance = scipy.spatial.distance.cdist(obj_loc, 
                                           sensor_loc, 
                                           metric='euclidean')
    distance += np.random.randn(n, k) 
    return obj_loc, distance

def generate_data_given_location(sensor_loc, obj_loc, k = 7, d = 2):
    """
    Generate the distance measurements given location of a single object and sensor. 

    Input:
    obj_loc: 1 * d numpy array. Location of object
    sensor_loc: k * d numpy array. Location of sensor. 
    k: The number of sensors.
    d: The spatial dimension. 

    Output: 
    distance: 1 * k numpy array. The distance between object and 
    the k sensors. 
    """
    assert k, d == sensor_loc.shape 

    distance = scipy.spatial.distance.cdist(obj_loc, 
                                           sensor_loc, 
                                           metric='euclidean')
    distance += np.random.randn(1, k)*VAR_MEASUREMENT_NOISE 
    return obj_loc, distance


# In[112]:


########################################################################
######### Part b ###################################
########################################################################

########################################################################
#########  Gradient Computing and MLE ###################################
########################################################################
def compute_gradient_of_likelihood(single_obj_loc, sensor_loc, 
                                single_distance):
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
#     grad = np.zeros_like(single_obj_loc)
#     #Your code: implement the gradient of loglikelihood   
#     d = len(single_obj_loc[0])
#     k = len(single_distance)

#     for i in range(d):
#         #grad[0][i] = np.sum([2*(1 - single_distance[j]*(1/np.linalg.norm(sensor_loc[j]-single_obj_loc[0])))*(sensor_loc[j][i] - single_obj_loc[0][i]) for j in range(k)])
#         for j in range(k):
#             norm = np.linalg.norm(sensor_loc[j]-single_obj_loc[0]) #np.sqrt(np.square(sensor_loc[j]-single_obj_loc[0]))
#             scalar = 2*(sensor_loc[j][i] - single_obj_loc[0][i])

#             grad[0][i] += scalar*(norm - single_distance[j])/norm
    grad = np.zeros_like(single_obj_loc)


    for i in range(len(single_distance)):
        actual_dist = np.sqrt(np.sum((sensor_loc[i] - single_obj_loc) ** 2))
        observed_dist = single_distance[i]
        grad_i = single_obj_loc - sensor_loc[i]
        grad_i *= ((actual_dist - observed_dist) / actual_dist)
        grad += grad_i

    grad *= -1

    return grad

def find_mle_by_grad_descent_part_b(initial_obj_loc, 
           sensor_loc, single_distance, lr=0.001, num_iters = 10000):
    """
    Compute the gradient of the loglikelihood function for part a.   

    Input:
    initial_obj_loc: 1 * d numpy array. 
    Initialized Location of the single object.

    sensor_loc: k * d numpy array. Location of sensor.

    single_distance: k dimensional numpy array. 
    Observed distance of the object.

    Output:
    obj_loc: 1 * d numpy array. The mle for the location of the object.

    """    
    obj_loc = initial_obj_loc
    # Your code: do gradient descent
    for i in range(num_iters):
        grad = compute_gradient_of_likelihood(obj_loc, sensor_loc, single_distance)
        obj_loc = obj_loc + grad*lr

    return obj_loc


# In[113]:


########################################################################
#########  MAIN ########################################################
########################################################################

# Your code: set some appropriate learning rate here
def run(lr):
    print("Learning Rate is:", lr)
    np.random.seed(0)
    sensor_loc = generate_sensors()
    obj_loc, distance = generate_data(sensor_loc)
    single_distance = distance[0]
    print('The real object location is')
    print(obj_loc)
    # Initialized as [0,0]
    initial_obj_loc = np.array([[0.,0.]]) 
    estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
               sensor_loc, single_distance, lr=lr, num_iters = 10000)
    print('The estimated object location with zero initialization is')
    print(estimated_obj_loc)

    # Random initialization.
    initial_obj_loc = np.random.randn(1,2)
    estimated_obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
               sensor_loc, single_distance, lr=lr, num_iters = 10000)
    print('The estimated object location with random initialization is')
    print(estimated_obj_loc)
    print('\n\n')
    
lrs = [1.0, 0.01, 0.001, 0.0001]
for lr in lrs:
    run(lr)


# In[ ]:


########################################################################
######### Part c #################################################
########################################################################
def log_likelihood(obj_loc, sensor_loc, distance): 
    """
    This function computes the log likelihood (as expressed in Part a).
    Input: 
    obj_loc: shape [1,2]
    sensor_loc: shape [7,2]
    distance: shape [7]
    Output: 
    The log likelihood function value. 
    """  
    # Your code: compute the log likelihood
#     func_value = -1*np.sum(
#         [np.square(np.linalg.norm(sensor_loc[i] - obj_loc[0]) - distance[i])
#          for i in range(len(distance))])

#     return func_value
    func_value = 0.0

    for i in range(len(distance)):
        actual_dist = np.sqrt(np.sum((sensor_loc[i] - obj_loc) ** 2))
        observed_dist = distance[i]
        func_value += (actual_dist - observed_dist) ** 2

    func_value *= -1

    return func_value


# In[ ]:


########################################################################
######### Compute the function value at local minimum for all experiments.###
########################################################################
def run4(num_sensors):
    #num_sensors = 7

    np.random.seed(100)
    sensor_loc = generate_sensors(k=num_sensors)

    # num_data_replicates = 10
    num_gd_replicates = 100

    obj_locs = [[[i,i]] for i in np.arange(0,1000,100)]

    func_values = np.zeros((len(obj_locs),10, num_gd_replicates))
    # record sensor_loc, obj_loc, 100 found minimas
    minimas = np.zeros((len(obj_locs), 10, num_gd_replicates, 2))
    true_object_locs = np.zeros((len(obj_locs), 10, 2))

    for i, obj_loc in enumerate(obj_locs): 
        for j in range(10):
            obj_loc, distance = generate_data_given_location(sensor_loc, obj_loc, 
                                                           k = num_sensors, d = 2)
            true_object_locs[i, j, :] = np.array(obj_loc)

            for gd_replicate in range(num_gd_replicates): 
                initial_obj_loc = np.random.randn(1,2)* (100 * i+1)
                obj_loc = find_mle_by_grad_descent_part_b(initial_obj_loc, 
                         sensor_loc, distance[0], lr=0.1, num_iters = 1000) 
                minimas[i, j, gd_replicate, :] = np.array(obj_loc)
                func_value = log_likelihood(obj_loc, sensor_loc, distance[0])
                func_values[i, j, gd_replicate] = func_value

    ########################################################################
    ######### Calculate the things to be plotted. ###
    ########################################################################
    local_mins = [[np.unique(func_values[i,j].round(decimals=2)) for j in range(10)] for i in range(10)]
    num_local_min = [[len(local_mins[i][j]) for j in range(10)] for i in range(10)]
    proportion_global = [[sum(func_values[i,j].round(decimals=2) == min(local_mins[i][j]))*1.0/100                          for j in range(10)] for i in range(10)]


    num_local_min = np.array(num_local_min)
    num_local_min = np.mean(num_local_min, axis = 1)

    proportion_global = np.array(proportion_global)
    proportion_global = np.mean(proportion_global, axis = 1)

    ########################################################################
    ######### Plots. #######################################################
    ########################################################################
    fig, axes = plt.subplots(figsize=(8,6), nrows=2, ncols=1)
    fig.tight_layout()
    plt.subplot(211)

    plt.plot(np.arange(0,1000,100), num_local_min)
    plt.title('Number of local minimum found by 100 gradient descents.')
    plt.xlabel('Object Location')
    plt.ylabel('Number')
    #plt.savefig('num_obj.png')
    # Proportion of gradient descents that find the local minimum of minimum value. 

    plt.subplot(212)
    plt.plot(np.arange(0,1000,100), proportion_global)
    plt.title('Proportion of GD that finds the global minimum among 100 gradient descents.')
    plt.xlabel('Object Location')
    plt.ylabel('Proportion')
    fig.tight_layout()
    plt.savefig('prop_obj.png')

    ########################################################################
    ######### Plots of contours. ###########################################
    ########################################################################
    np.random.seed(0) 
    # sensor_loc = np.random.randn(7,2) * 10
    x = np.arange(-10.0, 10.0, 0.1)
    y = np.arange(-10.0, 10.0, 0.1)
    X, Y = np.meshgrid(x, y) 
    obj_loc = [[0,0]]
    obj_loc, distance = generate_data_given_location(sensor_loc, 
                                                   obj_loc, k = num_sensors, d = 2)

    Z =  np.array([[log_likelihood((X[i,j],Y[i,j]), 
                                 sensor_loc, distance[0]) for j in range(len(X))] \
                 for i in range(len(X))]) 


    plt.figure(figsize=(10,4))
    plt.subplot(121)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('With object at (0,0)')
    #plt.show()

    np.random.seed(0) 
    # sensor_loc = np.random.randn(7,2) * 10
    x = np.arange(-400,400, 4)
    y = np.arange(-400,400, 4)
    X, Y = np.meshgrid(x, y) 
    obj_loc = [[200,200]]
    obj_loc, distance = generate_data_given_location(sensor_loc, 
                                                   obj_loc, k = num_sensors, d = 2)

    Z =  np.array([[log_likelihood((X[i,j],Y[i,j]), 
                                 sensor_loc, distance[0]) for j in range(len(X))] \
                 for i in range(len(X))]) 


    # Create a simple contour plot with labels using default colors.  The
    # inline argument to clabel will control whether the labels are draw
    # over the line segments of the contour, removing the lines beneath
    # the label
    #plt.figure()
    plt.subplot(122)
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('With object at (200,200)')
    #plt.show()
    plt.savefig('likelihood_landscape.png')


    ########################################################################
    ######### Plots of Found local minimas. ###########################################
    ########################################################################
    #sensor_loc
    #minimas = np.zeros((len(obj_locs), 10, num_gd_replicates, 2))
    #true_object_locs = np.zeros((len(obj_locs), 10, 2))
    object_loc_i = 5
    trail = 0

    plt.figure()
    plt.plot(sensor_loc[:, 0], sensor_loc[:, 1], 'r+', label="sensors")
    plt.plot(minimas[object_loc_i, trail, :, 0], minimas[object_loc_i, trail, :, 1], 'g.', label="minimas")
    plt.plot(true_object_locs[object_loc_i, trail, 0], true_object_locs[object_loc_i, trail, 1], 'b*', label="object")
    plt.title('object at location (%d, %d), gradient descent recovered locations' % (object_loc_i*100, object_loc_i*100))
    plt.legend()
    plt.savefig('2D_vis.png')
    
VAR_MEASUREMENT_NOISE = 1 # Used by generate_data_given_location function
run4(num_sensors=7)


# In[ ]:


VAR_MEASUREMENT_NOISE = 0.01 # Used by generate_data_given_location function
run4(num_sensors=7)


# In[ ]:


VAR_MEASUREMENT_NOISE = 1 # Used by generate_data_given_location function
run4(num_sensors=20)


# In[ ]:


########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################
def compute_grad_likelihood(sensor_loc, obj_loc, distance):
    """
    Compute the gradient of the loglikelihood function for part f.   

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
    # Your code: finish the grad loglike
    for i, sens in enumerate(sensor_loc):
        grad[i] = compute_gradient_of_likelihood(sens.reshape((1,-1)), obj_loc, distance[:, i])

    return grad

def find_mle_by_grad_descent(initial_sensor_loc, 
           obj_loc, distance, lr=0.001, num_iters = 1000):
    """
    Compute the gradient of the loglikelihood function for part f.   

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
    # Your code: finish the gradient descent
    for i in range(num_iters):
        grad = compute_grad_likelihood(sensor_loc, obj_loc, distance)
        sensor_loc += lr*grad

    return sensor_loc
########################################################################
#########  Gradient Computing and MLE ##################################
########################################################################

np.random.seed(0)
sensor_loc = generate_sensors()
obj_loc, distance = generate_data(sensor_loc, n = 100)
print('The real sensor locations are')
print(sensor_loc)
# Initialized as zeros.
initial_sensor_loc = np.zeros((7,2)) #np.random.randn(7,2)
estimated_sensor_loc = find_mle_by_grad_descent(initial_sensor_loc, 
            obj_loc, distance, lr=0.001, num_iters = 1000)
print('The predicted sensor locations are')
print(estimated_sensor_loc) 

 
########################################################################
#########  Estimate distance given estimated sensor locations. ######### 
########################################################################

def compute_distance_with_sensor_and_obj_loc(sensor_loc, obj_loc):
    """
    Estimate distance given estimated sensor locations.  

    Input:
    sensor_loc: k * d numpy array. 
    Location of the sensors.

    obj_loc: n * d numpy array. Location of the n objects.

    Output:
    distance: n * k dimensional numpy array. 
    """ 
    estimated_distance = scipy.spatial.distance.cdist(obj_loc, 
                                            sensor_loc, 
                                            metric='euclidean')
    return estimated_distance 
########################################################################
#########  MAIN  #######################################################
########################################################################    
np.random.seed(100)    
########################################################################
#########  Case 1. #####################################################
########################################################################

mse =0   
for i in range(100):
    obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, n = 1, original_dist = True)
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k = 7, d = 2)
    l = float('-inf')
    # Your code: compute the mse for this case
    estimated_dist = compute_distance_with_sensor_and_obj_loc(estimated_sensor_loc, obj_loc)
    mse += (1/100)*np.sum(np.square(distance - estimated_dist))
              
print('The MSE for Case 1 is {}'.format(mse))

########################################################################
#########  Case 2. #####################################################
########################################################################
mse =0
        
for i in range(100):
    obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, n = 1, original_dist = False)
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k = 7, d = 2)
    l = float('-inf')
    # Your code: compute the mse for this case
    estimated_dist = compute_distance_with_sensor_and_obj_loc(estimated_sensor_loc, obj_loc)
    mse += (1/100)*np.sum(np.square(distance - estimated_dist))

print('The MSE for Case 2 is {}'.format(mse)) 


########################################################################
#########  Case 3. #####################################################
########################################################################
mse =0
        
for i in range(100):
    obj_loc, distance = generate_data(sensor_loc, k = 7, d = 2, n = 1, original_dist = False)
    obj_loc, distance = generate_data_given_location(estimated_sensor_loc, obj_loc, k = 7, d = 2)
    l = float('-inf')
    # Your code: compute the mse for this case
    estimated_dist = compute_distance_with_sensor_and_obj_loc(estimated_sensor_loc, obj_loc)
    mse += (1/100)*np.sum(np.square(distance - estimated_dist))

print('The MSE for Case 2 (if we knew mu is [300,300]) is {}'.format(mse)) 

