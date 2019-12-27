
# coding: utf-8

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Question 2 

# # Part A: one solution

# Assuming that I want to find the $w$ that minimizes $\frac{1}{2n}||Xw - y||_2^2$. In this part, X is full rank, and $y \in range(X)$

# In[25]:


X = np.random.normal(scale = 20, size=(100,10))
print(np.linalg.matrix_rank(X)) # confirm that  the matrix is full rank
# Theoretical optimal solution
w = np.random.normal(scale = 10, size = (10,1))
y = X.dot(w)


# In[85]:


def sgd(X, y, w_actual, threshold, max_iterations, step_size, gd=False):
    if isinstance(step_size, float):
        step_size_func = lambda i: step_size
        
    else:
        step_size_func = step_size
        
    # run 10 gradient descent at the same time, for averaging purpose
    # w_guesses stands for the current iterates (for each run)
    w_guesses = [np.zeros((X.shape[1], 1)) for _ in range(10)]
    n = X.shape[0]
    error = []
    it = 0
    above_threshold = True
    previous_w = np.array(w_guesses)
    
    while it < max_iterations and above_threshold:
        it += 1
        curr_error = 0
        for j in range(len(w_guesses)):
            if gd:
                # Your code, implement the gradient for GD
                sample_gradient =  2/n*(X.T.dot(X.dot(w_guesses[j])) - X.T.dot(y))

            else:
                # Your code, implement the gradient for SGD
                # Batch size of 1 
                index = np.random.randint(0,n)
                X_sgd = X[index].reshape((10,1))
                y_sgd = y[index]
                        
                sample_gradient = 2*X_sgd*(X_sgd.T.dot(w_guesses[j]) - y_sgd)
                #sample_gradient =  2 * (X_sgd.T @ previous_w[j] - y_sgd) * X_sgd 
            # Your code: implement the gradient update
            
            # learning rate at this step is given by step_size_func(it)            
            # w_guesses[j] = ?
            w_guesses[j] = previous_w[j] - step_size_func(it)*sample_gradient
            
            curr_error += np.linalg.norm(w_guesses[j]-w_actual)
        error.append(curr_error/10)
        
        diff = np.array(previous_w) - np.array(w_guesses)
        diff = np.mean(np.linalg.norm(diff, axis=1))
        above_threshold = (diff > threshold)
        previous_w = np.array(w_guesses)
    return w_guesses, error


# In[86]:


its = 5000
w_guesses, error = sgd(X, y, w, 1e-10, its, 0.0001)


# In[87]:


iterations = [i for i in range(len(error))]
#plt.semilogy(iterations, error, label = "Average error in w")
plt.semilogy(iterations, error, label = "Average error in w")
plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=False)
plt.title("Average Error vs Iterations for SGD with exact sol")
plt.legend()
plt.show()


# In[88]:


print("Required iterations: ", len(error))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses])
print("Final average error: ", average_error)


# # Part B: No solutions, constant step size

# In[89]:


y2 = y + np.random.normal(scale=5, size = y.shape)
w=np.linalg.inv(X.T @ X) @ X.T @ y2


# In[90]:


its = 5000
w_guesses2, error2 = sgd(X, y2, w, 1e-5, its, 0.0001)
w_guesses3, error3 = sgd(X, y2, w, 1e-5, its, 0.00001)
w_guesses4, error4 = sgd(X, y2, w, 1e-5, its, 0.000001)


# In[91]:


w_guess_gd, error_gd = sgd(X, y2, w, 1e-5, its, 0.001, True)


# In[92]:


plt.semilogy([i for i in range(len(error2))], error2, label="SGD, lr = 0.0001")
plt.semilogy([i for i in range(len(error3))], error3, label="SGD, lr = 0.00001")
plt.semilogy([i for i in range(len(error4))], error4, label="SGD, lr = 0.000001")
plt.semilogy([i for i in range(len(error_gd))], error_gd, label="GD, lr = 0.00001")
plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=False)
plt.title("Total Error vs Iterations for SGD without exact sol")
plt.legend()
plt.show()


# In[93]:


print("Required iterations, lr = 0.0001: ", len(error2))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses2])
print("Final average error: ", average_error)

print("Required iterations, lr = 0.00001: ", len(error3))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses3])
print("Final average error: ", average_error)

print("Required iterations, lr = 0.000001: ", len(error4))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses4])
print("Final average error: ", average_error)

print("Required iterations, GD: ", len(error_gd))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guess_gd])
print("Final average error: ", average_error)


# # Part C: No solutions, decreasing step size

# In[94]:


its = 5000
def step_size(step):
    if step < 500:
        return 1e-4 
    if step < 1500:
        return 1e-5
    if step < 3000:
        return 3e-6
    return 1e-6

w_guesses_variable, error_variable = sgd(X, y2, w, 1e-10, its, step_size, False)


# In[95]:


plt.semilogy([i for i in range(len(error_variable))], error_variable, label="Average error, decreasing lr")
plt.semilogy([i for i in range(len(error2))], error2, label="Average error, lr = 0.0001")
plt.semilogy([i for i in range(len(error3))], error3, label="Average error, lr = 0.00001")
plt.semilogy([i for i in range(len(error4))], error4, label="Average error, lr = 0.000001")

plt.xlabel("Iterations")
plt.ylabel("Norm of $w^t - w^*$",  usetex=False)
plt.title("Error vs Iterations for SGD with no exact sol")
plt.legend()
plt.show()


# In[96]:


print("Required iterations, variable lr: ", len(error_variable))
average_error = np.mean([np.linalg.norm(w-w_guess) for w_guess in w_guesses_variable])
print("Average error with decreasing lr:", average_error)


# # 2: Conclusions
# 
# ## Part A
# 
# SGD error is plotted against the number of iterations of SGD completed. SGD with this learning rate has stochastsic properties and reaches the threshold of accuracy before the maximum number of iterations is reached. 
# 
# 
# 
# ## Part B
# 
# 
# No exact solution. There is no solution so GD does not work. The SGD methods never get within the threshold, thus they go through all 5000 iterations and converge to minimal solutions which are different than the optimal solution. The sgd with the smallest learning rate never reaches a mimimal solution, it requires more iterations and takes the longest. Stochastic gradient descent with larger learning rates had higher variance, more stochastic plots. The learning rate was not large enough to leave the minimal solution which the other SGDs with smaller learning rates converge to. 
# 
# 
# 
# ## Part C
# 
# SGD for the plots with constant learning rates with no solutions behave similiarly as in part b. They all approach minmial points. In this case there is not a solution. 
# Average error with a decreasing learning rate works the best, it gets the smallest error through the iterations. It is able to break out of the non optimal mimial solutions that other SGD learning rates get stuck in. 
# 
# 
# 

# # Question 3 

# In[15]:


import time

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def optimize(x, y, pred, loss, optimizer, training_epochs, batch_size):
    acc = []
    with tf.Session() as sess:  # start training
        sess.run(tf.global_variables_initializer())  # Run the initializer
        for epoch in range(training_epochs):  # Training cycle
            avg_loss = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
                avg_loss += c / total_batch

            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy_ = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy = accuracy_.eval({x: mnist.test.images, y: mnist.test.labels})
            acc.append(accuracy)
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss),
                  "accuracy={:.9f}".format(accuracy))
    return acc


def train_linear(learning_rate=0.01, training_epochs=50, batch_size=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    pred = tf.matmul(x, W) + b
    loss = tf.reduce_mean((y - pred)**2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


# ## 3.a

# In[16]:


for batch_size in [50, 100, 200]:
    time_start = time.time()
    acc_linear = train_linear(batch_size=batch_size)
    print("train_linear finishes in %.3fs for batch size %s" % (time.time() - time_start, batch_size))

    plt.plot(acc_linear, label="linear bs=%d" % batch_size)
    plt.legend()


# 
#     
# - one epoch = one forward pass and one backward pass of all the training examples
# - batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
# - number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
# 
# https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
# 
# https://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent
# 
# 
# 
# A small batch size has more noise per step, but each training step takes less time. Smaller batch sizes results in the epoch taking longer time due to the noise. 
# 
# Larger batch size has less noise per step, the gradient direction fluctuates less compared to the small batch size model. The training steps lake longer, but its more accurate. The epochs will take less time due to less noise. 
# 
# 
# For a given accuracy/threshold smaller batch size may lead to a shorter total training time. 

# ## 3.b

# In[17]:


def softmax(z):
    z_ = tf.reduce_max(z)
    return tf.exp(z - z_) / tf.reduce_sum(tf.exp(z - z_))

def train_logistic(learning_rate=0.01, training_epochs=50, batch_size=100):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # YOUR CODE HERE
    pred = softmax(tf.matmul(x, W) + b)
    loss = -tf.reduce_mean(y*tf.log(pred))
    ################

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


# In[18]:


for batch_size in [50, 100, 200]:
    time_start = time.time()
    acc_logistic = train_logistic(batch_size=batch_size)
    print("train_logistic finishes in %.3fs for batch size %s" % (time.time() - time_start, batch_size))

    plt.plot(acc_logistic, label="logistic bs=%d" % batch_size)
    plt.legend()


# ## 3.c

# In[19]:


def train_nn(learning_rate=0.01, training_epochs=50, batch_size=50, n_hidden=64):
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    W1 = tf.Variable(tf.random_normal([784, n_hidden]))
    W2 = tf.Variable(tf.random_normal([n_hidden, 10]))
    b1 = tf.Variable(tf.random_normal([n_hidden]))
    b2 = tf.Variable(tf.random_normal([10]))

    # YOUR CODE HERE
    pred = softmax(tf.matmul(tf.tanh(tf.matmul(x, W1) + b1), W2) + b2)
    loss = -tf.reduce_mean(y*tf.log(pred))
    ################

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimize(x, y, pred, loss, optimizer, training_epochs, batch_size)


# In[20]:


for batch_size in [50, 100, 200]:
    time_start = time.time()
    acc_nn = train_nn(batch_size=batch_size)
    print("train_nn finishes in %.3fs for batch size %s" % (time.time() - time_start, batch_size))

    plt.plot(acc_nn, label="nonlinear bs=%d" % batch_size)
    plt.legend()


# ## 3.e

# In[21]:


import numpy as np

n_data = 6000
n_dim = 50

w_true = np.random.uniform(low=-2.0, high=2.0, size=[n_dim])

x_true = np.random.uniform(low=-10.0, high=10.0, size=[n_data, n_dim])
x_ob = x_true + np.random.randn(n_data, n_dim)
y_ob = x_true @ w_true + np.random.randn(n_data)

learning_rate = 0.0001
training_epochs = 100
batch_size = 6000


def main():
    x = tf.placeholder(tf.float32, [None, n_dim])
    y = tf.placeholder(tf.float32, [None, 1])

    w = tf.Variable(tf.random_normal([n_dim, 1]))

    # YOUR CODE HERE
    likelihood = -(n_data/2)*tf.log(tf.square(tf.norm(w)) + 1) - tf.square(tf.norm(y-tf.matmul(x, w))) / (2*(tf.square(tf.norm(w)) + 1))
    cost = -1.0 * likelihood
    ################

    # Adam is a fancier version of SGD, which is insensitive to the learning
    # rate.  Try replace this with GradientDescentOptimizer and tune the
    # parameters!
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        w_sgd = sess.run(w).flatten()

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_data / batch_size)
            for i in range(total_batch):
                start, end = i * batch_size, (i + 1) * batch_size
                _, c = sess.run(
                    [optimizer, cost],
                    feed_dict={
                        x: x_ob[start:end, :],
                        y: y_ob[start:end, np.newaxis]
                    })
                avg_cost += c / total_batch
            w_sgd = sess.run(w).flatten()
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost),
                  "|w-w_true|^2 = {:.9f}".format(np.sum((w_sgd - w_true)**2)))

    # Total least squares: SVD
    X = x_true
    y = y_ob
    stacked_mat = np.hstack((X, y[:, np.newaxis])).astype(np.float32)
    u, s, vh = np.linalg.svd(stacked_mat)
    w_tls = -vh[-1, :-1] / vh[-1, -1]

    error = np.sum(np.square(w_tls - w_true))
    print("TLS through SVD error: |w-w_true|^2 = {}".format(error))


if __name__ == "__main__":
    tf.set_random_seed(0)
    np.random.seed(0)
    main()


# The solutions are sensitive to the hyperparamaters. Regular gradient descent is the most costly but achieves the best results. Batch size and learning rate have the largest role. 

# ## Question 4

# PDF MERGED BELOW:
