
# coding: utf-8

# In[25]:


import os
import numpy as np
import cv2
import copy
import glob

import sys

from numpy.random import uniform

import pickle
from scipy.linalg import eig
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import svd
import numpy.linalg as LA
import matplotlib.pyplot as plt
import IPython
from sklearn.preprocessing import StandardScaler

def standardized(v):
	return (v/255.0) * 2.0 - 1.0

def flatten_and_standardize(data):
	result = []
	for d in data:
		d = d.flatten()
		d = standardized(d)
		result.append(d)
	return result

class Mooney(object):

	def __init__(self):
		self.lmbda = 1e-5

	def load_data(self):
		self.x_train = pickle.load(open('x_train.p','rb'))
		self.y_train = pickle.load(open('y_train.p','rb'))
		self.x_test = pickle.load(open('x_test.p','rb'))
		self.y_test = pickle.load(open('y_test.p','rb'))

	def compute_covariance_matrices(self):
		# USE STANDARD SCALAR TO DO MEAN SUBTRACTION
		ss_x = StandardScaler(with_std = False)
		ss_y = StandardScaler(with_std = False)

		num_data = len(self.x_train)

		x = self.x_train[0]
		y = self.y_train[0]

		x_f = x.flatten()
		y_f = y.flatten()

		x_f_dim = x_f.shape[0]
		y_f_dim = y_f.shape[0]

		self.x_dim = x_f_dim
		self.y_dim = y_f_dim

		self.C_xx = np.zeros([x_f_dim,x_f_dim])
		self.C_yy = np.zeros([y_f_dim,y_f_dim])
		self.C_xy = np.zeros([x_f_dim,y_f_dim])

		x_data = []
		y_data = []

		for i in range(num_data):
			x_image = self.x_train[i]
			y_image = self.y_train[i]

			# FLATTEN DATA
			x_f = x_image.flatten()
			y_f = y_image.flatten()

			# STANDARDIZE DATA
			x_f = standardized(x_f)
			y_f = standardized(y_f)

			x_data.append(x_f)
			y_data.append(y_f)

		# SUBTRACT MEAN
		ss_x.fit(x_data)
		x_data = ss_x.transform(x_data)

		ss_y.fit(y_data)
		y_data = ss_y.transform(y_data)

		for i in range(num_data):
			x_f = np.array([x_data[i]])
			y_f = np.array([y_data[i]])
			# TODO: COMPUTE COVARIANCE MATRICES
			self.C_xx += np.dot(x_f.T, x_f)
			self.C_yy += np.dot(y_f.T, y_f)
			self.C_xy += np.dot(x_f.T, y_f)

		# DIVIDE BY THE NUMBER OF DATA POINTS
		self.C_xx = 1.0/float(num_data)*self.C_xx
		self.C_yy = 1.0/float(num_data)*self.C_yy
		self.C_xy = 1.0/float(num_data)*self.C_xy

	def compute_projected_data_matrix(self,X_proj):

		Y = []
		X = []

		Y_test = []
		X_test = []

		# LOAD TRAINING DATA
		for x in self.x_train:
			x_f = np.array([x.flatten()])
			# STANDARDIZE DATA
			x_f = standardized(x_f)

			# TODO: PROJECT DATA
			x_p = np.matmul(X_proj,x_f.T)

			X.append(x_p)

		Y = flatten_and_standardize(self.y_train)

		for x in self.x_test:
			x_f = np.array([x.flatten()])
			# STANDARDIZE DATA
			x_f = standardized(x_f)
			# TODO: PROJECT DATA
			x_p = np.matmul(X_proj,x_f.T)

			X_test.append(x_p)

		Y_test = flatten_and_standardize(self.y_test)

		# CONVERT TO MATRIX
		self.X_ridge = np.vstack(X)
		self.Y_ridge = np.vstack(Y)

		self.X_test_ridge = np.vstack(X_test)
		self.Y_test_ridge = np.vstack(Y_test)

	def compute_data_matrix(self):
		X = flatten_and_standardize(self.x_train)
		Y = flatten_and_standardize(self.y_train)
		X_test = flatten_and_standardize(self.x_test)
		Y_test = flatten_and_standardize(self.y_test)

		# CONVERT TO MATRIX
		self.X_ridge = np.vstack(X)
		self.Y_ridge = np.vstack(Y)

		self.X_test_ridge = np.vstack(X_test)
		self.Y_test_ridge = np.vstack(Y_test)


	def solve_for_variance(self):
		eigen_values = np.zeros((675,))
		eigen_vectors = np.zeros((675, 675))
		lambda_ = 0.00001
		# TODO: COMPUTE CORRELATION MATRIX ??
		X = inv(sqrtm(self.C_xx + lambda_ * np.eye(self.x_dim)))
		Y = inv(sqrtm(self.C_yy + lambda_ * np.eye(self.y_dim)))

		Z = np.matmul(X, np.matmul(self.C_xy, Y))

		eigen_vectors, eigen_values, w = svd(Z)

		plt.plot(eigen_values)
		plt.xlabel("Direction")
		plt.ylabel("Singular Vals")
		plt.show()

		return eigen_values, eigen_vectors

	def project_data(self, eig_val,eig_vec,proj=150):
		# TODO: COMPUTE PROJECTION SINGULAR VECTORS
		X_proj = eig_vec[: , 0:proj]
		return X_proj.T

	def ridge_regression(self):
		w_ridge = []
		X_G = np.matmul(self.X_ridge.T, self.X_ridge)
		lambda_ = 0.00001
		A = X_G+lambda_*np.eye(X_G.shape[0])
		p_1 = LA.inv(A)
		for i in range(self.y_dim):
			# TODO: IMPLEMENT RIDGE REGRESSION ???
			print(self.Y_ridge.shape)
			print()
			print(self.X_ridge.shape)
			p_2 = np.matmul(self.X_ridge[:,i], self.Y_ridge)
			ridge = np.matmul(p_1, p_2)
			w_ridge.append(ridge)

		self.w_ridge = np.vstack(w_ridge)


	def plot_image(self, vector):
		vector = ((vector+1.0)/2.0)*255.0
		vector = np.reshape(vector,(15,15,3))
		p = vector.astype("uint8")
		p = cv2.resize(p,(100,100))
		count = 0

		cv2.imwrite('a_face_'+str(count)+'.png',p)

	def measure_error(self, X_ridge, Y_ridge):
		prediction = np.matmul(self.w_ridge,X_ridge.T)
		evaluation = Y_ridge.T - prediction

		print(evaluation)

		dim,num_data = evaluation.shape

		error = []

		for i in range(num_data):
			# COMPUTE L2 NORM for each vector then square
			error.append(LA.norm(evaluation[:,i])**2)

		# Return average error
		return np.mean(error)


	def draw_images(self):
		for count, x in enumerate(self.X_test_ridge):
			prediction = np.matmul(self.w_ridge,x)
			prediction = ((prediction+1.0)/2.0)*255.0
			prediction = np.reshape(prediction,(15,15,3))
			p = prediction.astype("uint8")
			p = cv2.resize(p,(100,100))
			cv2.imwrite('face_'+str(count)+'.png',p)

		for count, x in enumerate(self.x_test):
			x = x.astype("uint8")
			x = cv2.resize(x,(100,100))
			cv2.imwrite('og_face_'+str(count)+'.png',x)

		for count, x in enumerate(self.y_test):
			x = x.astype("uint8")
			x = cv2.resize(x,(100,100))
			cv2.imwrite('gt_face_'+str(count)+'.png',x)
            
            
            
def plot_image(self,vector):
    vector = ((vector+1.0)/2.0)*255.0
    vector = np.reshape(vector,(15,15,3))
    p = vector.astype("uint8")
    p = cv2.resize(p,(100,100))
    count = 0
    cv2.imwrite('singular_face.png',p)

if __name__ == '__main__':

	mooney = Mooney()

	mooney.load_data()
	mooney.compute_covariance_matrices()
	eig_val, eig_vec = mooney.solve_for_variance()

	proj = [0,50,100,150,200,250,300,350,400,450,500,650]
	error_test = []
	for p in proj:

		X_proj = mooney.project_data(eig_val,eig_vec,proj=p)

		# COMPUTE REGRESSION
		mooney.compute_projected_data_matrix(X_proj)
		mooney.ridge_regression()
		training_error = mooney.measure_error(mooney.X_ridge, mooney.Y_ridge)
		test_error = mooney.measure_error(mooney.X_test_ridge, mooney.Y_test_ridge)
		#mooney.draw_images()
		mooney.plot_image(p)
		error_test.append(test_error)

	plt.plot(proj,error_test)
	plt.show()

	# COMPUTE REGRESSION NO PROJECT
	mooney.compute_data_matrix()
	mooney.ridge_regression()
	mooney.draw_images()
	training_error = mooney.measure_error(mooney.X_ridge, mooney.Y_ridge)
	test_error = mooney.measure_error(mooney.X_test_ridge, mooney.Y_test_ridge)


# #3.a
# 
# 
# The covariance matrices for mean centered data sets are given by 
# 
# $Cov(X,Y) = \frac{1}{n}\sum_{i=1}^{n}x_i\cdot y_i^{T} = \frac{1}{n}\cdot X^{T}Y$
# 
# $Cov(X,X) = \frac{1}{n}\sum_{i=1}^{n}x_i\cdot x_i^{T} = \frac{1}{n}\cdot X^{T}X$
# 
# 
# $Cov(Y,Y) = \frac{1}{n}\sum_{i=1}^{n}y_i\cdot y_i^{T} = \frac{1}{n}\cdot Y^{T}Y$
