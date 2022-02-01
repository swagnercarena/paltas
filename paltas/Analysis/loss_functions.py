# -*- coding: utf-8 -*-
"""
Construct the loss functions for use in NN training.

This module contains classes that can be used to initialize tensorflow loss
functions for NN training on the strong lensing problem.
"""

import tensorflow as tf
import numpy as np
import itertools


class BaseLoss():
	"""	A base class for the loss functions.

	Args:
		num_params (int): The number of parameters to predict.
		flip_pairs ([[int,...],...]): A list of lists. Each list contains
			the index of parameters that when flipped together return an
			equivalent lens model.
		weight_terms ([[int,...],...]): A list of lists. Each list contains
			the index of a parameter and the value by which you would like
			that parameters loss to be weighted.

	Notes:
		If multiple lists are provided, all possible combinations of
		flips will be considered. For example, if flip_pairs is [[0,1],[2,3]]
		then flipping 0,1,2,3 all at the same time will also be considered.
	"""

	def __init__(self, num_params, flip_pairs=None, weight_terms=None):
		# Save the flip pairs for later use
		self.flip_pairs = flip_pairs
		self.num_params = num_params
		self.weight_terms = weight_terms

		# Create a vector that will be used to rescale the loss
		self.weight_vector = np.ones(num_params)
		if weight_terms is not None:
			for wt in self.weight_terms:
				self.weight_vector[wt[0]] *= wt[1]
		self.weight_vector = tf.constant(self.weight_vector,dtype=tf.float32)

		# Now for each flip pair (including no flip) we will add a flip
		# matrix to our list.
		self.flip_mat_list = [tf.linalg.diag(tf.ones((self.num_params),
			dtype=tf.float32))]
		if self.flip_pairs is not None:
			# Create a flip matrix for every possible unique combination
			for flip_i in range(1,len(self.flip_pairs)+1):
				for comb in itertools.combinations(self.flip_pairs,flip_i):
					flip_list = list(itertools.chain.from_iterable(comb))
					const_initializer = np.ones(self.num_params)
					const_initializer[flip_list] = -1
					self.flip_mat_list.append(tf.linalg.diag(tf.constant(
						const_initializer,dtype=tf.float32)))

	def convert_output(self,output):
		"""	Convert the model outputs into the different components of the
		loss function

		Args:
			output (tf.Tensor): The predicted values of the lensing parameters

		Returns:
			([tf.Tensor,...]): The list of tf.Tensor objects that correspond to
			the components of the loss function.
		"""

		raise NotImplementedError('convert_output has not been defined.')

	def draw_samples(self,output,n_samps):
		"""	Draw samples from the predicted posterior defined by the networks
		outputs.

		Args:
			output (tf.Tensor): The predicted values of the lensing parameters
			n_samps (int): The number of samples to draw from each posterior.

		Returns:
			(np.array): An array with shape (n_samps,batch_size,n_params)
			containing draws from the posterior of each lens defined in
			the output.
		"""

		raise NotImplementedError('convert_output has not been defined.')


class MSELoss(BaseLoss):
	""" MSE loss that includes parameter flipping

	Args:
		num_params (int): The number of parameters to predict.
		flip_pairs ([[int,...],...]): A list of lists. Each list contains
			the index of parameters that when flipped together return an
			equivalent lens model.
		weight_terms ([[int,...],...]): A list of lists. Each list contains
			the index of a parameter and the value by which you would like
			that parameters loss to be weighted.

	Notes:
		If multiple lists are provided, all possible combinations of
		flips will be considered. For example, if flip_pairs is [[0,1],[2,3]]
		then flipping 0,1,2,3 all at the same time will also be considered.
	"""

	def convert_output(self,output):
		"""	Convert the model outputs into the different components of the
		loss function

		Args:
			output (tf.Tensor): The predicted values of the lensing parameters

		Returns:
			([tf.Tensor,...]): The list of tf.Tensor objects that correspond to
			the components of the loss function.
		"""
		# Get the mle value from the output.
		y_pred, _ = tf.split(output,num_or_size_splits=(self.num_params,-1),
			axis=-1)
		return y_pred

	def draw_samples(self,output,n_samps):
		"""	Draw samples from the predicted posterior defined by the networks
		outputs.

		Args:
			output (tf.Tensor): The predicted values of the lensing parameters
			n_samps (int): The number of samples to draw from each posterior.

		Returns:
			(np.array): An array with shape (n_samps,batch_size,n_params)
			containing draws from the posterior of each lens defined in
			the output.
		"""
		# The posterior is just a delta function.
		y_pred = self.convert_output(output).numpy()
		return np.repeat(np.expand_dims(y_pred,axis=0),n_samps,axis=0)

	def loss(self,y_true,output):
		""" Returns the MSE loss of the predicted parameters.

		Args:
			y_true (tf.Tensor): The true values of the parameters
			output (tf.Tensor): The predicted values of the lensing parameters

		Returns:
			(tf.Tensor): The loss function as a tf.Tensor.

		Notes:
			Can be used on networks that output more than just the maximum
			likelihood estimate, so long as the first num_params values
			outputted are the MLE.
		"""
		y_pred = self.convert_output(output)

		# Build a loss list and take the minimum value according to the
		# flip pairs.
		loss_list = []
		for flip_mat in self.flip_mat_list:
			loss_list.append(tf.reduce_mean(tf.square(
				tf.matmul(y_pred,flip_mat)-y_true)*self.weight_vector,axis=-1))
		loss_stack = tf.stack(loss_list,axis=-1)
		return tf.reduce_min(loss_stack,axis=-1)


class DiagonalCovarianceLoss(BaseLoss):
	""" Diagonal covariance loss that includes parameter flipping

	Args:
		num_params (int): The number of parameters to predict.
		flip_pairs ([[int,...],...]): A list of lists. Each list contains
			the index of parameters that when flipped together return an
			equivalent lens model.

		Notes:
			If multiple lists are provided, all possible combinations of
			flips will be considered. For example, if flip_pairs is
			[[0,1],[2,3]] then flipping 0,1,2,3 all at the same time will
			also be considered.
	"""

	def convert_output(self,output):
		"""	Convert the model outputs into the different components of the
		loss function

		Args:
			output (tf.Tensor): The predicted values of the lensing parameters

		Returns:
			([tf.Tensor,...]): The list of tf.Tensor objects that correspond to
			the components of the loss function.
		"""
		# Get both the mle value and the standard deviation.
		y_pred, log_var_pred = tf.split(output,num_or_size_splits=2,axis=-1)
		return y_pred, log_var_pred

	def draw_samples(self,output,n_samps):
		"""	Draw samples from the predicted posterior defined by the networks
		outputs.

		Args:
			output (tf.Tensor): The predicted values of the lensing parameters
			n_samps (int): The number of samples to draw from each posterior.

		Returns:
			(np.array): An array with shape (n_samps,batch_size,n_params)
			containing draws from the posterior of each lens defined in
			the output.
		"""
		# The posterior is the mean with random noise.
		y_pred, log_var_pred = self.convert_output(output)
		y_pred = y_pred.numpy()
		log_var_pred = log_var_pred.numpy()

		predict_samps = y_pred + np.random.randn(n_samps,y_pred.shape[0],
			y_pred.shape[1])*np.exp(log_var_pred/2)
		return predict_samps

	@staticmethod
	def log_gauss_diag(y_true,y_pred,log_var_pred):
		""" Return the negative log posterior of a Gaussian with diagonal
		covariance matrix

		Args:
			y_true (tf.Tensor): The true values of the parameters
			y_pred (tf.Tensor): The predicted value of the parameters
			log_var_pred (tf.Tensor): The predicted diagonal entries of the
				covariance. Note that log_var_pred is assumed to be the log of
				the covariance matrix values.

		Returns:
			(tf.Tensor): The TF graph for calculating the nlp

		Notes:
			This loss does not include the constant factor of 1/(2*pi)^(d/2).
		"""
		return 0.5*tf.reduce_sum(tf.multiply(tf.square(y_pred-y_true),
			tf.exp(-log_var_pred)),axis=-1) + 0.5*tf.reduce_sum(
			log_var_pred,axis=-1)

	def loss(self,y_true,output):
		""" Returns the loss of the predicted parameters.

		Args:
			y_true (tf.Tensor): The true values of the parameters
			output (tf.Tensor): The predicted values of the lensing
				parameters. This should include 2*self.num_params parameters
				to account for the diagonal entries of our covariance matrix.
				Covariance matrix values are assumed to be in log space.

		Returns:
			(tf.Tensor): The loss function as a tf.Tensor.
		"""
		y_pred, log_var_pred = self.convert_output(output)

		# Add each possible flip to the loss list. We will then take the
		# minimum.
		loss_list = []
		for flip_mat in self.flip_mat_list:
			loss_list.append(self.log_gauss_diag(y_true,
				tf.matmul(y_pred,flip_mat),log_var_pred))
		loss_stack = tf.stack(loss_list,axis=-1)
		return tf.reduce_min(loss_stack,axis=-1)


class FullCovarianceLoss(BaseLoss):
	""" Full covariance loss that includes parameter flipping

	Args:
		num_params (int): The number of parameters to predict.
		flip_pairs ([[int,...],...]): A list of lists. Each list contains
			the index of parameters that when flipped together return an
			equivalent lens model.

		Notes:
			If multiple lists are provided, all possible combinations of
			flips will be considered. For example, if flip_pairs is
			[[0,1],[2,3]] then flipping 0,1,2,3 all at the same time will
			also be considered.
	"""
	def __init__(self, num_params, flip_pairs=None, weight_terms=None):
		super().__init__(num_params,flip_pairs=flip_pairs,
			weight_terms=weight_terms)

		# Calculate the split list for lower traingular matrix
		self.split_list = []
		for i in range(1,num_params+1):
			self.split_list += [i]

	def convert_output(self,output):
		"""	Convert the model outputs into the different components of the
		loss function

		Args:
			output (tf.Tensor): The predicted values of the lensing parameters

		Returns:
			([tf.Tensor,...]): The list of tf.Tensor objects that correspond to
			the components of the loss function.
		"""
		# Start by dividing the output into the L_elements and the prediction
		# values.
		L_elements_len = int(self.num_params*(self.num_params+1)/2)
		y_pred, L_mat_elements = tf.split(output,
			num_or_size_splits=[self.num_params,L_elements_len],axis=-1)

		# Build the precision matrix and extract the diagonal part
		prec_mat, L_diag = self.construct_precision_matrix(L_mat_elements)

		# Return the mle, the precision matrix, and the diagonal values.
		return y_pred, prec_mat, L_diag

	def draw_samples(self,output,n_samps):
		"""	Draw samples from the predicted posterior defined by the networks
		outputs.

		Args:
			output (tf.Tensor): The predicted values of the lensing parameters
			n_samps (int): The number of samples to draw from each posterior.

		Returns:
			(np.array): An array with shape (n_samps,batch_size,n_params)
			containing draws from the posterior of each lens defined in
			the output.
		"""
		# The posterior is the mean with random noise.
		y_pred, prec_mat, _ = self.convert_output(output)
		y_pred = y_pred.numpy()
		prec_mat = prec_mat.numpy()
		cov_mat = np.linalg.inv(prec_mat)

		predict_samps = np.zeros((n_samps,)+y_pred.shape)

		for i in range(len(y_pred)):
			predict_samps[:,i,:] = np.random.multivariate_normal(y_pred[i],
				cov_mat[i],size=n_samps)
		return predict_samps

	def construct_precision_matrix(self,L_mat_elements):
		""" Take the matrix elements for the log cholesky decomposition and
		convert them to the precision matrix and return the value of
		the diagonal elements before exponentiation

		Args:
			L_mat_elements (tf.Tensor): A tensor of length
				num_params*(num_params+1)/2 that define the lower traingular
				matrix elements of the log cholesky decomposition

		Returns:
			((tf.Tensor,tf.Tensor)): Both the precision matrix and the diagonal
			elements (before exponentiation) of the log cholesky L matrix.
			Note that this second value is important for the posterior
			calculation.
		"""
		# First split the tensor into the elements that will populate each row
		cov_elements_split = tf.split(L_mat_elements,
			num_or_size_splits=self.split_list,axis=-1)
		# Before we stack these elements, we have to pad them with zeros
		# (corresponding to the 0s of the lower traingular matrix).
		cov_elements_stack = []
		pad_offset = 1
		for cov_element in cov_elements_split:
			# Use tf pad function since it's likely the fastest option.
			pad = tf.constant([[0,0],[0,self.num_params-pad_offset]])
			cov_elements_stack.append(tf.pad(cov_element,pad))
			pad_offset+=1
		# Stack the tensors to form our matrix. Use axis=-2 to avoid issues
		# with batches of matrices being passed in.
		L_mat = tf.stack(cov_elements_stack,axis=-2)
		# Pull out the diagonal part, and then (since we're using log
		# cholesky) exponentiate the diagonal.
		L_mat_diag = tf.linalg.diag_part(L_mat)
		L_mat = tf.linalg.set_diag(L_mat,tf.exp(L_mat_diag))
		# Calculate the actual precision matrix
		prec_mat = tf.matmul(L_mat,tf.transpose(L_mat,perm=[0,2,1]))

		return prec_mat, L_mat_diag

	@staticmethod
	def log_gauss_full(y_true,y_pred,prec_mat,L_diag):
		""" Return the negative log posterior of a Gaussian with full
		covariance matrix

		Args:
			y_true (tf.Tensor): The true values of the parameters
			y_pred (tf.Tensor): The predicted value of the parameters
			prec_mat: The precision matrix
			L_diag (tf.Tensor): The diagonal (non exponentiated) values of the
				log cholesky decomposition of the precision matrix

		Returns:
			(tf.Tensor): The TF graph for calculating the nlp

		Notes:
			This loss does not include the constant factor of 1/(2*pi)^(d/2).
		"""
		y_dif = y_true - y_pred
		return -tf.reduce_sum(L_diag,-1) + 0.5 * tf.reduce_sum(
			tf.multiply(y_dif,tf.reduce_sum(tf.multiply(tf.expand_dims(
				y_dif,-1),prec_mat),axis=-2)),-1)

	def loss(self,y_true,output):
		""" Returns the loss of the predicted parameters.

		Args:
			y_true (tf.Tensor): The true values of the parameters
			output (tf.Tensor): The predicted values of the lensing
				parameters. This should include 2*self.num_params parameters
				to account for the diagonal entries of our covariance matrix.
				Covariance matrix values are assumed to be in log space.

		Returns:
			(tf.Tensor): The loss function as a tf.Tensor.
		"""
		# Extract the outputs
		y_pred, prec_mat, L_diag = self.convert_output(output)

		# Add each possible flip to the loss list. We will then take the
		# minimum.
		loss_list = []
		for flip_mat in self.flip_mat_list:
			loss_list.append(self.log_gauss_full(y_true,
				tf.matmul(y_pred,flip_mat),prec_mat,L_diag))
		loss_stack = tf.stack(loss_list,axis=-1)
		return tf.reduce_min(loss_stack,axis=-1)
