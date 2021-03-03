# -*- coding: utf-8 -*-
"""
Define a number of distributions that can be used by the sampler to draw
lensing parameters.

This module contains classes that define distributions that can be effecitvely
sampled from.
"""
import numpy as np
import numba


class MultivariateLogNormal():
	"""Class for drawing multiple parameters from a multivariate log normal
	distribution.

	Args:
		mean (np.array): The mean value of the log normal distribution
		covariance (np.array): The covariance matrix of the distribution

	Notes:
		Returned values will follow the distribution exp(normal(mu,covariance))
	"""

	def __init__(self,mean,covariance):
		# Make sure the dimension of the mean allows for easy matrix
		# operations
		if len(mean.shape)==1:
			mean = np.expand_dims(mean,axis=0)
		self.mean = mean
		self.covariance = covariance
		self.L = np.linalg.cholesky(self.covariance)

	def __call__(self,n=1):
		"""Returns n draws from the distribution

		Args:
			n (int): The number of draws.

		Returns
			(np.array): n x len(mean) array of draws.
		"""
		rand_draw = np.random.randn(self.mean.shape[1]*n).reshape(
			(self.mean.shape[1],n))
		return np.exp(self.mean.T+np.dot(self.L,rand_draw)).T


class TruncatedMultivariateNormal():
	"""Class for drawing multiple parameters from a truncated multivariate
	normal distribution.

	Args:
		mean (np.array): The mean value of the log normal distribution
		covariance (np.array): The covariance matrix of the distribution
		min_values (np.array): The minimum value for each parameter
		max_values (np.array): The maximum value for each parameter

	Notes:
		Note this code uses rejection sampling, which may be slow if most
		of the multivariate normal's pdf is outside the bounds. Be careful when
		using this in high dimensions.
	"""

	def __init__(self,mean,covariance,min_values,max_values):
		# Make sure that each of the n-dimensional inputs follows
		# the desired shape for matrix calculations
		if len(mean.shape)==1:
			mean = np.expand_dims(mean,axis=0)
		if len(min_values.shape)==1:
			min_values = np.expand_dims(min_values,axis=0)
		if len(max_values.shape)==1:
			max_values = np.expand_dims(max_values,axis=0)
		self.mean = mean
		self.covariance = covariance
		self.L = np.linalg.cholesky(self.covariance)
		self.min_values = min_values
		self.max_values = max_values

	def __call__(self,n=1):
		"""Returns n draws from the distribution

		Args:
			n (int): The number of draws.

		Returns
			(np.array): n x len(mean) array of draws.
		"""
		# Start with a regular draw
		n_accepted = 0
		n_samp = n
		keep_draws = np.zeros((n,self.mean.shape[1]))
		rand_draw = np.random.randn(self.mean.shape[1]*n_samp).reshape(
			(self.mean.shape[1],n_samp))
		draws = (self.mean.T+np.dot(self.L,rand_draw)).T

		# Check which draws are within our bounds
		keep_ind = np.prod(draws > self.min_values,axis=-1,dtype=np.bool)
		keep_ind *= np.prod(draws < self.max_values,axis=-1,dtype=np.bool)
		keep_draws[n_accepted:n_accepted+np.sum(keep_ind)] = draws[keep_ind]
		n_accepted += np.sum(keep_ind)

		# Keep drawing until we have enough samples.
		while n_accepted<n:
			# Draw
			rand_draw = np.random.randn(self.mean.shape[1]*n_samp).reshape(
				(self.mean.shape[1],n_samp))
			draws = (self.mean.T+np.dot(self.L,rand_draw)).T
			# Check for the values in bounds
			keep_ind = np.prod(draws > self.min_values,axis=-1,dtype=np.bool)
			keep_ind *= np.prod(draws < self.max_values,axis=-1,dtype=np.bool)
			# Only use the ones we need
			use_keep = np.minimum(n-n_accepted,np.sum(keep_ind))
			keep_draws[n_accepted:n_accepted+use_keep] = (
				draws[keep_ind][:use_keep])
			n_accepted += use_keep

		return keep_draws
