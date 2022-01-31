# -*- coding: utf-8 -*-
"""
Define a number of distributions that can be used by the sampler to draw
lensing parameters.

This module contains classes that define distributions that can be effecitvely
sampled from.
"""
import numpy as np
from scipy.stats import truncnorm


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

	def __init__(self,mean,covariance,min_values=None,max_values=None):
		# Make sure that each of the n-dimensional inputs follows
		# the desired shape for matrix calculations
		if len(mean.shape)==1:
			mean = np.expand_dims(mean,axis=0)
		# If none for min_values, set to negative infinity
		if min_values is None:
			min_values = np.ones(mean.shape) * -np.inf
		elif len(min_values.shape)==1:
			min_values = np.expand_dims(min_values,axis=0)
		# If none for max_values, set to positive infinity
		if max_values is None:
			max_values = np.ones(mean.shape) * np.inf
		elif len(max_values.shape)==1:
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

		# Remove first dimension for the n=1 case
		if n==1:
			keep_draws = np.squeeze(keep_draws)

		return keep_draws


class EllipticitiesTranslation():
	"""Class that takes in distributions for q_lens and phi_lens, returns 
	samples of e1 and e2 correspondingly
	
	Args: 
		q_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			axis ratio (can be callable or constant)
		phi_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			orientation angle in radians (can be callable or constant)

	Notes: 

	"""
	def __init__(self,q_dist,phi_dist):
		self.q_dist = q_dist
		self.phi_dist = phi_dist

	def __call__(self):
		"""Returns a sample of e1,e2 

		Returns:
			(float,float): samples of x-direction ellipticity 
				eccentricity, xy-direction ellipticity eccentricity
		"""

		if callable(self.q_dist):
			q = self.q_dist()
		else:
			q = self.q_dist
		if callable(self.phi_dist):
			phi = self.phi_dist()
		else:
			phi = self.phi_dist
		
		e1 = (1 - q)/(1+q) * np.cos(2*phi)
		e2 = (1 - q)/(1+q) * np.sin(2*phi)

		return e1,e2


class ExternalShearTranslation():
	"""Class that maps samples of gamma_ext, phi_ext distributions to 
		gamma1, gamma2
	
	Args: 
		gamma_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			external shear modulus (callable or constant) 
		phi_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			orientation angle in radians (callable or constant)
	
	Notes:
	"""

	def __init__(self, gamma_dist,phi_dist):
		self.gamma_dist = gamma_dist
		self.phi_dist = phi_dist
	
	def __call__(self):
		"""Returns gamma1, gamma2 samples

		Returns:
			(float,float): samples of external shear coordinate values 
		"""
		if callable(self.gamma_dist):
			gamma = self.gamma_dist()
		else:
			gamma = self.gamma_dist
		if callable(self.phi_dist):
			phi = self.phi_dist()
		else:
			phi = self.phi_dist
		
		gamma1 = gamma * np.cos(2*phi)
		gamma2 = gamma * np.sin(2*phi)
		
		return gamma1,gamma2


class KappaTransformDistribution():
	"""Class that samples Kext given 1 / (1-Kext) ~ n. n is sampled from a 
	distribution given by n_dist, then Kext is computed given the
	transformation
	
	Args: 
		n_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			1 / (1-Kext) (can be callable or constant)
	"""

	def __init__(self,n_dist):
		self.n_dist = n_dist

	def __call__(self):
		"""Samples 1/(1-Kext), then maps that sample to Kext value

		Returns: 
			(float): Kext sample
		"""
		if callable(self.n_dist):
			n = self.n_dist()
		else:
			n = self.n_dist
		
		return 1 - (1/n)


class Duplicate():
	"""Class that returns two copies of the same random draw.

	Args:
		dist (scipy.stats.rv_continuous.rvs or float): The distribution to
			draw the sample from.
	"""

	def __init__(self,dist):
		self.dist = dist

	def __call__(self):
		"""Returns two copies of the same sample

		Returns
			(float,float): Two copies of the sample.
		"""

		if callable(self.dist):
			samp = self.dist()
		else:
			samp = self.dist

		return samp,samp

class DuplicateXY():
	"""Class that returns two copies of x, y coordinates drawn from 
		distributions

	Args: 
		x_dist (scipy.stats.rv_continuous.rvs or float): distribution for x 
			(can be callable or constant)
		y_dist (scipy.stats.rv_continuous.rvs or float): distribution for y 
			(can be callable or constant)
	
	Notes:
	"""

	def __init__(self,x_dist,y_dist):
		self.x_dist = x_dist
		self.y_dist = y_dist
	
	def __call__(self):
		"""Returns two copies of x,y sample

		Returns
			(float,float,float,float): Two copies of x,y sampled from x_dist
				and y_dist
		"""

		if callable(self.x_dist):
			x = self.x_dist()
		else:
			x = self.x_dist
		if callable(self.y_dist):
			y = self.y_dist()
		else:
			y = self.y_dist
		
		return x,y,x,y


class RedshiftsTruncNorm():
	"""Class that samples z_lens and z_source from truncated normal 
		distributions, forcing z_source > z_lens to be true

	Args: 
		z_lens_min (float): minimum allowed lens redshift
		z_lens_mean (float): lens redshift mean
		z_lens_std (float): lens redshift standard deviation
		z_source_min (float): minimum allowed source redshift
		z_source_mean (float): source redshift mean
		z_source_std (float): source redshift standard deviation
	"""

	def __init__(self, z_lens_min,z_lens_mean,z_lens_std,z_source_min,
		z_source_mean,z_source_std):
		# transform z_lens_min, z_source_min to be in units of std. deviations
		self.z_lens_min = (z_lens_min - z_lens_mean) / z_lens_std 
		self.z_source_min = (z_source_min - z_source_mean) / z_source_std
		# define truncnorm dist for lens redshift
		self.z_lens_dist = truncnorm(self.z_lens_min,np.inf,loc=z_lens_mean,
			scale=z_lens_std).rvs
		# save z_source info
		self.z_source_mean = z_source_mean
		self.z_source_std = z_source_std
	
	def __call__(self):
		"""Returns samples of redshifts, ensuring z_source > z_lens

		Returns: 
			(float,float): z_lens,z_source 
		"""
		z_lens = self.z_lens_dist()
		clip = (z_lens - self.z_source_mean) / self.z_source_std
		# number of std. devs away to stop (negative)
		if(clip > self.z_source_min):
			self.z_source_min = clip
		# define truncnorm dist for source redshift
		z_source = truncnorm(self.z_source_min,np.inf,self.z_source_mean,
			self.z_source_std).rvs()

		return z_lens,z_source


class MultipleValues():
	"""Class to call dist.rvs(size=num)

	Args:
		dist (scipy.stats.rv_continuous.rvs): callable distribution
		num (int): number of samples to return in one call
	"""

	def __init__(self, dist, num):
		self.dist = dist
		self.num = num
	
	def __call__(self):
		"""Returns specified # of samples from dist
		
		Returns: 
			list(float): |num| samples from dist
		"""
		return self.dist(size=self.num)
