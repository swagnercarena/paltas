# -*- coding: utf-8 -*-
"""
Functions for conudcting hierarchical inference.

This module contains the tools to conduct hierarchical inference on our
network posteriors.
"""
import numpy as np
from scipy import special


# The predicted samples need to be et as a global variable for the pooling to
# be efficient when done by emcee. This will have shape (num_params,num_samps,
# batch_size).
predict_samps_hier = None

# As with the predicted samples, the predicted mu and cov for the analytical
# calculations should also be set at the global level for optimal
# performance.
predict_an_mu = None
predict_an_cov = None


def log_p_xi_omega(predict_samps_hier,hyperparameters,eval_func_xi_omega):
	""" Calculate log p(xi|omega), the probability of the lens paramaters given
	the proposed lens parameter population level distribution.

	Args:
		predict_samps_hier (np.array): An array with dimensions (num_params,
			num_samps,batch_size) containing samples drawn from the
			predicted distribution.
		hyperparameters (np.array): An array with the proposed hyperparameters
			for the population level lens parameter distribution
		eval_func_xi_omega (function): A callable function with inputs
			(predict_samps_hier,hyperparameters) that returns an array of shape
			(n_samps,batch_size) containing the value of log p(xi|omega) for
			each sample.

	Returns:
		(np.array): A numpy array of the shape (n_samps,batch_size) containing
			the value of log p(xi|omega) for each sample.

	Notes:
		predict_samps_hier has shape (num_params,num_samps,batch_size) to make
		it easier to write fast evaluation functions using numba.
	"""

	# We mostly just need to call eval_func_xi_omega and deal with nans
	logpdf = eval_func_xi_omega(predict_samps_hier,hyperparameters)
	logpdf[np.isnan(logpdf)] = -np.inf

	return logpdf


def log_p_omega(hyperparameters,eval_func_omega):
	""" Calculate log p(omega) - the probability of the hyperparameters given
	the hyperprior.

	Args:
		hyperparameters (np.array): An array with the proposed hyperparameters
			for the population level lens parameter distribution
		eval_func_omega (function): A callable function with inputs
			(hyperparameters) that returns an float equal to the value of
			log p(omega)

	Returns:
		(float): The value of log p(omega)
	"""

	# We mostly need to check for nans.
	logpdf = eval_func_omega(hyperparameters)
	if np.sum(np.isnan(logpdf))>0:
		logpdf = -np.inf

	return logpdf


class ProbabilityClass:
	""" A class for the hierarchical inference probability calculations given
	samples from the NN posterior on the data.

	Args:
		eval_func_xi_omega_i (function): A callable function with input
			predict_samps_hier that returns an array of shape (n_samps,n_lenses)
			containing the value of log p(xi|omega_interim) for each sample.
			omega_int is the distribution of the training data.
		eval_func_xi_omega (function): A callable function with inputs
			(predict_samps_hier,hyperparameters) that returns an array of shape
			(n_samps,n_lenses) containing the value of log p(xi|omega) for
			each sample. omega is the proposed distribution of the test data.
		eval_func_omega (function): A callable function with inputs
			(hyperparameters) that returns a float equal to the value of
			log p(omega)
	Notes:
		predict_samps_hier has shape (num_params,num_samps,batch_size) to make
		it easier to write fast evaluation functions using numba.
	"""

	def __init__(self,eval_func_xi_omega_i,eval_func_xi_omega,
		eval_func_omega):
		# Save these functions to the class for later use.
		self.eval_func_xi_omega_i = eval_func_xi_omega_i
		self.eval_func_xi_omega = eval_func_xi_omega
		self.eval_func_omega = eval_func_omega

		self.samples_init = False

	def set_samples(self,predict_samps_input=None,predict_samps_hier_input=None):
		""" Set the global lens samples value. Using a global helps avoid data
		being pickled.

		Args:
			predict_samps_input (np.array): An array of shape (n_samps,n_lenses,
				n_params) that represents the network predictions and will be
				reshaped for hierarchical analysis
			predict_samps_hier_input (np.array): An array of shape (n_params,
				n_samps,n_lenses) that represents the network predictions.

		Notes:
			Both predict_samps_input and predict_samps_hier_input are allowed
			as inputs. The functions in posterior_functions use the predict_samps
			convention (n_samps,n_lenses,n_params) while the functions in this
			package use predict_samps_hier (n_params,n_samps,n_lenses)
			convention.
		"""
		# Set the global samples variable
		global predict_samps_hier
		if predict_samps_hier_input is not None:
			predict_samps_hier = predict_samps_hier_input
		elif predict_samps_input is not None:
			predict_samps_hier = np.ascontiguousarray(np.transpose(
				predict_samps_input,[2,0,1]))
		else:
			raise ValueError('Either predict_samps_input or ' +
				'predict_samps_hier_input must be specified')
		self.samples_init = True

		# Calculate the probability of the sample on the interim training
		# distribution
		self.p_samps_omega_i = self.eval_func_xi_omega_i(predict_samps_hier)

	def log_post_omega(self,hyperparameters):
		""" Given the predicted samples, calculate the log posterior of a
		specific distribution.

		Args:
			hyperparameters (np.array): An array with the proposed
				hyperparameters describing the population level lens parameter
				distribution omega.

		Returns:
			(float): The log posterior of omega given the predicted samples.

		Notes:
			Constant factors are ignored.
		"""

		if self.samples_init is False:
			raise RuntimeError('Must set samples or behaviour is ill-defined.')

		global predict_samps_hier

		# Start with the prior on omega
		lprior = log_p_omega(hyperparameters,self.eval_func_omega)

		# No need to evaluate the samples if the proposal is outside the prior.
		if lprior == -np.inf:
			return lprior

		# Calculate the probability of each datapoint given omega
		p_samps_omega = log_p_xi_omega(predict_samps_hier,hyperparameters,
			self.eval_func_xi_omega)

		# We can use our pre-calculated value of p_samps_omega_i.
		like_ratio = p_samps_omega - self.p_samps_omega_i
		like_ratio[np.isinf(self.p_samps_omega_i)] = -np.inf
		like_ratio = special.logsumexp(like_ratio,axis=0)
		like_ratio[np.isnan(like_ratio)] = -np.inf

		# Return the likelihood and the prior combined
		return lprior + np.sum(like_ratio)


class ProbabilityClassAnalytical:
	""" A class for the hierarchical inference probability calculations that
	works analytically for the case of Gaussian outputs, priors, and target
	distributions.

	Args:
		xi_omega_i_mu (np.array): An array with the length n_params
			specifying the mean of each parameters in the training
			distribution.
		xi_omega_i_cov (np.array): A n_params x n_params array
			specifying the covariance matrix for the training
			distribution.
		eval_func_omega (function): A callable function with inputs
			(hyperparameters) that returns a float equal to the value of
			log p(omega).
	"""
	def __init__(self,xi_omega_i_mu,xi_omega_i_cov,eval_func_omega):
		# Save each parameter to the class
		self.xi_omega_i_mu = xi_omega_i_mu
		self.xi_omega_i_cov = xi_omega_i_cov
		self.eval_func_omega = eval_func_omega

		# A flag to make sure the prediction values are set
		self.predictions_init

	def set_predictions(self,predict_an_mu_input,predict_an_cov_input):
		""" Set the global lens mean and covariance prediction values.

		Args:
			predict_an_mu_input (np.array): An array of shape (n_lenses,
				n_params) that represents the network predictions and will be
				reshaped for hierarchical analysis
			predict_an_cov_input (np.array): An array of shape (n_params,
				n_samps,n_lenses) that represents the network predictions.

		Notes:
			Both predict_samps_input and predict_samps_hier_input are allowed
			as inputs. The functions in posterior_functions use the predict_samps
			convention (n_samps,n_lenses,n_params) while the functions in this
			package use predict_samps_hier (n_params,n_samps,n_lenses)
			convention.
		"""
		# Call up the globals and set them.
		global predict_an_mu
		global predict_an_cov
		predict_an_mu = predict_an_mu_input
		predict_an_cov = predict_an_cov_input

		# Set the flag for the predictions being initialized
		self.predictions_init = True

	def log_post_omega(self,hyperparameters):
		""" Given the predicted means and covariances, calculate the log
		posterior of a specific distribution.

		Args:
			hyperparameters (np.array): An array with the proposed
				hyperparameters describing the population level lens parameter
				distribution omega. Should be length n_params*2, the first
				n_params being the mean and the second n_params being the log
				of the standard deviation for each parameter.

		Returns:
			(float): The log posterior of omega given the predicted samples.

		Notes:
			For now only supports diagonal covariance matrix.
		"""

		if self.predictions_init is False:
			raise RuntimeError('Must set predictions or behaviour is '
				+'ill-defined.')

		global predict_an_mu
		global predict_an_cov

		# Start with the prior on omega
		lprior = log_p_omega(hyperparameters,self.eval_func_omega)

		# No need to evaluate the samples if the proposal is outside the prior.
		if lprior == -np.inf:
			return lprior



		# We can use our pre-calculated value of p_samps_omega_i.
		like_ratio = p_samps_omega - self.p_samps_omega_i
		like_ratio[np.isinf(self.p_samps_omega_i)] = -np.inf
		like_ratio = special.logsumexp(like_ratio,axis=0)
		like_ratio[np.isnan(like_ratio)] = -np.inf

		# Return the likelihood and the prior combined
		return lprior + np.sum(like_ratio)
