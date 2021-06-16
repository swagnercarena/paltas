# -*- coding: utf-8 -*-
"""
Functions for conudcting hierarchical inference.

This module contains the tools to conduct hierarchical inference on our
network posteriors.
"""
import numpy as np
from scipy import special


# The predicted samples need to be et as a global variable for the pooling to
# be efficient when done by emcee.
predict_samps = None


def log_p_xi_omega(predict_samps,hyperparameters,eval_func_xi_omega):
	""" Calculate log p(xi|omega), the probability of the lens paramaters given
	the proposed lens parameter population level distribution.

	Args:
		predict_samps (np.array): An array with dimensions (n_samples,
			batch_size,num_params) containing samples drawn from the
			predicted distribution.
		hyperparameters (np.array): An array with the proposed hyperparameters
			for the population level lens parameter distribution
		eval_func_xi_omega (function): A callable function with inputs
			(predict_samps,hyperparameters) that returns an array of shape
			(n_samps,batch_size) containing the value of log p(xi|omega) for
			each sample.

	Returns:
		(np.array): A numpy array of the shape (n_samps,batch_size) containing
			the value of log p(xi|omega) for each sample.
	"""

	# We mostly just need to call eval_func_xi_omega and deal with nans
	logpdf = eval_func_xi_omega(predict_samps,hyperparameters)
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
	""" A companion class to HierarchicalClass that does all of the probability
	calculations. These functions are seperate from the main class to allow for
	pickling from emcee.

	Args:
		eval_func_xi_omega_i (function): A callable function with input
			predict_samps that returns an array of shape (n_samps,n_lenses)
			containing the value of log p(xi|omega_interim) for each sample.
			omega_int is the distribution of the training data.
		eval_func_xi_omega (function): A callable function with inputs
			(predict_samps,hyperparameters) that returns an array of shape
			(n_samps,n_lenses) containing the value of log p(xi|omega) for
			each sample. omega is the proposed distribution of the test data.
		eval_func_omega (function): A callable function with inputs
			(hyperparameters) that returns an float equal to the value of
			log p(omega)
	Notes:
		predict_samps has shape (n_params,n_samps,n_lenses) where n_samps is
		the number of samples drawn from the
	"""

	def __init__(self,eval_func_xi_omega_i,eval_func_xi_omega,
		eval_func_omega):
		# Save these functions to the class for later use.
		self.eval_func_xi_omega_i = eval_func_xi_omega_i
		self.eval_func_xi_omega = eval_func_xi_omega
		self.eval_func_omega = eval_func_omega

		self.samples_init = False

	def set_samples(self,predict_samps_input):
		""" Set the global lens samples value. Using a global helps avoid data
		being pickled.

		Args:
			predict_samps_input (np.array): A ()
		"""
		# Set the global samples variable
		global predict_samps
		predict_samps = predict_samps_input
		self.samples_init = True

		# Calculate the probability of the sample on the interim training
		# distribution
		self.p_samps_omega_i = self.eval_func_xi_omega_i(predict_samps)

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

		global predict_samps

		# Start with the prior on omega
		lprior = log_p_omega(hyperparameters,self.eval_func_omega)

		# No need to evaluate the samples if the proposal is outside the prior.
		if lprior == -np.inf:
			return lprior

		# Calculate the probability of each datapoint given omega
		p_samps_omega = log_p_xi_omega(predict_samps,self.eval_func_xi_omega,
			hyperparameters)

		# We can use our pre-calculated value of p_samps_omega_i.
		like_ratio = p_samps_omega - self.p_samps_omega_i
		like_ratio[np.isinf(self.pt_omega_i)] = -np.inf
		like_ratio = special.logsumexp(like_ratio,axis=0)
		like_ratio[np.isnan(like_ratio)] = -np.inf

		# Return the likelihood and the prior combined
		return lprior + np.sum(like_ratio)
