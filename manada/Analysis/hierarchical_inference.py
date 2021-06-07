# -*- coding: utf-8 -*-
"""
Functions for conudcting hierarchical inference.

This module contains the tools to conduct hierarchical inference on our
network posteriors.
"""
import numpy as np


# The lens samples need to be et as a global variable for the pooling being
# done by emcee to be efficient.
lens_samps = None


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

	Parameters:
		hyperparameters (np.array): An array with the proposed hyperparameters
			for the population level lens parameter distribution
		eval_func_xi_omega (function): A callable function with inputs
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
	"""

	"""
