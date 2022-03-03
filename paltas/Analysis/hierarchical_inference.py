# -*- coding: utf-8 -*-
"""
Conduct hierarchical inference on a population of lenses.

This module contains the tools to conduct hierarchical inference on our
network posteriors.
"""
import numpy as np
from scipy import special
import numba


# The predicted samples need to be et as a global variable for the pooling to
# be efficient when done by emcee. This will have shape (num_params,num_samps,
# batch_size).
predict_samps_hier = None

# As with the predicted samples, the predicted mu and cov for the analytical
# calculations should also be set at the global level for optimal
# performance.
mu_pred_array = None
prec_pred_array = None
mu_pred_array_ensemble = None
prec_pred_array_ensemble = None


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


@numba.njit
def gaussian_product_analytical(mu_pred,prec_pred,mu_omega_i,prec_omega_i,
	mu_omega,prec_omega):  # pragma: no cover
	""" Calculate the log of the integral of p(xi_k|omega)*p(xi_k|d_k,omega_int)/
	p(xi_k|omega_int) when all three pdfs are Gaussian.

	Args:
		mu_pred (np.array): The mean output by the network
		prec_pred (np.array): The precision matrix output by the network
		mu_omega_i (np.array): The mean output of the interim prior
		prec_omega_i (np.array): The precision matrix of the interim prior
		mu_omega (np.array): The mean of the proposed hyperparameter
			posterior.
		prec_omega (np.array): The precision matrix of the proposed
			hyperparameter posterior.

	Returns:
		(float): The lof of the product of the three Gaussian integrated over
		all space.
	Notes:
		The equation used here breaks down when the combination of precision
		matrices does not yield a valid precision matrix. When this happen, the
		output will be -np.inf.
	"""
	# This implements the final formula derived in the appendix of
	# Wagner-Carena et al. 2021.

	# Calculate the values of eta and the combined precision matrix
	prec_comb = prec_pred+prec_omega-prec_omega_i

	# This is not guaranteed to return a valid precision matrix. When it
	# doesn't the analytical equation used here is wrong. In those cases
	# return -np.inf
	eigvals = np.linalg.eigvals(prec_comb.astype(np.complex128))
	if (np.any(np.real(eigvals)<=0) or np.any(np.imag(eigvals)!=0)):
		return -np.inf

	cov_comb = np.linalg.inv(prec_comb)
	eta_pred = np.dot(prec_pred,mu_pred)
	eta_omega_i = np.dot(prec_omega_i,mu_omega_i)
	eta_omega = np.dot(prec_omega,mu_omega)
	eta_comb = eta_pred + eta_omega - eta_omega_i

	# Now calculate each of the terms in our exponent
	exponent = 0
	exponent -= np.log(abs(np.linalg.det(prec_pred)))
	exponent -= np.log(abs(np.linalg.det(prec_omega)))
	exponent += np.log(abs(np.linalg.det(prec_omega_i)))
	exponent += np.log(abs(np.linalg.det(prec_comb)))
	exponent += np.dot(mu_pred.T,np.dot(prec_pred,mu_pred))
	exponent += np.dot(mu_omega.T,np.dot(prec_omega,mu_omega))
	exponent -= np.dot(mu_omega_i.T,np.dot(prec_omega_i,mu_omega_i))
	exponent -= np.dot(eta_comb.T,np.dot(cov_comb,eta_comb))

	return -0.5*exponent


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
		mu_omega_i (np.array): An array with the length n_params
			specifying the mean of each parameters in the training
			distribution.
		cov_omega_i (np.array): A n_params x n_params array
			specifying the covariance matrix for the training
			distribution.
		eval_func_omega (function): A callable function with inputs
			(hyperparameters) that returns a float equal to the value of
			log p(omega).
	"""
	def __init__(self,mu_omega_i,cov_omega_i,eval_func_omega):
		# Save each parameter to the class
		self.mu_omega_i = mu_omega_i
		self.cov_omega_i = cov_omega_i
		# Store the precision matrix for later use.
		self.prec_omega_i = np.linalg.inv(cov_omega_i)
		self.eval_func_omega = eval_func_omega

		# A flag to make sure the prediction values are set
		self.predictions_init = False

	def set_predictions(self,mu_pred_array_input,prec_pred_array_input):
		""" Set the global lens mean and covariance prediction values.

		Args:
			mu_pred_array_input (np.array): An array of shape (n_lenses,
				n_params) that represents the mean network prediction on each
				lens.
			prec_pred_array_input (np.array): An array of shape (n_lenses,
				n_params,n_params) that represents the predicted precision
				matrix on each lens.
		"""
		# Call up the globals and set them.
		global mu_pred_array
		global prec_pred_array
		mu_pred_array = mu_pred_array_input
		prec_pred_array = prec_pred_array_input

		# Set the flag for the predictions being initialized
		self.predictions_init = True

	@staticmethod
	@numba.njit
	def log_integral_product(mu_pred_array,prec_pred_array,mu_omega_i,
		prec_omega_i,mu_omega,prec_omega):  # pragma: no cover
		""" For the case of Gaussian distributions, calculate the log of the
		integral p(xi_k|omega)*p(xi_k|d_k,omega_int)/p(xi_k|omega_int) summed
		over all of the lenses in the sample.

		Args:
			mu_pred_array (np.array): An array of the mean output by the
				network for each lens
			prec_pred_array (np.array): An array of the precision matrix output
				by the network for each lens.
			prec_pred (np.array): The precision matrix output by the network
			mu_omega_i (np.array): The mean output of the interim prior
			prec_omega_i (np.array): The precision matrix of the interim prior
			mu_omega (np.array): The mean of the proposed hyperparameter
				posterior.
			prec_omega (np.array): The precision matrix of the proposed
				hyperparameter posterior.
		"""
		# In log space, the product over lenses in the posterior becomes a sum
		integral = 0
		for pi in range(len(mu_pred_array)):
			mu_pred = mu_pred_array[pi]
			prec_pred = prec_pred_array[pi]
			integral += gaussian_product_analytical(mu_pred,prec_pred,
				mu_omega_i,prec_omega_i,mu_omega,prec_omega)
		# Treat nan as probability 0.
		if np.isnan(integral):
			integral = -np.inf
		return integral

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

		global mu_pred_array
		global prec_pred_array

		# Start with the prior on omega
		lprior = log_p_omega(hyperparameters,self.eval_func_omega)

		# No need to evaluate the samples if the proposal is outside the prior.
		if lprior == -np.inf:
			return lprior

		# Extract mu_omega and prec_omega from the provided hyperparameters
		mu_omega = hyperparameters[:len(hyperparameters)//2]
		cov_omega = np.diag(np.exp(hyperparameters[len(hyperparameters)//2:]*2))
		prec_omega = np.linalg.inv(cov_omega)

		like_ratio = self.log_integral_product(mu_pred_array,prec_pred_array,
			self.mu_omega_i,self.prec_omega_i,mu_omega,prec_omega)

		# Return the likelihood and the prior combined
		return lprior + like_ratio


class ProbabilityClassEnsemble(ProbabilityClassAnalytical):
	""" An extension of the class ProbabilityClassAnalytical that allows for
	hierarchical inference with ensemble predictions.

	Args:
		mu_omega_i (np.array): An array with the length n_params
			specifying the mean of each parameters in the training
			distribution.
		cov_omega_i (np.array): A n_params x n_params array
			specifying the covariance matrix for the training
			distribution.
		eval_func_omega (function): A callable function with inputs
			(hyperparameters) that returns a float equal to the value of
			log p(omega).
	"""

	def set_predictions(self,mu_pred_array_input,prec_pred_array_input):
		""" Set the global lens mean and covariance prediction values.

		Args:
			mu_pred_array_input (np.array): An array of shape (n_ensembles,
				n_lenses,n_params) that represents the mean network prediction
				on each lens.
			prec_pred_array_input (np.array): An array of shape (n_ensembles,
				n_lenses,n_params,n_params) that represents the predicted
				precision matrix on each lens.
		"""
		# Call up the globals and set them.
		global mu_pred_array_ensemble
		global prec_pred_array_ensemble
		mu_pred_array_ensemble = mu_pred_array_input
		prec_pred_array_ensemble = prec_pred_array_input

		# Set the flag for the predictions being initialized
		self.predictions_init = True
		self.n_ensemble = len(mu_pred_array_ensemble)

	@staticmethod
	@numba.njit
	def log_integral_product(mu_pred_array,prec_pred_array,mu_omega_i,
		prec_omega_i,mu_omega,prec_omega):  # pragma: no cover
		""" For the case of Gaussian distributions, calculate the log of the
		integral p(xi_k|omega)*p(xi_k|d_k,omega_int)/p(xi_k|omega_int) summed
		over all of the lenses in the sample.

		Args:
			mu_pred_array (np.array): An array of the mean output by the
				network for each lens
			prec_pred_array (np.array): An array of the precision matrix output
				by the network for each lens.
			prec_pred (np.array): The precision matrix output by the network
			mu_omega_i (np.array): The mean output of the interim prior
			prec_omega_i (np.array): The precision matrix of the interim prior
			mu_omega (np.array): The mean of the proposed hyperparameter
				posterior.
			prec_omega (np.array): The precision matrix of the proposed
				hyperparameter posterior.
		"""
		# In log space, the product over lenses in the posterior becomes a sum
		integral = 0
		n_ensemble = len(mu_pred_array)
		for pi in range(mu_pred_array.shape[1]):
			# For each lens, the integral for each ensemble must be summed
			# together.
			ensemble_integral = -np.inf
			for ei in range(mu_pred_array.shape[0]):
				mu_pred = mu_pred_array[ei,pi]
				prec_pred = prec_pred_array[ei,pi]
				ensemble_integral = np.logaddexp(ensemble_integral,
					gaussian_product_analytical(mu_pred,prec_pred,mu_omega_i,
						prec_omega_i,mu_omega,prec_omega))
			# Divide by 1/N_ensemble for each prediction.
			integral += ensemble_integral - np.log(n_ensemble)
		# Treat nan as probability 0.
		if np.isnan(integral):
			integral = -np.inf
		return integral

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

		global mu_pred_array_ensemble
		global prec_pred_array_ensemble

		# Extract mu_omega and prec_omega from the provided hyperparameters
		mu_omega = hyperparameters[:len(hyperparameters)//2]
		cov_omega = np.diag(np.exp(hyperparameters[len(hyperparameters)//2:]*2))
		prec_omega = np.linalg.inv(cov_omega)

		# Start with the prior on omega
		lprior = log_p_omega(hyperparameters,self.eval_func_omega)

		# No need to evaluate the samples if the proposal is outside the prior.
		if lprior == -np.inf:
			return lprior

		like_ratio = self.log_integral_product(mu_pred_array_ensemble,
			prec_pred_array_ensemble,self.mu_omega_i,self.prec_omega_i,mu_omega,
			prec_omega)

		# Return the likelihood and the prior combined
		return lprior + like_ratio
