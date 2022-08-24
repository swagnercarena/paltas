from pathlib import Path
import warnings

import numba
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

import paltas
import paltas.Analysis

# Mappings from short to long parameter names and back
mdef = 'main_deflector_parameters_'
short_names = dict((
	(mdef + 'theta_E', 'theta_E'),
	('subhalo_parameters_sigma_sub', 'sigma_sub'),
	('subhalo_parameters_shmf_plaw_index', 'shmf_plaw_index'),
	('los_parameters_delta_los', 'delta_los'),
	(mdef + 'center_x', 'center_x'),
	(mdef + 'center_y', 'center_y'),
	(mdef + 'gamma', 'gamma'),
	(mdef + 'gamma1', 'gamma1'),
	(mdef + 'gamma2', 'gamma2'),
	(mdef + 'e1', 'e1'),
	(mdef + 'e2', 'e2')))
long_names = {v: k for k, v in short_names.items()}

# Parameter order used in the March 2022 paper
MARCH_2022_PARAMETERS = (
	'main_deflector_parameters_theta_E',
	'main_deflector_parameters_gamma1',
	'main_deflector_parameters_gamma2',
	'main_deflector_parameters_gamma',
	'main_deflector_parameters_e1',
	'main_deflector_parameters_e2',
	'main_deflector_parameters_center_x',
	'main_deflector_parameters_center_y',
	'subhalo_parameters_sigma_sub')

DEFAULT_PARAMETERS = tuple([
	long_names[p]
	for p in ('theta_E', 'sigma_sub', 'gamma')])
DEFAULT_TRAINING_SET = (
	Path(paltas.__path__[0]) / 'Configs' / 'paper_2203_00690' / 'config_train.py')


class GaussianInference:

	def __init__(
			self,
			y_pred,
			cov_pred,
			test_config_path,
			log_sigma=False,
			select_parameters=DEFAULT_PARAMETERS,
			all_parameters=MARCH_2022_PARAMETERS,
			train_config_path=DEFAULT_TRAINING_SET):
		"""Infer means and (uncorrelated) standard deviations / sigmas
			of a lens population.

		Required arguments:
		 - y_pred: (n_images, n_params) array with predicted means
		 - cov_pred: (n_images, n_params, n_params) array with predicted
			covariances
		 - test_config_path: path to a configuration .py from which to pull
			true/guess values

		Optional arguments:
		 - log_sigma: if True, hyperprior will be uniform in Log[sigma]'s.
			Otherwise in hyperprior will be uniform in sigma's.
		 - select_parameters: sequence of strings, parameter names to do
			inference for. Others will be ignored.
		 - all_parameters: sequence of strings, parameter names ordered
			as in y_pred.
		 - train_config_path: path to configuration .py for the training set
			(interim prior)
		"""
		# Get indices of parameters to select
		select_is = [all_parameters.index(p) for p in select_parameters]
		# Apply the parameter selection to the vectors/matrices we need
		params = np.asarray(all_parameters)[select_is].tolist()
		y_pred = y_pred[:,select_is]
		cov_pred = select_from_matrix_stack(cov_pred, select_is)

		# Recover precision matrices from covariance matrices
		prec_pred = np.linalg.inv(cov_pred)

		n_params = len(params)

		# mean/cov of the training data (interim prior)
		mu_omega_i, cov_omega_i = extract_mu_cov(train_config_path, params)

		# True mean/cov of the test data.
		mu_omega, cov_omega = extract_mu_cov(test_config_path, params)
		true_hyperparameters = np.concatenate([
			mu_omega,
			np.diag(np.sqrt(cov_omega))])
		if log_sigma:
			true_hyperparameters[n_params:] = np.log(
				true_hyperparameters[n_params:])

		# A uniform hyperprior, mainly to force some parameters positive.
		# (For MAP/MLE even this is not needed.)
		positive_is = np.asarray(
			[params.index(long_names[p])
			 for p in ['theta_E', 'sigma_sub', 'gamma']])

		# jit because this will be called from inside jitted functions
		@numba.njit
		def log_hyperprior(hyperparameters):
			if hyperparameters[positive_is].min() < 0:
				return -np.inf
			# The hyperparameters we get are always log(sigma),
			# since paltas always uses log(sigma).
			log_sigmas = hyperparameters[n_params:]
			if log_sigmas.min() < -15: # or (sigmas.max() > 0):
				return -np.inf
			return 0

		# Initialize the  posterior and give it the network predictions.
		# TODO: can we still multiprocess.Pool now that these are local vars?
		# Do we care?
		prob_class = (
			paltas.Analysis.hierarchical_inference.ProbabilityClassAnalytical(
				mu_omega_i, cov_omega_i, log_hyperprior))
		prob_class.set_predictions(
			mu_pred_array_input=y_pred,
			prec_pred_array_input=prec_pred)

		if log_sigma:
			# Paltas' posterior always takes log(sigma) parameters,
			# so the hyperprior will be uniform in log(sigma) ...
			log_posterior = prob_class.log_post_omega
		else:
			# ... unless we wrap the posterior with a coordinate transform.
			def log_posterior(x):
				# Don't want to modify input in-place
				x = x.copy()
				# Paltas expects log sigma
				sigmas = x[n_params:]
				if sigmas.min() <= 0:
					# No need to ask Paltas, impossible
					return -np.inf
				x[n_params:] = np.log(sigmas)
				return prob_class.log_post_omega(x)

		# Store only attributes we need later
		self.log_sigma = log_sigma
		self.params = params
		self.n_params = n_params
		self.true_hyperparameters = true_hyperparameters
		self.log_posterior = log_posterior

	def _summary_df(self):
		sigma_prefix = 'log_std' if self.log_sigma else 'std'
		return pd.DataFrame(
			dict(param=(
					['mean_' + p for p in self.params]
					+ [sigma_prefix + '_' + p for p in self.params]
				 ),
				 truth=self.true_hyperparameters))

	def frequentist_asymptotic(
			self,
			hessian_step=1e-4,
			hessian_method='central'):
		"""Returns the maximum likelihood estimate and covariance matrix
		for asymptotic frequentist confidence intervals.

		Arguments:
		 - hessian_step: step size to use in Hessian computation.
		 - hessian_method: method to use in Hessian computation;
			see numdifftools for details.

		Returns tuple of:
		 - DataFrame with results
		 - estimated covariance matrix

		The covariance matrix is estimated from the Hessian of the -2 log
		likelihood; the Hessian is estimated using finite-difference methods.
		"""
		import numdifftools

		if self.log_sigma:
			warnings.warn("Not using a uniform prior, good luck...")

		# Find the maximum a posteriori / maximum likelihood estimate
		# (they agree for a uniform prior)
		# .. or at least a local max close to the truth.
		def objective(x):
			return -2*self.log_posterior(x)
		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore",
				message='invalid value encountered in subtract'
			)
			optresult = minimize(
				objective,
				x0=self.true_hyperparameters,
			)

		# Estimate covariance using the inverse Hessian
		# the minimizers's Hessian inverse estimate (optresult.hess_inv) is
		# not reliable enough, so do a new calculation with numdifftools
		hess = numdifftools.Hessian(
			objective,
			base_step=hessian_step,
			method=hessian_method)(optresult.x)
		cov = np.linalg.inv(hess)

		summary = self._summary_df()
		summary['fit'] = optresult.x
		summary['fit_unc'] = cov_to_std(cov)[0]

		return summary, cov

	def bayesian_mcmc(
			self,
			initial_walker_scatter=1e-3,
			n_samples=int(1e4),
			n_burnin=int(1e3),
			n_walkers=40,
			chains_path='chains.h5'):
		"""Return MCMC inference results

		Arguments:
		 - initial_walker_scatter: amplitude with which to vary walker
			positions. Multiplied by an (n_walkers, n_hyperparams) vector.
		 - n_samples: Number of MCMC samples to use (excluding burn-in)
		 - n_burnin: Number of burn-in samples to use
		 - n_walkers: Number of walkers to use
		 - chains_path: path in which to store sample chain as HDF5.

		Returns tuple with:
		 - DataFrame with summary of results
		 - chain excluding burn-in, (n_samples, n_hyperparams) array
		"""
		import emcee

		ndim = len(self.true_hyperparameters)

		# Scatter the initial walker states around the true values
		cur_state = (
			self.true_hyperparameters
			+ initial_walker_scatter * np.random.randn(n_walkers, ndim))
		if not self.log_sigma:
			# Don't start at negative sigmas: reflect initial state in 0
			cur_state_sigmas = cur_state[:,self.n_params:]
			cur_state[:,self.n_params:] = np.where(
				cur_state_sigmas < 0,
				-cur_state_sigmas,
				cur_state_sigmas)

		# Delete previous chains and start backend
		chains_path = Path(chains_path)
		if chains_path.exists():
			chains_path.unlink()
		backend = emcee.backends.HDFBackend(chains_path)

		sampler = emcee.EnsembleSampler(
			n_walkers,
			ndim,
			self.log_posterior,
			backend=backend)
		sampler.run_mcmc(cur_state, n_burnin + n_samples, progress=True)
		chain = sampler.chain[:,n_burnin:,:].reshape((-1,ndim))

		summary = self._summary_df()
		summary['fit'] = chain.mean(axis=0)
		summary['fit_unc'] = chain.std(axis=0)

		return summary, chain


def extract_mu_cov(config_path, params):
	"""Return (mean, cov) arrays of distribution of params
	as defined by paltas config at config_path
	"""
	if not str(config_path).endswith('.py'):
		# Maybe the user gave a folder name
		# If it has only one python file, fine, that must be the config
		py_files = list(Path(config_path).glob('*.py'))
		if len(py_files) == 1:
			config_path = py_files[0]
		else:
			raise ValueError(f"{config_path} has multiple python files")

	ch = paltas.Configs.config_handler.ConfigHandler(config_path)

	# Frst extract the mean and std of all possible parameters
	# Most are mean deflector parameters ...
	mean_std = {
		pname: _get_mean_std(
			ch.config_dict['main_deflector']['parameters'][short_names[pname]],
			short_names[pname])
		for pname in params if pname.startswith('main_deflector')
	}
	# ... except for sigma_sub
	# TODO: other subhalo params!
	mean_std[long_names['sigma_sub']] = _get_mean_std(
		ch.config_dict['subhalo']['parameters']['sigma_sub'],
		'sigma_sub')

	# Produce mean vector / cov matrix in the right order
	mu, std = np.array([mean_std[pname] for pname in params]) .T
	cov = np.diag(std**2)
	return mu, cov


def _get_mean_std(x, pname):
	# Helper to extract mean/std given a paltas config value
	# (paltas configs contain .rvs methods, not dists themselves)
	if isinstance(x, (int, float)):
		# Value was kept constant
		return x
	# Let's hope it is a scipy stats distribution, so we can
	# back out the mean and std through sneaky ways
	self = x.__self__
	dist = self.dist
	if not isinstance(dist, (
			stats._continuous_distns.norm_gen)):
		warnings.warn(
			f"Approximating {dist.name} for {pname} with a normal distribution",
			UserWarning)
	return self.mean(), self.std()


def select_from_matrix_stack(matrix_stack, select_i):
	"""Select specific simultaneous row and column indices
	from a stack of matrices"""
	sel_x, sel_y = np.meshgrid(select_i, select_i, indexing='ij')
	return (
		matrix_stack[:, sel_x.ravel(), sel_y.ravel()]
		.reshape([-1] + list(sel_x.shape)))


def cov_to_std(cov):
	"""Return (std errors, correlation coefficent matrix)
	given covariance matrix cov
	"""
	std_errs = np.diag(cov) ** 0.5
	corr = cov * np.outer(1 / std_errs, 1 / std_errs)
	return std_errs, corr
