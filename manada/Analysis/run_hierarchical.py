import numpy as np
import os, argparse, sys
import numba
from manada.Analysis import hierarchical_inference
import emcee
from multiprocessing import Pool
from importlib import import_module


def parse_args():
	"""Parse the input arguments by the user

	Returns:
		(argparse.Namespace): An instance of the Namespace object with the
			users provided values.

	"""
	# Initialize the parser and the possible inputs
	parser = argparse.ArgumentParser()
	parser.add_argument('test_index',help='The test set to run hierarchical ' +
		'inference on.',type=int)
	parser.add_argument('--n_samps', default=10000, type=int, dest='n_samps',
		help='Number of emcee samples to draw.')
	parser.add_argument('--n_lenses', default=10, type=int,dest='n_lenses',
		help='Number of lenses to include in analysis.')
	args = parser.parse_args()
	return args


def main():
	# Get the arguments
	args = parse_args()

	# Extract the parameters from the input args.
	n_lenses = args.n_lenses
	n_emcee_samps = args.n_samps
	chains_folder = '/scratch/users/swagnerc/manada/chains'
	test_index = args.test_index
	backend_path = 'test_set_%d_lenses_%d.h5'%(test_index,n_lenses)
	chains_path = os.path.join(chains_folder,backend_path)
	n_ensemble = 5

	# Load the predictions for the mean and covariance for one of our
	# ensemble models.
	mi = 3
	y_pred = (np.load('/scratch/users/swagnerc/manada/chains/'
		+ 'marg_outputs/y_pred_marg_model_%d_set_%d.npy'%(mi,
			test_index)).astype(np.float64))
	cov_pred = (np.load('/scratch/users/swagnerc/manada/chains/' +
		'marg_outputs/y_cov_marg_model_%d_set_%d.npy'%(mi,
			test_index)).astype(np.float64))
	prec_pred = (np.linalg.inv(cov_pred))

	# Pull the target parameters straight from the config file.
	sys.path.insert(0,'/scratch/users/swagnerc/manada/datasets/marg_ns/' +
		'marg_ns_test_shift_%d/'%(test_index))
	config_module = import_module('config_marg_shift_%d'%(test_index))
	md_params = config_module.config_dict['main_deflector']['parameters']
	subhalo_params = config_module.config_dict['subhalo']['parameters']

	# The interim training distribution.
	mu_omega_i = np.array([1.1,0.0,0.0,2.0,0.0,0.0,0.0,0.0,2e-3])
	cov_omega_i =np.diag(np.array([0.15,0.05,0.05,0.1,0.1,0.1,0.16,0.16,
		1.1e-3])**2)

	# We will want to initialize emcee near the correct values.
	mu_omega = np.array([md_params['theta_E'].__self__.mean(),
		md_params['gamma1'].__self__.mean(),md_params['gamma2'].__self__.mean(),
		md_params['gamma'].__self__.mean(),md_params['e1'].__self__.mean(),
		md_params['e2'].__self__.mean(),md_params['center_x'].__self__.mean(),
		md_params['center_y'].__self__.mean(),
		subhalo_params['sigma_sub'].__self__.mean()])
	cov_omega =np.diag(np.array([md_params['theta_E'].__self__.std(),
		md_params['gamma1'].__self__.std(),md_params['gamma2'].__self__.std(),
		md_params['gamma'].__self__.std(),md_params['e1'].__self__.std(),
		md_params['e2'].__self__.std(),md_params['center_x'].__self__.std(),
		md_params['center_y'].__self__.std(),
		subhalo_params['sigma_sub'].__self__.std()])**2)
	true_hyperparameters = np.concatenate([mu_omega,
		np.log(np.diag(np.sqrt(cov_omega)))])

	# A prior function that mainly just bounds the uncertainty estimation.
	@numba.njit()
	def eval_func_omega(hyperparameters):
		upper_lim = np.log(np.diag(np.sqrt(cov_omega_i)))+0.5
		lower_lim = np.log(np.diag(np.sqrt(cov_omega)))-0.2
		if (np.any(hyperparameters[len(hyperparameters)//2:]>upper_lim) or
			np.any(hyperparameters[len(hyperparameters)//2:]<lower_lim)):
			return -np.inf
		# Make sure the means don't drift too far into absurdity
		if (np.any(hyperparameters[:len(hyperparameters)//2]>mu_omega_i+
				np.sqrt(np.diag(cov_omega_i))*5) or
			np.any(hyperparameters[:len(hyperparameters)//2]<mu_omega_i-
				np.sqrt(np.diag(cov_omega_i))*5)):
			return -np.inf
		return 0

	prob_class = hierarchical_inference.ProbabilityClassAnalytical(mu_omega_i,
		cov_omega_i,eval_func_omega)
	prob_class.set_predictions(mu_pred_array_input=y_pred[:,:n_lenses],
		prec_pred_array_input=prec_pred[:,:n_lenses])

	with Pool() as pool:
		n_walkers = 40
		ndim = 18
		initial_std = true_hyperparameters*0.002
		initial_std[initial_std==0] += 0.002
		initial_std[:len(initial_std)//2] *= 5
		cur_state = (np.random.rand(n_walkers, ndim)*2-1)*initial_std
		cur_state += true_hyperparameters
		backend = emcee.backends.HDFBackend(chains_path)
		sampler = emcee.EnsembleSampler(n_walkers, ndim,prob_class.log_post_omega,
			backend=backend,pool=pool)
		sampler.run_mcmc(cur_state,n_emcee_samps,progress=True,
			skip_initial_state_check=True)


if __name__ == '__main__':
	main()
