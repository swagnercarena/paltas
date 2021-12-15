import numpy as np
import os, argparse
import numba
from manada.Analysis import hierarchical_inference
import emcee
from multiprocessing import Pool


def parse_args():
    """Parse the input arguments by the user

    Returns:
        (argparse.Namespace): An instance of the Namespace object with the
            users provided values.

    """
    # Initialize the parser and the possible inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('backend_path',help='The path to save the emcee ' +
                        'samples to.')
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
    chains_path = os.path.join(chains_folder,args.backend_path)
    y_pred = np.load(os.path.join(chains_folder,'y_pred.npy'))
    prec_pred = np.load(os.path.join(chains_folder,'prec_pred.npy'))

    # The mean anhierarchical_inference.ProbabilityClassAnalytical the training distribution.
    mu_omega_i = np.array([1.1,0.0,0.0,2.0,0.0,0.0,0.0,0.0,-1.78,2e-3,1])
    cov_omega_i =np.diag(np.array([0.15,0.05,0.05,0.1,0.1,0.1,0.16,0.16,0.23,1.1e-3,0.6])**2)

    # It will also be useful to have the correct hyperparameters for the test set on hand
    mu_omega = np.array([1.1,0.0,0.0,2.0,0.0,0.0,0.0,0.0,-1.9,3e-3,1])
    cov_omega =np.diag(np.array([0.15,0.05,0.05,0.1,0.1,0.1,0.16,0.16,0.05,1.5e-4,0.6])**2)
    true_hyperparameters = np.concatenate([mu_omega,np.log(np.diag(np.sqrt(cov_omega)))])

    @numba.njit()
    def eval_func_omega(hyperparameters):
        upper_lim = np.log(np.diag(np.sqrt(cov_omega_i)))+0.5
        lower_lim = np.log(np.diag(np.sqrt(cov_omega)))-0.2
        if (np.any(hyperparameters[len(hyperparameters)//2:]>upper_lim) or 
            np.any(hyperparameters[len(hyperparameters)//2:]<lower_lim)):
            return -np.inf
        # Make sure the means don't drift too far into absurdity
        if (np.any(hyperparameters[:len(hyperparameters)//2]>mu_omega_i+np.sqrt(np.diag(cov_omega_i))*5) or
            np.any(hyperparameters[:len(hyperparameters)//2]<mu_omega_i-np.sqrt(np.diag(cov_omega_i))*5)):
            return -np.inf
        return 0

    prob_class = hierarchical_inference.ProbabilityClassAnalytical(mu_omega_i,cov_omega_i,eval_func_omega)
    prob_class.set_predictions(mu_pred_array_input=y_pred[:n_lenses],
                               prec_pred_array_input=prec_pred[:n_lenses].astype(np.float64))

    with Pool() as pool:
        n_walkers = 44
        ndim = 22
        initial_std = true_hyperparameters*0.002
        initial_std[initial_std==0] += 0.002
        initial_std[:len(initial_std)//2] *= 5
        cur_state = (np.random.rand(n_walkers, ndim)*2-1)*initial_std + true_hyperparameters
        backend = emcee.backends.HDFBackend(chains_path)
        sampler = emcee.EnsembleSampler(n_walkers, ndim,prob_class.log_post_omega, backend=backend,pool=pool)
        sampler.run_mcmc(cur_state,n_emcee_samps,progress=True,skip_initial_state_check=True)
        
if __name__ == '__main__':
    main()