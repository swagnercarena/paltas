from paltas.Configs.config_d_los_sigma_sub import *
from paltas.Sampling import distributions
import numpy as np

plaw_sigma_mean = np.array([-1.78,2e-3])
plaw_sigma_cov = np.array([[5.29e-2,1.012e-4],[1.012e-4,1.21e-6]])
tmn = distributions.TruncatedMultivariateNormal(plaw_sigma_mean,plaw_sigma_cov,
	None,None)

config_dict['cross_object'] = {
	'parameters':{
		'subhalo:shmf_plaw_index,subhalo:sigma_sub':tmn
	}
}
