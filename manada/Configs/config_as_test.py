from manada.Configs.config_d_los_sigma_sub import *
from scipy.stats import norm, lognorm

config_dict['subhalo']['parameters']['shmf_plaw_index'] = norm(loc=-1.9,
	scale=0.05).rvs
config_dict['subhalo']['parameters']['sigma_sub'] = lognorm(scale=5.0e-3,
	s=0.05).rvs
