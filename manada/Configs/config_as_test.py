from manada.Configs.config_as_val import *
from scipy.stats import norm, lognorm

config_dict['subhalo']['parameters']['shmf_plaw_index'] = norm(loc=-1.9,
	scale=0.05).rvs
config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=3.0e-3,
	scale=1.5e-4).rvs
