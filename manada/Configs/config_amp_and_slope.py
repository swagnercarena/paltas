from manada.Configs.config_d_los_sigma_sub import *
from scipy.stats import norm

config_dict['subhalo']['parameters']['shmf_plaw_index'] = norm(loc=-1.85,
	scale=0.5).rvs
