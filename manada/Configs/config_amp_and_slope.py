from manada.Configs.config_d_los_sigma_sub import *
from scipy.stats import truncnorm
import numpy as np

config_dict['subhalo']['parameters']['shmf_plaw_index'] = truncnorm(-2.5,
	np.inf,loc=-1.7,scale=0.4).rvs
