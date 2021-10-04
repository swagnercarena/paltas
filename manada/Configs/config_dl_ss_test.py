from manada.Configs.config_d_los_sigma_sub import *
from scipy.stats import norm

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=1.3e-1,
	scale=1e-3).rvs
config_dict['los']['parameters']['delta_los'] = 1