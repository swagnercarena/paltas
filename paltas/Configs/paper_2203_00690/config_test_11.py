from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=2.2e-3,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-196.7,
	np.inf,loc=1.967,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-76.4,
	np.inf,loc=1.146,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=-0.008,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=0.031,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=-0.011,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=0.036,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=-0.013,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=0.014,
	scale=0.005).rvs
