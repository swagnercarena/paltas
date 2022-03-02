from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=2.0e-3,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-198.8,
	np.inf,loc=1.988,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-75.2,
	np.inf,loc=1.128,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=0.050,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=0.031,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=0.002,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=-0.006,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=0.018,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=-0.022,
	scale=0.005).rvs
