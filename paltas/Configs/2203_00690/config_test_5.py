from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=1.0e-3,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-202.7,
	np.inf,loc=2.027,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-69.2,
	np.inf,loc=1.038,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=0.033,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=0.017,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=-0.062,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=-0.051,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=0.000,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=-0.000,
	scale=0.005).rvs
