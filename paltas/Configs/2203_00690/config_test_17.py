from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=3.4e-3,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-202.0,
	np.inf,loc=2.020,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-72.8,
	np.inf,loc=1.091,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=-0.001,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=0.039,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=-0.076,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=-0.025,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=0.012,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=0.008,
	scale=0.005).rvs
