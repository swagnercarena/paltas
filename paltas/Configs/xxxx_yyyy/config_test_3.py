from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=6.0e-4,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-201.5,
	np.inf,loc=2.015,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-71.7,
	np.inf,loc=1.076,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=-0.032,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=-0.025,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=0.070,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=0.048,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=-0.012,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=-0.007,
	scale=0.005).rvs
