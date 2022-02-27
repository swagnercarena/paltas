from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=2.6e-3,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-200.4,
	np.inf,loc=2.004,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-76.2,
	np.inf,loc=1.143,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=-0.023,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=0.038,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=0.071,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=0.007,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=-0.011,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=-0.002,
	scale=0.005).rvs
