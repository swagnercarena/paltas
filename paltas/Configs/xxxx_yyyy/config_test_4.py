from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=8.0e-4,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-197.4,
	np.inf,loc=1.974,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-69.1,
	np.inf,loc=1.037,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=-0.029,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=0.018,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=0.034,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=0.059,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=0.021,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=-0.008,
	scale=0.005).rvs
