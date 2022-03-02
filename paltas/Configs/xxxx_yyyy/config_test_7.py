from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=1.4e-3,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-203.8,
	np.inf,loc=2.038,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-70.6,
	np.inf,loc=1.059,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=0.044,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=-0.028,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=0.009,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=-0.013,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=-0.022,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=0.023,
	scale=0.005).rvs
