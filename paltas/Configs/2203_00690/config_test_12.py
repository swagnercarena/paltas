from paltas.Configs.xxxx_yyyy.config_val import *

config_dict['subhalo']['parameters']['sigma_sub'] = norm(loc=2.4e-3,
	scale=1.5e-4).rvs
config_dict['main_deflector']['parameters']['gamma'] = truncnorm(-197.2,
	np.inf,loc=1.972,scale=0.01).rvs
config_dict['main_deflector']['parameters']['theta_E'] = truncnorm(-70.9,
	np.inf,loc=1.063,scale=0.015).rvs
config_dict['main_deflector']['parameters']['e1'] = norm(loc=0.043,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['e2'] = norm(loc=0.040,
	scale=0.01).rvs
config_dict['main_deflector']['parameters']['center_x'] = norm(loc=-0.057,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['center_y'] = norm(loc=-0.075,
	scale=0.016).rvs
config_dict['main_deflector']['parameters']['gamma1'] = norm(loc=0.003,
	scale=0.005).rvs
config_dict['main_deflector']['parameters']['gamma2'] = norm(loc=-0.005,
	scale=0.005).rvs
