# Configuration to reproduce h0rton Sersic training set, but changing magnitude,
# R_sersic, and n_sersic to match distributions of COSMOS images that pass cuts

from paltas.Configs.Horton.config_horton import *
from scipy.stats import lognorm

config_dict['source']['parameters']['magnitude'] = truncnorm(-3.31,6.37,
	loc=20.72,scale=0.82).rvs
config_dict['source']['parameters']['R_sersic'] = lognorm(0.562,loc=0.0,
	scale=0.529).rvs
config_dict['source']['parameters']['n_sersic'] = uniform(loc=0.398,
	scale=5.6).rvs
