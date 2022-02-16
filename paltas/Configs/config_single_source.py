from copy import deepcopy
from paltas.Sources.cosmos import COSMOSSersicCatalog
from paltas.Configs.config_d_los_sigma_sub import *
from scipy.stats import norm

# Avoid mutating the original config_dict
# (in case someone loads both configs)
config_dict = deepcopy(config_dict)

# Force no noise
no_noise = True

# This forces the selection of only one source.
config_dict['source']['parameters'] = {'z_source':1.5,
	'cosmos_folder':cosmos_folder,'max_z':0.05,'minimum_size_in_pixels':650,
	'faintest_apparent_mag':20,'smoothing_sigma':0.00,'random_rotation':True,
	'center_x':norm(loc=0.0,scale=0.16).rvs,
	'center_y':norm(loc=0.0,scale=0.16).rvs,'min_flux_radius':10.0}
