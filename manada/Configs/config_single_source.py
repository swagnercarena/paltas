from copy import deepcopy
from manada.Sources.cosmos import COSMOSSersicCatalog
from manada.Configs.config_d_los_sigma_sub import *

# Avoid mutating the original config_dict
# (in case someone loads both configs)
config_dict = deepcopy(config_dict)

# Force no noise
no_noise = True

# This forces the selection of only one source.
config_dict['source']['parameters'] = {'z_source':1.5,
	'cosmos_folder':cosmos_folder,'max_z':0.05,'minimum_size_in_pixels':650,
	'min_apparent_mag':20,'smoothing_sigma':0.00,'random_rotation':True,
	'min_flux_radius':10.0}
