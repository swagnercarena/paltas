from copy import deepcopy
from manada.Sources.cosmos import COSMOSSersicCatalog
from manada.Configs.config_d_los_sigma_sub import *

# Avoid mutating the original config_dict
# (in case someone loads both configs)
config_dict = deepcopy(config_dict)

config_dict['source']['class'] = COSMOSSersicCatalog
