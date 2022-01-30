from copy import deepcopy
from paltas.Sources.cosmos import COSMOSSersicCatalog
from paltas.Configs.config_d_los_sigma_sub import *

# Avoid mutating the original config_dict
# (in case someone loads both configs)
config_dict = deepcopy(config_dict)

config_dict['source']['class'] = COSMOSSersicCatalog
