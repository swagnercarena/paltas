from copy import deepcopy
from manada.Configs.config_d_los_sigma_sub import *

# Avoid mutating the original config_dict
# (in case someone loads both configs)
config_dict = deepcopy(config_dict)

repeat_non_substructure = 16

# Do NOT generate line-of-sight substructure
# (to avoid competing with the sigma_sub signal)
del config_dict['los']
