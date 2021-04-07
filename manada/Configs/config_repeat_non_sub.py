from copy import deepcopy
from manada.Configs.config_d_los_sigma_sub import *

# Avoid mutating the original config_dict
# (in case someone loads both configs)
config_dict = deepcopy(config_dict)

repeat_non_substructure = 16

# Do NOT generate line-of-sight substructure
# (to avoid competing with the sigma_sub signal)
del config_dict['los']


class CyclingSampler:
    def __init__(self, cycle_through):
        # Manada calls one extra sample for initialization...
        self.index = -1
        self.cycle_through = cycle_through

    def __call__(self):
        result = self.cycle_through[self.index]
        self.index = (self.index + 1) % len(self.cycle_through)
        return result


config_dict['subhalo']['parameters']['sigma_sub'] = \
    CyclingSampler(np.concatenate([
        # First draw is for the non-substructure
        [float('nan')],
        # Next draws are substructure
        np.linspace(0, 0.2, 16)]))