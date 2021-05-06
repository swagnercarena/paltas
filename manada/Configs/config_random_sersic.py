from copy import deepcopy
from manada.Sources.sersic import SingleSersicSource
from manada.Sources.cosmos import HUBBLE_ACS_PIXEL_WIDTH
from manada.Configs.config_d_los_sigma_sub import *

from scipy import stats

# Avoid mutating the original config_dict
# (in case someone loads both configs)
config_dict = deepcopy(config_dict)

config_dict['source']['class'] = SingleSersicSource
config_dict['source']['parameters'] = dict(
	z_source=config_dict['source']['parameters']['z_source'],
	# Fitted to COSMOS Sersicfit results
	# Note the pixel scaling uses the COSMOS pixel width , not Manada's
	amp=lambda: stats.lognorm(s=0.895, scale=0.0410).rvs() / HUBBLE_ACS_PIXEL_WIDTH**2,
	# z_scaling to 1.5 is applied before fit.
	R_sersic=lambda : HUBBLE_ACS_PIXEL_WIDTH * stats.lognorm(s=0.554, scale=17.1).rvs(),
	# NOT fitted to COSMOS.
	# COSMOS has a big mass at the maximum n=6, probably indicating poor fits / very spread out galaxies?
	n_sersic=1.,
	# Fitted to COSMOS again, with phi replaced by uniform random angles
	e1=stats.t(df=5.69, scale=0.171).rvs,
	e2=stats.t(df=5.69, scale=0.171).rvs,
	# Keep galaxy in center; lens position is already varied.
	# (probably unnecessary to fix though, check?)
	center_x=0.,
	center_y=0.,
)
