"""Config to generate a training set under extremely unrealistic / generous
assumptions.
"""
from copy import deepcopy
from manada.Sources.cosmos import COSMOSSersicCatalog
from manada.Configs.config_d_los_sigma_sub import *

# Avoid mutating the original config_dict
# (in case someone loads both configs)
config_dict = deepcopy(config_dict)

# Use Sersic sources instead of real images
config_dict['source']['class'] = COSMOSSersicCatalog

# Do NOT generate line-of-sight substructure
# (to avoid competing with the sigma_sub signal)
del config_dict['los']

# Do NOT mask any pixels in the center
del mask_radius

# Half pixel size, double image width (quadruple image size)
# Turn off read noise and sky background
numpix = 128
config_dict['detector']['parameters'].update(dict(
    pixel_scale=0.04,
    read_noise=1e-9,
    sky_brightness=99.))

# Turn off the Poisson readout noise
no_noise = True

# Turn off PSF
config_dict['psf']['parameters']['fwhm'] = 1e-9
