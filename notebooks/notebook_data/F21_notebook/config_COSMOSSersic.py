# Configuration for use with the F21_Manada_Changes notebook.

import numpy as np
from scipy.stats import norm, truncnorm, uniform
from manada.MainDeflector.simple_deflectors import PEMD
from manada.Sources.cosmos_sersic import COSMOSSersic

output_ab_zeropoint=25.127
# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':2}

# The number of pixels in the CCD.
numpix = 100

cosmos_folder = '/Users/smericks/Desktop/StrongLensing/COSMOS_23.5_training_sample/'

config_dict = {
	'main_deflector':{
		'class': PEMD,
		'parameters':{
			'M200': 1e13,
			'z_lens': 0.5,
			'gamma': truncnorm(-20,np.inf,loc=2.0,scale=0.1).rvs,
			'theta_E': truncnorm(-1.1/0.15,np.inf,loc=1.1,scale=0.15).rvs,
			'e1': norm(loc=0.0,scale=0.1).rvs,
			'e2': norm(loc=0.0,scale=0.1).rvs,
			'center_x': norm(loc=0.0,scale=0.1).rvs,
			'center_y': norm(loc=0.0,scale=0.16).rvs,
		}
	},
	'source':{
		'class': COSMOSSersic,
		'parameters':{
			'z_source':1.5,'cosmos_folder':cosmos_folder,
			'max_z':1.0,'minimum_size_in_pixels':64,'min_apparent_mag':20,
			'smoothing_sigma':0.08,'random_rotation':True,
			'output_ab_zeropoint':output_ab_zeropoint,
			'min_flux_radius':10.0,
			# need to ask about this \/
			'mag_sersic':uniform(loc=20, scale=2).rvs,
			'R_sersic':truncnorm(-1.0/0.2,np.inf,loc=1.0,scale=0.2).rvs,
			'n_sersic':truncnorm(-1.2/0.2,np.inf,loc=1.2,scale=0.2).rvs,
			'e1_sersic':norm(loc=0.0,scale=0.1).rvs,
			'e2_sersic':norm(loc=0.0,scale=0.1).rvs,
			'center_x_sersic':0.0,
			'center_y_sersic':0.0}
	},
	'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
	'psf':{
        'parameters':{'psf_type':'NONE'}
    },
	'detector':{
		'parameters':{
			'pixel_scale':0.04,'ccd_gain':1.58,'read_noise':3.0,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':1380,'sky_brightness':21.83,
			'num_exposures':1,'background_noise':None
		}
	}
}

"""
	'psf':{
		'parameters':{
			'psf_type':'GAUSSIAN',
			'fwhm': 0.04
		}
	},
"""