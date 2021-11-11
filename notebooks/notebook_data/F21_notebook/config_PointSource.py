# Example configuration for use with the Understanding_Manada_Pipeline notebook.

import numpy as np
from scipy.stats import norm, truncnorm
from manada.MainDeflector.simple_deflectors import PEMD
from manada.Sources.sersic import SingleSersicSource
from manada.Sources.cosmos import COSMOSCatalog
from manada.PointSource.single_point_source import SinglePointSource

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':2}

# The number of pixels in the CCD.
numpix = 128

# Define some general image kwargs for the dataset
# The radius in arcseconds of a mask to apply at the center of the image
mask_radius = 0.5
# A magnification cut - images where the source is magnified by less than this
# factor will be resampled.
mag_cut = 2.0

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
		'class': COSMOSCatalog,
		'parameters':{
			'z_source':1.5,'cosmos_folder':cosmos_folder,
			'max_z':1.0,'minimum_size_in_pixels':64,'min_apparent_mag':20,
			'smoothing_sigma':0.08,'random_rotation':True,
			'output_ab_zeropoint':25.127,
			'min_flux_radius':10.0,}
	},
	'point_source':{
		'class': SinglePointSource,
		'parameters':{
			'x_point_source':0.01,
			'y_point_source':0.01,
			'magnitude':24.8,
			'mag_zeropoint':25.127,
			'compute_time_delays':False}
	},
	'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
	'psf':{
		'parameters':{
			'psf_type':'GAUSSIAN',
			'fwhm': 0.04
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.04,'ccd_gain':1.58,'read_noise':3.0,
			'magnitude_zero_point':25.127,
			'exposure_time':1380,'sky_brightness':21.83,
			'num_exposures':1,'background_noise':None
		}
	}
}
