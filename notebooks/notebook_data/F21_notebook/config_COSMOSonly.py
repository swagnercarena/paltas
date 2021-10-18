# Configuration for use with the F21_Manada_Changes notebook.

from manada.Sources.cosmos import COSMOSCatalog
import numpy as np
from scipy.stats import norm, truncnorm, uniform
from manada.MainDeflector.simple_deflectors import PEMD
from manada.Sources.cosmos import COSMOSCatalog

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':2}

# The number of pixels in the CCD.
numpix = 100

cosmos_folder = '/mnt/c/Users/idkwh/Desktop/LSST_DESC/COSMOS_23.5_training_sample/'

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
			'minimum_size_in_pixels':650,
			'min_apparent_mag':20,
			'max_z':0.05,
			'smoothing_sigma':0.08,
			'cosmos_folder':cosmos_folder,
			'random_rotation':True,
			'min_flux_radius':10.0,
			'output_ab_zeropoint':25.127,
			'z_source':1.5}
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