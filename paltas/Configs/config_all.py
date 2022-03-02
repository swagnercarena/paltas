# Includes line-of-sight halos, subhalos, a PEMD deflector with external shear,
# COSMOS sources, and HST observaitonal effects for Wide Field Camera 3 (WFC3)
# UVIS channel with the F814W filter that includes drizzling. Assumes a planck
# 2018 cosmology.

import numpy as np
from scipy.stats import norm, truncnorm, uniform
from paltas.Substructure.los_dg19 import LOSDG19
from paltas.Substructure.subhalos_dg19 import SubhalosDG19
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.cosmos import COSMOSExcludeCatalog
from lenstronomy.Util.kernel_util import degrade_kernel
from paltas.Sampling import distributions
from astropy.io import fits
import pandas as pd
import paltas
import os

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':2,'supersampling_convolution':True}
# We do not use point_source_supersampling_factor but it must be passed in to
# surpress a warning.
kwargs_numerics['point_source_supersampling_factor'] = (
	kwargs_numerics['supersampling_factor'])
# This is always the number of pixels for the CCD. If drizzle is used, the
# final image will be larger.
numpix = 128

# Define some general image kwargs for the dataset
mask_radius = 0.5
mag_cut = 2.0

# Define arguments that will be used multiple times
output_ab_zeropoint = 25.127

# Define the cosmos path
root_path = paltas.__path__[0][:-7]
cosmos_folder = root_path + r'/datasets/cosmos/COSMOS_23.5_training_sample/'

# Load the empirical psf. Grab a psf from the middle of the first chip.
# Degrade to account for the 4x supersample
hdul = fits.open(os.path.join(root_path,
	'datasets/hst_psf/emp_psf_f814w.fits'))
# Don't leave any 0 values in the psf.
psf_pix_map = degrade_kernel(hdul[0].data[17]-np.min(hdul[0].data[17]),2)

config_dict = {
	'subhalo':{
		'class': SubhalosDG19,
		'parameters':{
			'sigma_sub':norm(loc=2e-3,scale=1.1e-3).rvs,
			'shmf_plaw_index':uniform(loc=-1.92,scale=0.1).rvs,
			'm_pivot': 1e10,'m_min': 1e7,'m_max': 1e10,
			'c_0':uniform(loc=16,scale=2).rvs,
			'conc_zeta':uniform(loc=-0.3,scale=0.1).rvs,
			'conc_beta':uniform(loc=0.55,scale=0.3).rvs,
			'conc_m_ref': 1e8,
			'dex_scatter': uniform(loc=0.1,scale=0.06).rvs,
			'k1':0.0, 'k2':0.0
		}
	},
	'los':{
		'class': LOSDG19,
		'parameters':{
			'delta_los':norm(loc=1,scale=0.6).rvs,
			'm_min':1e7,'m_max':1e10,'z_min':0.01,
			'dz':0.01,'cone_angle':8.0,'r_min':0.5,'r_max':10.0,
			'c_0':uniform(loc=16,scale=2).rvs,
			'conc_zeta':uniform(loc=-0.3,scale=0.1).rvs,
			'conc_beta':uniform(loc=0.55,scale=0.3).rvs,
			'conc_m_ref': 1e8,
			'dex_scatter': uniform(loc=0.1,scale=0.06).rvs,
			'alpha_dz_factor':5.0
		}
	},
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'M200': 1e13,
			'z_lens': 0.5,
			'gamma': truncnorm(-20,np.inf,loc=2.0,scale=0.1).rvs,
			'theta_E': truncnorm(-1.1/0.15,np.inf,loc=1.1,scale=0.15).rvs,
			'e1': norm(loc=0.0,scale=0.1).rvs,
			'e2': norm(loc=0.0,scale=0.1).rvs,
			'center_x': norm(loc=0.0,scale=0.16).rvs,
			'center_y': norm(loc=0.0,scale=0.16).rvs,
			'gamma1': norm(loc=0.0,scale=0.05).rvs,
			'gamma2': norm(loc=0.0,scale=0.05).rvs,
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'source':{
		'class': COSMOSExcludeCatalog,
		'parameters':{
			'z_source':1.5,'cosmos_folder':cosmos_folder,
			'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
			'smoothing_sigma':0.00,'random_rotation':True,
			'output_ab_zeropoint':output_ab_zeropoint,
			'center_x':norm(loc=0.0,scale=0.16).rvs,
			'center_y':norm(loc=0.0,scale=0.16).rvs,
			'min_flux_radius':10.0,'source_exclusion_list':np.append(
				pd.read_csv(
					os.path.join(root_path,'paltas/Sources/bad_galaxies.csv'),
					names=['catalog_i'])['catalog_i'].to_numpy(),
				pd.read_csv(
					os.path.join(root_path,'paltas/Sources/val_galaxies.csv'),
					names=['catalog_i'])['catalog_i'].to_numpy())}
	},
	'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
	'psf':{
		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source': psf_pix_map,
			'point_source_supersampling_factor':2
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.040,'ccd_gain':1.58,'read_noise':3.0,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':1380,'sky_brightness':21.83,
			'num_exposures':1,'background_noise':None
		}
	},
	'drizzle':{
		'parameters':{
			'supersample_pixel_scale':0.020,'output_pixel_scale':0.030,
			'wcs_distortion':None,
			'offset_pattern':[(0,0),(0.5,0),(0.0,0.5),(-0.5,-0.5)],
			'psf_supersample_factor':2
		}
	},
	'cross_object':{
		'parameters':{
			'subhalo:c_0,los:c_0':distributions.Duplicate(
				dist=uniform(loc=16,scale=2).rvs),
			'subhalo:conc_zeta,los:conc_zeta':distributions.Duplicate(
				dist=uniform(loc=-0.3,scale=0.1).rvs),
			'subhalo:conc_beta,los:conc_beta':distributions.Duplicate(
				dist=uniform(loc=0.55,scale=0.3).rvs),
			'subhalo:dex_scatter,los:dex_scatter':distributions.Duplicate(
				dist=uniform(loc=0.1,scale=0.06).rvs)
		}
	}
}
