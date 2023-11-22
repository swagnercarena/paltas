# Configuration for lenses with varied redshift and realistic lens light.

import numpy as np
from scipy.stats import norm, truncnorm, uniform
from paltas.Substructure.los_dg19 import LOSDG19
from paltas.Substructure.subhalos_dg19 import SubhalosDG19
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.cosmos import COSMOSExcludeCatalog
from paltas.Sources.sersic import DoubleSersicCOSMODC2
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
mag_cut = 2.0

# Define arguments that will be used multiple times
output_ab_zeropoint = 25.9

# Define the cosmos path
root_path = paltas.__path__[0][:-7]
cosmos_folder = root_path + r'/datasets/cosmos/COSMOS_23.5_training_sample/'

# Set up a callable to grab a tinytim psf from our list. These PSF are
# generated with a supersampling factor of 2 and include geometric
# distortions. More details can be found in the parameter files.
acs_tt_psf_list = [
	root_path + r'/datasets/hst_psf/acs_tinytim/psf_tt_%d.fits'%(i)
	for i in range(1,21)]


def get_acs_psf():
	psf_path = np.random.choice(acs_tt_psf_list)
	hdul = fits.open(psf_path)
	return hdul[0].data


config_dict = {
	'subhalo':{
		'class': SubhalosDG19,
		'parameters':{
			'sigma_sub':norm(loc=2e-3,scale=1.1e-3).rvs,
			'shmf_plaw_index':uniform(loc=-1.92,scale=0.1).rvs,
			'm_pivot': 1e10,'m_min': 1e8,'m_max': 1e10,
			'c_0':None,'conc_zeta':None,'conc_beta':None,'conc_m_ref': 1e8,
			'dex_scatter': None,'k1':0.0, 'k2':0.0
		}
	},
	'los':{
		'class': LOSDG19,
		'parameters':{
			'delta_los':norm(loc=1,scale=0.6).rvs,
			'm_min':1e8,'m_max':1e10,'z_min':0.01,
			'dz':0.01,'cone_angle':8.0,'r_min':0.5,'r_max':10.0,
			'c_0':None,'conc_zeta':None,'conc_beta':None,'conc_m_ref':1e8,
			'dex_scatter':None,'alpha_dz_factor':5.0
		}
	},
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'M200': 1e13,'z_lens': None,
			'gamma': truncnorm(-20,np.inf,loc=2.0,scale=0.1).rvs,
			'theta_E': truncnorm(-1.25/0.3,np.inf,loc=1.25,scale=0.3).rvs,
			'e1': norm(loc=0.0,scale=0.1).rvs,
			'e2': norm(loc=0.0,scale=0.1).rvs,
			'center_x': None,
			'center_y': None,
			'gamma1': norm(loc=0.0,scale=0.05).rvs,
			'gamma2': norm(loc=0.0,scale=0.05).rvs,
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'source':{
		'class': COSMOSExcludeCatalog,
		'parameters':{
			'z_source':None,'cosmos_folder':cosmos_folder,
			'max_z':1.0,'minimum_size_in_pixels':64,'faintest_apparent_mag':20,
			'smoothing_sigma':0.00,'random_rotation':True,
			'output_ab_zeropoint':output_ab_zeropoint,
			'center_x':norm(loc=0.0,scale=0.16).rvs,
			'center_y':norm(loc=0.0,scale=0.16).rvs,
			'source_absolute_magnitude':uniform(loc=-24.0,scale=3.0).rvs,
			'min_flux_radius':10.0,'source_exclusion_list':np.append(
				pd.read_csv(
					os.path.join(root_path,'paltas/Sources/bad_galaxies.csv'),
					names=['catalog_i'])['catalog_i'].to_numpy(),
				pd.read_csv(
					os.path.join(root_path,'paltas/Sources/val_galaxies.csv'),
					names=['catalog_i'])['catalog_i'].to_numpy())}
	},
	'lens_light': {
		'class': DoubleSersicCOSMODC2,
		'parameters': {
			'cosmodc2_file': root_path + r'/datasets/cosmodc2/cosmodc2_selected_10132_10142_18mag.npz',
			'output_ab_zeropoint': output_ab_zeropoint,
			'center_x': None,
			'center_y': None,
			'z_source': None
		}
	},
	'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
	'psf':{
		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source': get_acs_psf,
			'point_source_supersampling_factor':2
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.050,
			'ccd_gain':uniform(loc=1.886,scale=0.134).rvs,
			# See https://hst-docs.stsci.edu/acsdhb/chapter-4-acs-data-processing-considerations/4-1-read-noise-and-a-to-d-conversion,
			# SLACS observations were around 2006?
			# Which amplifiers were used? All of them?
			'read_noise':uniform(loc=5.25,scale=0.25).rvs,
			'magnitude_zero_point':output_ab_zeropoint,
			# Exposure of one of the drizzled exposures.
			# The 42 full orbit SLACS observations with F814W data have
			# a total exposure distributed roughly as 2184 +- 116 seconds.
			# (minimum is 2088 sec, maximum 2520 sec)
			'exposure_time':uniform(loc=2184/4,scale=116/4).rvs,
			'sky_brightness':uniform(21.13,scale=0.57).rvs,
			# This parameter appears to be ignored
			#'num_exposures':1,
			'background_noise':None
		}
	},
	'drizzle':{
		'parameters':{
			'supersample_pixel_scale':0.025,'output_pixel_scale':0.050,
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
				dist=uniform(loc=0.1,scale=0.06).rvs),
			'main_deflector:z_lens,lens_light:z_source,source:z_source':(
				distributions.RedshiftsLensLight(z_lens_min=0.06,
					z_lens_mean=0.21,z_lens_std=0.08,z_source_min=0.3,
					z_source_mean=0.65,z_source_std=0.2)),
			'main_deflector:center_x,lens_light:center_x':(
				distributions.DuplicateScatter(
					dist=norm(loc=0,scale=0.16).rvs,scatter=0.05)),
			'main_deflector:center_y,lens_light:center_y':(
				distributions.DuplicateScatter(
					dist=norm(loc=0,scale=0.16).rvs,scatter=0.05))
		}
	}
}
