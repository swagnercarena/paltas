# Configuration for a tight spread on main deflector with PEMD + SHEAR, sources
# drawn from COSMOS, and only varying the d_los and sigma_sub of the DG_19
# subhalo and los classes

from manada.Sampling import distributions
import numpy as np
from scipy.stats import norm, lognorm
from manada.Substructure.los_dg19 import LOSDG19
from manada.Substructure.subhalos_dg19 import SubhalosDG19
from manada.Sources.cosmos import COSMOSExcludeCatalog
import pandas as pd
import manada
import os

# Define a multivariate distribution we'll use
mean = np.ones(2)
cov = np.array([[1.0,0.7],[0.7,1.0]])
min_values = np.zeros(2)
tmn = distributions.TruncatedMultivariateNormal(mean,cov,min_values,None)

# Define the numerics kwargs
kwargs_numerics = {'supersampling_factor':2}
numpix = 64

# Define some general image kwargs for the dataset
mask_radius = 0.5
mag_cut = 2.0

# Define the cosmos path
root_path = manada.__path__[0][:-7]
cosmos_folder = root_path + r'/datasets/cosmos/COSMOS_23.5_training_sample/'

config_dict = {
	'subhalo':{
		'class': SubhalosDG19,
		'parameters':{
			'sigma_sub':lognorm(scale=0.1,s=0.5).rvs,
			'shmf_plaw_index':-1.83,
			'm_pivot': 1e8,'m_min': 1e7,'m_max': 1e10,
			'c_0':18,'conc_zeta':-0.2,'conc_beta':0.8,
			'conc_m_ref': 1e8,'dex_scatter': 0.1,
			'k1':0.0, 'k2':0.0
		}
	},
	'los':{
		'class': LOSDG19,
		'parameters':{
			'delta_los':lognorm(scale=1,s=0.7).rvs,
			'm_min':1e7,'m_max':1e10,'z_min':0.01,
			'dz':0.01,'cone_angle':8.0,'r_min':0.5,'r_max':10.0,
			'c_0':18,'conc_zeta':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.1,'alpha_dz_factor':5.0
		}
	},
	'main_deflector':{
		'models': ['PEMD','SHEAR'],
		'parameters':{
			'M200': 1e13,
			'z_lens': 0.5,
			'gamma': lognorm(scale=2.01,s=0.05).rvs,
			'theta_E': lognorm(scale=1.1,s=0.1).rvs,
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
			'max_z':1.0,'minimum_size_in_pixels':64,'min_apparent_mag':20,
			'smoothing_sigma':0.08,'random_rotation':True,
			'min_flux_radius':10.0,'source_exclusion_list':pd.read_csv(
				os.path.join(root_path,'manada/Sources/bad_galaxies.csv'),
				names=['catalog_i'])['catalog_i'].to_numpy()}
	},
	'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
	'psf':{
		'parameters':{
			'psf_type':'GAUSSIAN',
			'fwhm': 0.1
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.08,'ccd_gain':2.5,'read_noise':4.0,
			'magnitude_zero_point':25.9463,
			'exposure_time':5400.0,'sky_brightness':22,
			'num_exposures':1, 'background_noise':None
		}
	}
}
