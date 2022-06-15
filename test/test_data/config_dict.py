# A test configuration dict

from paltas.Sampling import distributions
import numpy as np
from scipy.stats import uniform, norm, lognorm, multivariate_normal
from paltas.Substructure.los_dg19 import LOSDG19
from paltas.Substructure.subhalos_dg19 import SubhalosDG19
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.cosmos import COSMOSCatalog

# Define a multivariate distribution we'll use
mean = np.array([1.0,2e-3])
cov = np.array([[1.0,7e-4],[7e-4,1e-6]])
min_values = np.zeros(2)
tmn = distributions.TruncatedMultivariateNormal(mean,cov,min_values,None)

# Define the numerics kwargs
kwargs_numerics = {'supersampling_factor':1}
numpix = 64
seed = 10

# Define some general image kwargs for the dataset
mask_radius = 0.2
mag_cut = 1.0

# Define arguments that will be used multiple times
output_ab_zeropoint = 25.127

# Define the cosmos path
cosmos_folder = './test_data/cosmos/'

config_dict = {
	'subhalo':{
		'class': SubhalosDG19,
		'parameters':{
			'sigma_sub':uniform(loc=0,scale=5e-4).rvs,
			'shmf_plaw_index':-1.83,
			'm_pivot': 1e10,'m_min': 1e8,'m_max': 1e10,
			'c_0':18,'conc_zeta':-0.2,'conc_beta':0.8,
			'conc_m_ref': 1e8,'dex_scatter': 0.1, 'k1':0.0,
			'k2':0.0
		}
	},
	'los':{
		'class': LOSDG19,
		'parameters':{
			'm_min':1e9,'m_max':1e10,'z_min':0.01,
			'dz':0.01,'cone_angle':8.0,'r_min':0.5,'r_max':10.0,
			'c_0':18,'conc_zeta':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.1,'delta_los':uniform(loc=0,scale=5e-3).rvs,
			'alpha_dz_factor':20.0
		}
	},
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'M200': 1e13,
			'z_lens': 0.5,
			'gamma': lognorm(scale=2.01,s=0.1).rvs,
			'theta_E': lognorm(scale=1.1,s=0.05).rvs,
			'e1,e2': multivariate_normal(np.zeros(2),
				np.array([[0.03,0.015],[0.015,0.03]])).rvs,
			'center_x': norm(loc=0.0,scale=0.16).rvs,
			'center_y': norm(loc=0.0,scale=0.16).rvs,
			'gamma1': norm(loc=0.0,scale=0.05).rvs,
			'gamma2': norm(loc=0.0,scale=0.05).rvs,
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'source':{
		'class': COSMOSCatalog,
		'parameters':{
			'z_source':1.5,
			'cosmos_folder':cosmos_folder,'max_z':None,
			'minimum_size_in_pixels':None,'faintest_apparent_mag':None,
			'smoothing_sigma':0.0,'random_rotation':False,
			'min_flux_radius':None,'output_ab_zeropoint':25.95,
			'center_x':norm(loc=0.0,scale=0.16).rvs,
			'center_y':norm(loc=0.0,scale=0.16).rvs}
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
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':5400.0,'sky_brightness':22,
			'num_exposures':1, 'background_noise':None
		}
	},
	'cross_object':{
		'parameters':{
			'los:delta_los,subhalo:sigma_sub':tmn
		}
	}
}
