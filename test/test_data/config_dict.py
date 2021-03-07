# A test configuration dict

from manada.Sampling import distributions
import numpy as np
from scipy.stats import uniform, norm, loguniform, lognorm, multivariate_normal
from manada.Substructure.los_dg19 import LOSDG19
from manada.Substructure.subhalos_dg19 import SubhalosDG19
from manada.Sources.cosmos import COSMOSCatalog
import manada

# Define a multivariate distribution we'll use
mean = np.ones(2)
cov = np.array([[1.0,0.7],[0.7,1.0]])
min_values = np.zeros(2)
tmn = distributions.TruncatedMultivariateNormal(mean,cov,min_values,None)

# Define the numerics kwargs
kwargs_numerics = {'supersampling_factor':1}
numpix = 64
seed = 10

# Define some general image kwargs for the dataset
mask_radius = 0.2
mag_cut = 1.0

# Define the cosmos path
root_path = manada.__path__[0][:-7]
cosmos_folder = root_path + '/test/test_data/cosmos/'

config_dict = {
	'subhalo':{
		'class': SubhalosDG19,
		'parameters':{
			'sigma_sub':uniform(loc=0,scale=5e-4).rvs,
			'shmf_plaw_index':-1.83,
			'm_pivot': 1e8,'m_min': 1e9,'m_max': 1e10,
			'c_0':18,'conc_xi':-0.2,'conc_beta':0.8,
			'conc_m_ref': 1e8,'dex_scatter': 0.1
		}
	},
	'los':{
		'class': LOSDG19,
		'parameters':{
			'm_min':1e9,'m_max':1e10,'z_min':0.01,
			'dz':0.01,'cone_angle':8.0,'r_min':0.5,'r_max':10.0,
			'c_0':18,'conc_xi':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.1,'delta_los':uniform(loc=0,scale=5e-3).rvs,
			'alpha_dz_factor':20.0
		}
	},
	'main_deflector':{
		'models': ['PEMD','SHEAR'],
		'parameters':{
			'M200': loguniform(a=1e11,b=5e13).rvs,
			'z_lens': 0.5,
			'gamma': lognorm(scale=2.01,s=0.1).rvs,
			'theta_E': lognorm(scale=1.1,s=0.05).rvs,
			'e1,e2': multivariate_normal(np.zeros(2),
				np.array([[1,0.5],[0.5,1]])).rvs,
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
			'z_source':1.5,'catalog_i':5,'cosmos_folder':cosmos_folder,
			'max_z':None,'minimum_size_in_pixels':None,'min_apparent_mag':None,
			'smoothing_sigma':0.0}
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
	},
	'cross_object':{
		'parameters':{
			'los:delta_los,subhalo:sigma_sub':tmn
		}
	}
}
