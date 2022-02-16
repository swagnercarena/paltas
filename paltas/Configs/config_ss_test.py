from paltas.Configs.config_d_los_sigma_sub import *
from paltas.Sources.cosmos import COSMOSCatalog
from paltas.Substructure.los_dg19 import LOSDG19
from paltas.Substructure.subhalos_dg19 import SubhalosDG19
from paltas.MainDeflector.simple_deflectors import PEMDShear
from scipy.stats import norm

config_dict = {
	'subhalo':{
		'class': SubhalosDG19,
		'parameters':{
			'sigma_sub':norm(loc=1.3e-1,scale=1e-2).rvs,
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
			'delta_los':1,
			'm_min':1e7,'m_max':1e10,'z_min':0.01,
			'dz':0.01,'cone_angle':8.0,'r_min':0.5,'r_max':10.0,
			'c_0':18,'conc_zeta':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.1,'alpha_dz_factor':5.0
		}
	},
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'M200': 1e13,
			'z_lens': 0.5,
			'gamma': norm(loc=2.0,scale=0.02).rvs,
			'theta_E': norm(loc=1.1,scale=0.02).rvs,
			'e1': norm(loc=0.0,scale=0.05).rvs,
			'e2': norm(loc=0.0,scale=0.05).rvs,
			'center_x': norm(loc=0.0,scale=0.05).rvs,
			'center_y': norm(loc=0.0,scale=0.05).rvs,
			'gamma1': norm(loc=0.0,scale=0.05).rvs,
			'gamma2': norm(loc=0.0,scale=0.05).rvs,
			'ra_0':0.0, 'dec_0':0.0
		}
	},
	'source':{
		'class': COSMOSCatalog,
		'parameters':{
			'z_source':1.5,'cosmos_folder':cosmos_folder,'max_z':0.05,
			'minimum_size_in_pixels':650,'faintest_apparent_mag':20,
			'smoothing_sigma':0.00,'random_rotation':True,
			'center_x':norm(loc=0.0,scale=0.16).rvs,
			'center_y':norm(loc=0.0,scale=0.16).rvs,'min_flux_radius':10.0}
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
