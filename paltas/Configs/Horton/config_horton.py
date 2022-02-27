# Configuration to reproduce H0RTON dataset

import numpy as np
from scipy.stats import norm, uniform, truncnorm
import paltas.Sampling.distributions as dist
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource
from astropy.io import fits
from lenstronomy.Util import kernel_util
import paltas

# define constants used across the config
output_ab_zeropoint = 25.9463

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}

# The number of pixels in the CCD.
numpix = 64

# prepare psf (from baobab)
kernel_size=91
root_path = paltas.__path__[0][:-7]
psf_path = root_path + '/datasets/hst_psf/psf_101.fits'
psf_map = fits.getdata(psf_path)
kernel_cut = kernel_util.cut_psf(psf_map, kernel_size)

# define general image kwargs
mag_cut = 2.0

# Note that some parameters in the config dict are set to None
# because they will be sampled by the cross_object dict.
config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': truncnorm(-2.5,np.inf,loc=0.5,scale=0.2).rvs,
			'gamma': norm(loc=2.0,scale=0.1).rvs,
			'theta_E': truncnorm(-6.0,np.inf,loc=1.1,scale=0.1).rvs,
			'e1,e2': dist.EllipticitiesTranslation(
				q_dist=truncnorm(-2.666,2.,loc=0.7,scale=0.15).rvs,
				phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'center_x':None,
			'center_y':None,
			'gamma1,gamma2': dist.ExternalShearTranslation(
				gamma_dist=uniform(loc=0.,scale=0.05).rvs,
				phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'ra_0':0.0,
			'dec_0':0.0,
		}
	},
	'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':truncnorm(-5,np.inf,loc=2.,scale=0.4).rvs,
			'magnitude':uniform(loc=20,scale=5).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-2,2,loc=0.35,scale=0.05).rvs,
			'n_sersic':truncnorm(-6.,np.inf,loc=3.,scale=0.5).rvs,
			'e1,e2':dist.EllipticitiesTranslation(
				q_dist=truncnorm(-3.,4.,loc=0.6,scale=0.1).rvs,
				phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'center_x':None,
			'center_y':None}

	},
	'lens_light':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':None,
			'magnitude':uniform(loc=19,scale=2).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-1.333,np.inf,loc=0.8,scale=0.15).rvs,
			'n_sersic':truncnorm(-2.,np.inf,loc=3,scale=0.5).rvs,
			'e1,e2':dist.EllipticitiesTranslation(
				q_dist=truncnorm(-np.inf,1.,loc=0.85,scale=0.15).rvs,
				phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'center_x':None,
			'center_y':None}
	},
	'point_source':{
		'class': SinglePointSource,
		'parameters':{
			'x_point_source':None,
			'y_point_source':None,
			'magnitude':uniform(loc=20,scale=2.5).rvs,
			'output_ab_zeropoint':25.127,
			'mag_pert': dist.MultipleValues(dist=norm(1,0.1).rvs,num=5),
			'compute_time_delays':True,
			'kappa_ext': dist.KappaTransformDistribution(
				n_dist=norm(loc=1.,scale=0.025).rvs),
			'time_delay_error': dist.MultipleValues(
				dist=norm(loc=0.,scale=0.25).rvs, num=5)
		}
	},
	'lens_equation_solver':{
		'parameters':{
			'min_distance':0.05,
			'search_window':numpix*0.08,
			'num_iter_max':100,
			'precision_limit':10**(-10)
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
			'kernel_point_source':kernel_cut,
			'point_source_supersampling_factor':2
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.08,'ccd_gain':2.5,'read_noise':4.0,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':5400.,'sky_brightness':22.,
			'num_exposures':1,'background_noise':None
		}
	},
	'drizzle':{
		'parameters':{
			'supersample_pixel_scale':0.040,'output_pixel_scale':0.060,
			'wcs_distortion':None,
			'offset_pattern':[(0,0),(0.5,0),(0.0,0.5),(-0.5,-0.5)],
			'psf_supersample_factor':2
		}
	},
	'cross_object':{
		'parameters':{
			('main_deflector:center_x,main_deflector:center_y,lens_light:'+
				'center_x,lens_light:center_y'):
			dist.DuplicateXY(x_dist=norm(loc=0,scale=0.07).rvs,
			y_dist=norm(loc=0,scale=0.07).rvs),
			('source:center_x,source:center_y,point_source:x_point_source,'+
				'point_source:y_point_source'):
			dist.DuplicateXY(x_dist=uniform(loc=-0.2,scale=0.4).rvs,
			y_dist=uniform(loc=-0.2,scale=0.4).rvs),
			'main_deflector:z_lens,source:z_source':dist.RedshiftsTruncNorm(
				z_lens_min=0,z_lens_mean=0.5,z_lens_std=0.2,
				z_source_min=0,z_source_mean=2,z_source_std=0.4)
		}
	}
}
