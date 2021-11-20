# Configuration to reproduce H0RTON dataset

import numpy as np
from scipy.stats import norm, uniform, truncnorm
import manada.Sampling.distributions as dist
from manada.MainDeflector.simple_deflectors import PEMDShear
from manada.Sources.sersic import SingleSersicSource
from manada.Sources.cosmos import COSMOSCatalog
from manada.PointSource.single_point_source import SinglePointSource
from astropy.io import fits
from lenstronomy.Util import kernel_util
import manada

# define constants used across the config
output_ab_zeropoint = 25.9463
cosmos_folder = '/Users/smericks/Desktop/StrongLensing/COSMOS_23.5_training_sample/'
numpix = 64

# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}


# prepare psf (from baobab)
kernel_size=91
root_path = manada.__path__[0][:-7]
psf_path = root_path + '/datasets/hst_psf/psf_101.fits'
psf_map = fits.getdata(psf_path)
kernel_cut = kernel_util.cut_psf(psf_map, kernel_size)

# define general image kwargs
mag_cut = 2.0

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': truncnorm(-2.5,np.inf,loc=0.5,scale=0.2).rvs,
			'gamma': norm(loc=2.0,scale=0.1).rvs,
			'theta_E': truncnorm(-6.0,np.inf,loc=1.1,scale=0.1).rvs,
			'e1,e2': dist.EllipticitiesTranslation(q_dist=
                truncnorm(-2.666,2.,loc=0.7,scale=0.15).rvs,
                phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'center_x':None,
			'center_y':None,
			'gamma1,gamma2': dist.ExternalShearTranslation(gamma_dist=
				uniform(loc=0.,scale=0.05).rvs, 
				phi_dist=uniform(loc=-np.pi/2,scale=np.pi).rvs),
			'ra_0':0.0,
			'dec_0':0.0,
		}
	},
	'source':{
		'class': COSMOSCatalog,
		'parameters':{
			'z_source':None,
			'cosmos_folder':cosmos_folder,
			'max_z':1.0,'minimum_size_in_pixels':64,'min_apparent_mag':20,
			'smoothing_sigma':0.08,'random_rotation':True,
			'output_ab_zeropoint':output_ab_zeropoint,
			'min_flux_radius':10.0,
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
			'e1,e2':dist.EllipticitiesTranslation(q_dist=
                truncnorm(-np.inf,1.,loc=0.85,scale=0.15).rvs,
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
			'kappa_ext': dist.KappaTransformDistribution(n_dist=
                norm(loc=1.,scale=0.025).rvs),
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
			'kernel_point_source':kernel_cut
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
    'cross_object':{
        'parameters':{
            'main_deflector:center_x,main_deflector:center_y,lens_light:center_x,lens_light:center_y': 
            dist.DuplicateXY(x_dist=norm(loc=0,scale=0.07).rvs,
            y_dist=norm(loc=0,scale=0.07).rvs),
            'source:center_x,source:center_y,point_source:x_point_source,point_source:y_point_source': 
            dist.DuplicateXY(x_dist=uniform(loc=-0.2,scale=0.4).rvs,
            y_dist=uniform(loc=-0.2,scale=0.4).rvs),
			'main_deflector:z_lens,source:z_source':dist.RedshiftsTruncNorm(
				z_lens_min=-2.5,z_lens_mean=0.5,z_lens_std=0.2,
				z_source_min=-5,z_source_mean=2,z_source_std=0.4)
        }
    }
}