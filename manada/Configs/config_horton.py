# Configuration to reproduce H0RTON dataset

import numpy as np
from scipy.stats import norm, uniform, truncnorm
import manada.Sampling.distributions as dist
from manada.MainDeflector.simple_deflectors import PEMDShear
from manada.Sources.sersic import SingleSersicSource
from manada.PointSource.single_point_source import SinglePointSource
from astropy.io import fits
from lenstronomy.Util import kernel_util
import manada

output_ab_zeropoint = 25.9463
# Define the numerics kwargs.
kwargs_numerics = {'supersampling_factor':1}

# The number of pixels in the CCD.
numpix = 64

# prepare psf (from baobab)
kernel_size=91
root_path = manada.__path__[0][:-7]
psf_path = root_path + '/datasets/hst_psf/psf_101.fits'
psf_map = fits.getdata(psf_path)
kernel_cut = kernel_util.cut_psf(psf_map, kernel_size)


config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': truncnorm(2.5,np.inf,loc=0.5,scale=0.2).rvs,
			'gamma': norm(loc=2.0,scale=0.1).rvs,
			'theta_E': norm(loc=1.1,scale=0.1).rvs,
			'e1,e2': dist.EllipticitiesTranslation(q_dist=
                norm(loc=0.7,scale=0.15).rvs,
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
		'class': SingleSersicSource,
		'parameters':{
			'z_source':truncnorm(5,np.inf,loc=2.,scale=0.4).rvs,
            'magnitude':uniform(loc=20,scale=5).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':norm(loc=0.35,scale=0.05).rvs,
			'n_sersic':norm(loc=3.,scale=0.5).rvs,
			'e1,e2':dist.EllipticitiesTranslation(q_dist=
                norm(loc=0.6,scale=0.15).rvs,
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
			'R_sersic':norm(loc=0.8,scale=0.15).rvs,
			'n_sersic':norm(loc=3,scale=0.55).rvs,
			'e1,e2':dist.EllipticitiesTranslation(q_dist=
                norm(loc=0.85,scale=0.15).rvs,
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
			'mag_pert':norm(1,0.2).rvs(size=5),
			'compute_time_delays':True,
			'kappa_ext': dist.KappaTransformDistribution(n_dist=
                norm(loc=1.,scale=0.025).rvs)
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
				z_lens_min=2.5,z_lens_mean=0.5,z_lens_std=0.2,
				z_source_min=5,z_source_mean=2,z_source_std=0.4)
        }
    }
}