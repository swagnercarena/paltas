# -*- coding: utf-8 -*-
"""
Define a number of distributions that can be used by the sampler to draw
lensing parameters.

This module contains classes that define distributions that can be effecitvely
sampled from.
"""
import numpy as np
from scipy.stats import truncnorm, uniform
from astropy.io import fits
from lenstronomy.Util import kernel_util


class MultivariateLogNormal():
	"""Class for drawing multiple parameters from a multivariate log normal
	distribution.

	Args:
		mean (np.array): The mean value of the log normal distribution
		covariance (np.array): The covariance matrix of the distribution

	Notes:
		Returned values will follow the distribution exp(normal(mu,covariance))
	"""

	def __init__(self,mean,covariance):
		# Make sure the dimension of the mean allows for easy matrix
		# operations
		if len(mean.shape)==1:
			mean = np.expand_dims(mean,axis=0)
		self.mean = mean
		self.covariance = covariance
		self.L = np.linalg.cholesky(self.covariance)

	def __call__(self,n=1):
		"""Returns n draws from the distribution

		Args:
			n (int): The number of draws.

		Returns
			(np.array): n x len(mean) array of draws.
		"""
		rand_draw = np.random.randn(self.mean.shape[1]*n).reshape(
			(self.mean.shape[1],n))
		return np.exp(self.mean.T+np.dot(self.L,rand_draw)).T


class TruncatedMultivariateNormal():
	"""Class for drawing multiple parameters from a truncated multivariate
	normal distribution.

	Args:
		mean (np.array): The mean value of the log normal distribution
		covariance (np.array): The covariance matrix of the distribution
		min_values (np.array): The minimum value for each parameter
		max_values (np.array): The maximum value for each parameter

	Notes:
		Note this code uses rejection sampling, which may be slow if most
		of the multivariate normal's pdf is outside the bounds. Be careful when
		using this in high dimensions.
	"""

	def __init__(self,mean,covariance,min_values=None,max_values=None):
		# Make sure that each of the n-dimensional inputs follows
		# the desired shape for matrix calculations
		if len(mean.shape)==1:
			mean = np.expand_dims(mean,axis=0)
		# If none for min_values, set to negative infinity
		if min_values is None:
			min_values = np.ones(mean.shape) * -np.inf
		elif len(min_values.shape)==1:
			min_values = np.expand_dims(min_values,axis=0)
		# If none for max_values, set to positive infinity
		if max_values is None:
			max_values = np.ones(mean.shape) * np.inf
		elif len(max_values.shape)==1:
			max_values = np.expand_dims(max_values,axis=0)

		self.mean = mean
		self.covariance = covariance
		self.L = np.linalg.cholesky(self.covariance)
		self.min_values = min_values
		self.max_values = max_values

	def __call__(self,n=1):
		"""Returns n draws from the distribution

		Args:
			n (int): The number of draws.

		Returns
			(np.array): n x len(mean) array of draws.
		"""
		# Start with a regular draw
		n_accepted = 0
		n_samp = n
		keep_draws = np.zeros((n,self.mean.shape[1]))
		rand_draw = np.random.randn(self.mean.shape[1]*n_samp).reshape(
			(self.mean.shape[1],n_samp))
		draws = (self.mean.T+np.dot(self.L,rand_draw)).T

		# Check which draws are within our bounds
		keep_ind = np.prod(draws > self.min_values,axis=-1,dtype=np.bool)
		keep_ind *= np.prod(draws < self.max_values,axis=-1,dtype=np.bool)
		keep_draws[n_accepted:n_accepted+np.sum(keep_ind)] = draws[keep_ind]
		n_accepted += np.sum(keep_ind)

		# Keep drawing until we have enough samples.
		while n_accepted<n:
			# Draw
			rand_draw = np.random.randn(self.mean.shape[1]*n_samp).reshape(
				(self.mean.shape[1],n_samp))
			draws = (self.mean.T+np.dot(self.L,rand_draw)).T
			# Check for the values in bounds
			keep_ind = np.prod(draws > self.min_values,axis=-1,dtype=np.bool)
			keep_ind *= np.prod(draws < self.max_values,axis=-1,dtype=np.bool)
			# Only use the ones we need
			use_keep = np.minimum(n-n_accepted,np.sum(keep_ind))
			keep_draws[n_accepted:n_accepted+use_keep] = (
				draws[keep_ind][:use_keep])
			n_accepted += use_keep

		# Remove first dimension for the n=1 case
		if n==1:
			keep_draws = np.squeeze(keep_draws)

		return keep_draws


class EllipticitiesTranslation():
	"""Class that takes in distributions for q_lens and phi_lens, returns 
	samples of e1 and e2 correspondingly
	
	Args: 
		q_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			axis ratio (can be callable or constant)
		phi_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			orientation angle in radians (can be callable or constant)

	Notes: 

	"""
	def __init__(self,q_dist,phi_dist):
		self.q_dist = q_dist
		self.phi_dist = phi_dist

	def __call__(self):
		"""Returns a sample of e1,e2 

		Returns:
			(float,float): samples of x-direction ellipticity 
				eccentricity, xy-direction ellipticity eccentricity
		"""

		if callable(self.q_dist):
			q = self.q_dist()
		else:
			q = self.q_dist
		if callable(self.phi_dist):
			phi = self.phi_dist()
		else:
			phi = self.phi_dist
		
		e1 = (1 - q)/(1+q) * np.cos(2*phi)
		e2 = (1 - q)/(1+q) * np.sin(2*phi)

		return e1,e2


class ExternalShearTranslation():
	"""Class that maps samples of gamma_ext, phi_ext distributions to 
		gamma1, gamma2
	
	Args: 
		gamma_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			external shear modulus (callable or constant) 
		phi_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			orientation angle in radians (callable or constant)
	
	Notes:
	"""

	def __init__(self, gamma_dist,phi_dist):
		self.gamma_dist = gamma_dist
		self.phi_dist = phi_dist
	
	def __call__(self):
		"""Returns gamma1, gamma2 samples

		Returns:
			(float,float): samples of external shear coordinate values 
		"""
		if callable(self.gamma_dist):
			gamma = self.gamma_dist()
		else:
			gamma = self.gamma_dist
		if callable(self.phi_dist):
			phi = self.phi_dist()
		else:
			phi = self.phi_dist
		
		gamma1 = gamma * np.cos(2*phi)
		gamma2 = gamma * np.sin(2*phi)
		
		return gamma1,gamma2


class KappaTransformDistribution():
	"""Class that samples Kext given 1 / (1-Kext) ~ n. n is sampled from a 
	distribution given by n_dist, then Kext is computed given the
	transformation
	
	Args: 
		n_dist (scipy.stats.rv_continuous.rvs or float): distribution for 
			1 / (1-Kext) (can be callable or constant)
	"""

	def __init__(self,n_dist):
		self.n_dist = n_dist

	def __call__(self):
		"""Samples 1/(1-Kext), then maps that sample to Kext value

		Returns: 
			(float): Kext sample
		"""
		if callable(self.n_dist):
			n = self.n_dist()
		else:
			n = self.n_dist
		
		return 1 - (1/n)


class Duplicate():
	"""Class that returns two copies of the same random draw.

	Args:
		dist (scipy.stats.rv_continuous.rvs or float): The distribution to
			draw the sample from.
	"""

	def __init__(self,dist):
		self.dist = dist

	def __call__(self):
		"""Returns two copies of the same sample

		Returns
			(float,float): Two copies of the sample.
		"""

		if callable(self.dist):
			samp = self.dist()
		else:
			samp = self.dist

		return samp,samp


class DuplicateScatter(Duplicate):
	"""Class that returns two copies of the same random draw with some
	additional scatter on the second value.

	Args:
		dist (scipy.stats.rv_continuous.rvs or float): The distribution to
			draw the sample from.
		scatter (float): The additional scatter to add to the second draw
			of the variable.
	"""

	def __init__(self,dist,scatter):
		self.dist = dist
		self.scatter = scatter

	def __call__(self):
		"""Returns two copies of the same sample with additional
		scatter on the second sample.

		Returns
			(float,float): The two samples.
		"""
		# Get the samples from the super function and add the scatter.
		samp,samp = super().__call__()
		return samp,samp+np.random.randn()*self.scatter


class DuplicateXY():
	"""Class that returns two copies of x, y coordinates drawn from 
		distributions

	Args: 
		x_dist (scipy.stats.rv_continuous.rvs or float): distribution for x 
			(can be callable or constant)
		y_dist (scipy.stats.rv_continuous.rvs or float): distribution for y 
			(can be callable or constant)
	"""

	def __init__(self,x_dist,y_dist):
		self.x_dist = x_dist
		self.y_dist = y_dist
	
	def __call__(self):
		"""Returns two copies of x,y sample

		Returns
			(float,float,float,float): Two copies of x,y sampled from x_dist
				and y_dist
		"""

		if callable(self.x_dist):
			x = self.x_dist()
		else:
			x = self.x_dist
		if callable(self.y_dist):
			y = self.y_dist()
		else:
			y = self.y_dist
		
		return x,y,x,y

class FourComponentCorrelatedCenter():
	"""
    Sample center coordinates for lens mass, lens light, source light, ps light
        given an initial distribution for lens mass center, 
	    scatter on distance to lens light, 
	    scatter on distance to source/ps light
	    
	Note: assumes source and point source at same location
	    
	Args: 
        lm_dist (callable or float): Distribution to pull lens mass coordinates from
		ll_sigma (float): Gaussian sigma for distance from lens mass to lens light
		src_sigma (float): Gaussian sigma for distance from lens mass to source light

	Returns:
		x_lm,y_lm,x_ll,y_ll,x_src,y_src,x_ps,y_ps
    """
	
	def __init__(self,lm_dist,ll_sigma,src_sigma):
		self.lm_dist = lm_dist
		self.ll_sigma = ll_sigma
		self.src_sigma = src_sigma

	def _calc_xy_scatter(self):
		"""
		Given a Gaussian sigma for distance between center and componenet, 
		draw a radius from that Gaussian truncated & centered at zero. 
		Also draw random angle between 0,2pi. Then, return corresponding x,y
		components.

		Returns:
			x_scat_ll, y_scat_ll, x_scat_src, y_scat_src (Scatter to be added
				to lens mass coordinate)
		"""

		R_ll = truncnorm.rvs(0,np.inf,loc=0.0,scale=self.ll_sigma)
		phi_ll = uniform.rvs(0,2*np.pi)
		R_src = truncnorm.rvs(0,np.inf,loc=0.0,scale=self.ll_sigma)
		phi_src = uniform.rvs(0,2*np.pi)
		return R_ll*np.cos(phi_ll),R_ll*np.sin(phi_ll),R_src*np.cos(phi_src),R_src*np.sin(phi_src)

	def __call__(self):

		if callable(self.lm_dist):
			x_lm = self.lm_dist()
			y_lm = self.lm_dist()
		else:
			x_lm = self.lm_dist
			y_lm = self.lm_dist

		x_scat_ll, y_scat_ll, x_scat_src, y_scat_src = self._calc_xy_scatter()
		x_ll = x_lm + x_scat_ll
		y_ll = y_lm + y_scat_ll
		x_src = x_lm + x_scat_src
		y_src = y_lm + y_scat_src

		return x_lm,y_lm,x_ll,y_ll,x_src,y_src,x_src,y_src
	

class RedshiftsTruncNorm():
	"""Class that samples z_lens and z_source from truncated normal 
		distributions, forcing z_source > z_lens to be true

	Args: 
		z_lens_min (float): minimum allowed lens redshift
		z_lens_mean (float): lens redshift mean
		z_lens_std (float): lens redshift standard deviation
		z_source_min (float): minimum allowed source redshift
		z_source_mean (float): source redshift mean
		z_source_std (float): source redshift standard deviation
	"""

	def __init__(self, z_lens_min,z_lens_mean,z_lens_std,z_source_min,
		z_source_mean,z_source_std):
		# transform z_lens_min, z_source_min to be in units of std. deviations
		self.z_lens_min = (z_lens_min - z_lens_mean) / z_lens_std 
		self.z_source_min = (z_source_min - z_source_mean) / z_source_std
		# define truncnorm dist for lens redshift
		self.z_lens_dist = truncnorm(self.z_lens_min,np.inf,loc=z_lens_mean,
			scale=z_lens_std).rvs
		# save z_source info
		self.z_source_mean = z_source_mean
		self.z_source_std = z_source_std
	
	def __call__(self):
		"""Returns samples of redshifts, ensuring z_source > z_lens

		Returns: 
			(float,float): z_lens,z_source 
		"""
		z_lens = self.z_lens_dist()
		clip = (z_lens - self.z_source_mean) / self.z_source_std
		# number of std. devs away to stop (negative)
		if(clip > self.z_source_min):
			self.z_source_min = clip
		# define truncnorm dist for source redshift
		z_source = truncnorm(self.z_source_min,np.inf,self.z_source_mean,
			self.z_source_std).rvs()

		return z_lens,z_source


class RedshiftsLensLight(RedshiftsTruncNorm):
	"""Class that samples z_lens, z_lens_light, and z_source from truncated
		normal distributions, forcing z_source > z_lens to be true and
		z_lens_light = z_lens.

	Args:
		z_lens_min (float): minimum allowed lens redshift
		z_lens_mean (float): lens redshift mean
		z_lens_std (float): lens redshift standard deviation
		z_source_min (float): minimum allowed source redshift
		z_source_mean (float): source redshift mean
		z_source_std (float): source redshift standard deviation
	"""

	def __call__(self):
		"""Returns samples of redshifts, ensuring z_source > z_lens and
		z_lens_light = z_lens.

		Returns:
			(float,float): z_lens,z_lens_light,z_source
		"""
		z_lens,z_source = super().__call__()
		return z_lens,z_lens,z_source


class RedshiftsPointSource(RedshiftsTruncNorm):
    """Class that samples z_lens, z_lens_light, z_source, and z_point_source 
        from truncated normal distributions, forcing z_source > z_lens to be 
        true and z_lens_light = z_lens, z_source = z_point_source

	Args:
		z_lens_min (float): minimum allowed lens redshift
		z_lens_mean (float): lens redshift mean
		z_lens_std (float): lens redshift standard deviation
		z_source_min (float): minimum allowed source redshift
		z_source_mean (float): source redshift mean
		z_source_std (float): source redshift standard deviation
	"""

    def __call__(self):
        """Returns samples of redshifts, ensuring z_source > z_lens and
        z_lens_light = z_lens.

		Returns:
			(float,float): z_lens,z_lens_light,z_source
		"""
        z_lens,z_source = super().__call__()
        return z_lens,z_lens,z_source,z_source


class MultipleValues():
	"""Class to call dist.rvs(size=num)

	Args:
		dist (scipy.stats.rv_continuous.rvs): callable distribution
		num (int): number of samples to return in one call
	"""

	def __init__(self, dist, num):
		self.dist = dist
		self.num = num
	
	def __call__(self):
		"""Returns specified # of samples from dist
		
		Returns: 
			list(float): |num| samples from dist
		"""
		return self.dist(size=self.num)


# PSF Generator

def xy_helper(x,y):
	# 1st 7 are bottom row, 2nd 7 are 2nd row, etc...
	return int(((y%4)*7)-1 + x%7)


class PSFGenerator():
	"""
	Args:
		psf_filepath: .fits file that stores PSFs
		coords_dict dict{scipy.stats.rv_continuous.rvs}: dict of 
			distributions to sample for coordinates 
		psf_option (string): option for how PSF file is handled

    Notes:
        psf_option 'emp_f814w' requires coord_dict keys: 'CCD', 'x', 'y', 'focus'
	"""

	def __init__(self,psf_filepath,coords_dict,psf_option='emp_f814w'):
		self.psf_filepath = psf_filepath
		with fits.open(self.psf_filepath) as hdu:
			self.psf_kernels = hdu[0].data
		self.coords_dict = coords_dict
		self.psf_option = psf_option

	def hst_emp_f814w_mapping(self,coords):
		"""
		Args: 
			coords dict{float}: Dict of coordinate values: [CCD,x,y,focus]
		"""
		if not all(key in coords.keys() for key in {'CCD','x','y','focus'}):
			raise ValueError('emp_f814w option expects 4 coordinates: CCD,x,y,focus')

		# array has size (10,56,101,101)
		if coords['CCD'] == 1:
			psf_ccd = self.psf_kernels[:,0:28,:,:]
		elif coords['CCD'] == 2:
			psf_ccd = self.psf_kernels[:,28:56,:,:]
		else: 
			raise ValueError('invalid CCD coordinate in PSF interpolation: %d'%(coords['CCD']))
		# now, array has size (10,28,101,101)
		psf_ccd[psf_ccd < 0] = 0
		x = coords['x']
		y = coords['y']
		f = coords['focus']

		# check values of x, y, & f before proceeding
		if x <= 4096/14 or x >= (4096 - 4096/14):
			raise ValueError('invalid x coordinate in PSF interpolation: %d'%(x))
		if y <=  2051/8 or y >= (2051 - 2051/8):
			raise ValueError('invalid y coordinate in PSF interpolation: %d'%(y))
		if f < 1 or f > 10:
			raise ValueError('invalid focus coordinate in PSF interpolation: %.2f'%(f))
		# pick the 8 points you need
		# pick x below/above, y below/above, f below/above
		# ex: how many times does 4096/7 fit into x value
		x = (x-4096/14) / (4096/7)
		x_below = np.floor(x)
		x_above = x_below+1
		y = (x-2051/8) / (2051/4)
		y_below = np.floor(y)
		y_above = y_below+1
		f_below = int(np.floor(f))
		f_above = f_below+1
		# do the 3D interpolation
		# source: https://en.wikipedia.org/wiki/Trilinear_interpolation
		xd = (x - x_below)/(x_above-x_below)
		print('xd: ', xd)
		yd = (y - y_below)/(y_above-y_below)
		print('yd: ', yd)
		fd = (f - f_below)/(f_above-f_below)
		print('fd: ', fd)
		
		c00 = (psf_ccd[f_below-1,xy_helper(x_below,y_below),:,:]*(1-xd) + 
			psf_ccd[f_below-1,xy_helper(x_above,y_below),:,:]*xd)
		c01 = (psf_ccd[f_above-1,xy_helper(x_below,y_below),:,:]*(1-xd) + 
			psf_ccd[f_above-1,xy_helper(x_above,y_below),:,:]*xd)
		c10 = (psf_ccd[f_below-1,xy_helper(x_below,y_above),:,:]*(1-xd) + 
			psf_ccd[f_below-1,xy_helper(x_above,y_above),:,:]*xd)
		c11 = (psf_ccd[f_above-1,xy_helper(x_below,y_above),:,:]*(1-xd) + 
			psf_ccd[f_above-1,xy_helper(x_above,y_above),:,:]*xd)

		c0 = c00*(1-yd) + c10*yd
		c1 = c01*(1-yd) + c11*yd
		
		c = c0*(1-fd) + c1*fd
		print('less than 0: ', np.sum(c<0))
		
		return kernel_util.degrade_kernel(c,4)


	def __call__(self):

		# sample coordinates
		coords = {}
		for k in self.coords_dict.keys():
			# check if callable
			if callable(self.coords_dict[k]):
				coords[k] = self.coords_dict[k]()
			else:
				coords[k] = self.coords_dict[k]
		# map coordinates to a PSF
		if self.psf_option == 'emp_f814w':
			final_psf = self.hst_emp_f814w_mapping(coords)
		return final_psf
