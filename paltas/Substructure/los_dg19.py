# -*- coding: utf-8 -*-
"""
Define the class to draw line of sight substructure for a lens according to
https://arxiv.org/pdf/1909.02573.pdf

This module contains the functions needed to turn the parameters of the
los halo distribution into masses, concentrations, and positions.
"""
from .los_base import LOSBase
import numba
import numpy as np
from colossus.lss import peaks, bias
from ..Utils import power_law, cosmology_utils
import functools
from . import nfw_functions
import lenstronomy.Util.util as util
from lenstronomy.LensModel.Profiles.nfw import NFW
from scipy.signal import fftconvolve


class LOSDG19(LOSBase):
	"""Class for rendering the line of sight structure according to DG19.

	Args:
		los_parameters (dict): A dictionary containing the type of
			los distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		source_parameters (dict): A dictionary containing the type of the
			source and the value for each of its parameters.
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).

	Notes:

	Required Parameters

	- delta_los - mass function normalization
	- m_min - minimum rendered mass in units of M_sun
	- m_max - maximum rendered mass in units of M_sun
	- z_min - minimum redshift at which to render los halos
	- dz - redshift bin width
	- cone_angle - cone opening angle for rendering los halos
	- c_0 - concentration normalization
	- conc_zeta - concentration redshift power law slope
	- conc_beta - concentration peak height power law slope
	- conc_m_ref - concentration peak height pivot mass
	- dex_scatter - scatter in concentration in units of dex
	- alpha_dz_factor- deflection angle correction redshift bin width
	"""
	# Define the parameters we expect to find for the DG_19 model
	required_parameters = ('m_min','m_max','z_min','dz','cone_angle',
		'r_max','r_min','c_0','conc_zeta','conc_beta','conc_m_ref',
		'dex_scatter','delta_los','alpha_dz_factor')

	def __init__(self,los_parameters,main_deflector_parameters,
		source_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(los_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

	@staticmethod
	@numba.njit
	def nu_f_nu(nu):
		"""Calculates nu f(nu) for the Sheth Tormen 2001 model.

		Args:
			nu (np.array): An array of nu values to at which to
				calculate nu_f_nu

		Returns:
			(np.array): The value of nu f(nu)

		"""
		# Parameters fit to simulations (A is fixed so that the integral
		# for all nu returns 1).
		A = 0.32218
		q = 0.3
		a = 0.707
		# Calculate the fundamental unit of our equation
		nu_2 = a*np.square(nu)
		# Finally calculate and return nu f(nu)
		nu_f_nu = (2*A*(1+nu_2**(-q))*np.sqrt(nu_2/(2*np.pi))*np.exp(-nu_2/2))
		return nu_f_nu

	def dn_dm(self,m,z):
		"""Returns the mass function at a given mass and redshift.

		Args:
			m (np.array): An array of the mass values at which to calculate
				the mass function in units of M_sun.
			z (np.array): Either one redshift at which to calculate the
				mass function or one redshift for each mass input.

		Returns:
			(np.array): The values of the mass function at each mass in units
			of physical number density in units of 1/(M_sun*kpc^3).

		Notes:
			It is slower to pass in multiple z values that are identical.
			If only a few redshift bins are being considered, run this
			function on each bin with a float redshift.
		"""
		# First we need to do some unit conversaion to the units used by the
		# nu_f_nu function. Note that these are all comoving values we're
		# using because colossus expects them, but nu is dimensionless.
		h = self.cosmo.h
		delta_c = peaks.collapseOverdensity(z=z,corrections=True)
		r = peaks.lagrangianR(m*h)
		sigma = self.cosmo.sigma(r,z)
		nu = delta_c/sigma

		# Calcualte our mass function from its parts
		nfn_eval = self.nu_f_nu(nu)
		d_ln_sigma_d_ln_r = self.cosmo.sigma(r,z,derivative=True)
		# Density is returned in units of M_sun*h^2/kpc^3
		rho_m = self.cosmo.rho_m(z)*h**2

		return -1/3*nfn_eval*rho_m/m**2*d_ln_sigma_d_ln_r

	@functools.lru_cache(maxsize=None)
	def power_law_dn_dm(self,z,m_min,m_max,n_dm=100):
		"""Returns the best fit power law parameters for the physical number
		density at a given redshift and mass range.

		Args:
			z (float): The redshift at which to calculate the power law
				parameters.
			m_min (float): The lower bound of the mass M_sun
			m_max (float): The upper bound of the mass M_sun
			n_dm (int): The number of dm samples to consider in the fit.

		Returns:
			(tuple): The power law slope and the norm for the power law in
			units of 1/M_sun/kpc**3
		"""
		m = np.logspace(np.log10(m_min),np.log10(m_max),n_dm)
		lm = np.log(m)
		dn_dm = self.dn_dm(m,z)
		ldn_dm = np.log(dn_dm)

		# The MLE estimate from the slope assuming Gaussian noise on the
		# log quantity
		slope_estimate = 1/n_dm * np.sum(ldn_dm) * np.sum(lm) - np.sum(
			ldn_dm*lm)
		slope_estimate /= -np.sum(lm*lm) + 1/n_dm*np.sum(lm)**2

		# The MLE estimate on the norm
		norm_estimate = np.exp(1/n_dm*np.sum(ldn_dm-slope_estimate*lm))

		return slope_estimate, norm_estimate

	def two_halo_boost(self,z,z_lens,dz,lens_m200,r_max,r_min,n_quads=100):
		"""Calculates the boost from the two halo term of the host halo at
		the given redshift.

		Args:
			z (float): The redshift to calculate the boost at.
			z_len (float): The redshift of the main deflector
			dz (float): The thickness of the redshift slice to consider
			lens_m200 (float): The mass of the host lens in units of
				M_sun with mass definition 200c.
			r_max (float): The maximum radius to calculate the correlation
				function out to.
			r_min (float): The minimum radius to consider the correlation
				funciton to (to avoid overlap with substructure draws).
			n_quads (int): The number of points to use in averaging the
				two halo term of the redshift slice.

		Returns
			(float): The boost at the given redshift.
		"""
		# Get a range of z values we will average over.
		z_range = np.linspace(z,z+dz,n_quads)
		# Only consider the two point function in the regime where it is
		# large and we are outside the virial radius of the host halo.
		z_min = z_range[np.argmin(np.abs(z_range-z_lens))]
		r_cmv_min = np.abs(self.cosmo.comovingDistance(z_min,z_lens))
		if r_cmv_min >= r_max*self.cosmo.h:
			return 1

		r_cmv = np.abs(self.cosmo.comovingDistance(z_range,z_lens))
		# Don't try to calculate inside the virial radius.
		r_cmv = r_cmv[r_cmv>r_min*self.cosmo.h]

		# Get the two halo term in the slice
		xi_halo = self.cosmo.correlationFunction(r_cmv,z_lens)
		xi_halo *= bias.haloBias(lens_m200*self.cosmo.h,z_lens,mdef='200c',
			model='tinker10')
		return 1+np.mean(xi_halo)

	@functools.lru_cache(maxsize=1024)
	def cone_angle_to_radius(self,z,z_lens,z_source,cone_angle,
		angle_buffer=0.8):
		"""Returns the radius in kpc at the given redshift for the given cone
		angle.

		Args:
			z (float): The redshift at which to calculate the radius
			z_lens (float): The redshift of the main deflector
			z_source (float): The redshift of the source
			dz (float): The thickness of the redshift slice to consider
			cone_angle (float): The opening angle of the cone for the los
				realization in units of arcseconds.
			angle_buffer (float): A buffer for how small the second cone
				gets as it approaches the source. Should be between 0 and
				1 with 1 meaning that the second cone converges to a point
				at the source.

		Retruns:
			(float): The radius in units of physical kpc.
		"""
		# Get the conversion between the angular opening and kpc
		kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,self.cosmo)
		r_los = kpc_per_arcsecond*cone_angle*0.5
		# If we're past the lens, shrink the light cone proportional to the
		# distance to the source.
		if z > z_lens:
			# Formula picked to match double cone in DG19
			scal_factor = angle_buffer
			scal_factor *= self.cosmo.comovingDistance(z_source)
			scal_factor /= self.cosmo.comovingDistance(z_lens,z_source)
			scal_factor *= self.cosmo.comovingDistance(z_lens,z)
			scal_factor /= self.cosmo.comovingDistance(z)
			r_los *= 1 - scal_factor
		return r_los

	@functools.lru_cache(maxsize=1024)
	def volume_element(self,z,z_lens,z_source,dz,cone_angle,angle_buffer=0.8):
		"""Returns the physical volume element at the given redshift

		Args:
			z (float): The redshift at which to calculate the volume element
			z_lens (float): The redshift of the main deflector
			z_source (float): The redshift of the source
			dz (float): The thickness of the redshift slice to consider
			cone_angle (float): The opening angle of the cone for the los
				realization in units of arcseconds.

		Retruns:
			(float): The physical volume element in kpc**3

		Notes:
			This depends on parameters like the cone angle, the redshift of
			the source, and the redshift of the lens.
		"""
		r_los = self.cone_angle_to_radius(z+dz/2,z_lens,z_source,cone_angle,
			angle_buffer=angle_buffer)

		# Get the thickness of our cone slice in physical units of kpc
		dz_in_kpc = self.cosmo.comovingDistance(z,z+dz)/self.cosmo.h
		dz_in_kpc /= (1+z)
		# Mpc to kpc
		dz_in_kpc *= 1000

		return dz_in_kpc * np.pi * r_los**2

	def draw_nfw_masses(self,z):
		"""Draws from the Sheth Tormen mass function with an additional
		correction for two point correlation with main lens.

		Args:
			z (float): The redshift at which to draw the masses

		Returns:
			(np.array): An array with the drawn masses in units of M_sun.

		"""
		# Pull the parameters we need from the input dictionaries
		# Units of M_sun
		lens_m200 = self.main_deflector_parameters['M200']
		z_lens = self.main_deflector_parameters['z_lens']
		z_source = self.source_parameters['z_source']
		dz = self.los_parameters['dz']
		# Units of arcsecond
		cone_angle = self.los_parameters['cone_angle']
		# Units of Mpc
		r_max = self.los_parameters['r_max']
		# Units of Mpc
		r_min = self.los_parameters['r_min']
		# Units of M_sun
		m_min = self.los_parameters['m_min']
		# Units of M_sun
		m_max = self.los_parameters['m_max']
		delta_los = max(0, self.los_parameters['delta_los'])
		# Get the parameters of the power law fit to the Sheth Tormen mass
		# function
		pl_slope, pl_norm = self.power_law_dn_dm(z+dz/2,m_min,m_max)

		# Scale the norm by the total volume and the two point correlation.
		dV = self.volume_element(z,z_lens,z_source,dz,cone_angle)
		halo_boost = self.two_halo_boost(z,z_lens,dz,lens_m200,r_max,r_min)
		pl_norm *= dV * halo_boost * delta_los

		# Draw from our power law and return the masses.
		masses = power_law.power_law_draw(m_min,m_max,pl_slope,pl_norm)
		return masses

	def sample_los_pos(self,z,n_los):
		"""Draws the positions for the line of sight substructure at the given
		redshift.

		Args:
			z (float): The redshift to place the los halos at.
			n_los (int): The number of los halos to draw position for

		Returns:
			(np.array): A n_los x 2 array giving the x,y position of the
			line of sight structure in units of kpc.
		"""
		# Pull the parameters we need from the input dictionaries
		# Units of M_sun
		z_lens = self.main_deflector_parameters['z_lens']
		z_source = self.source_parameters['z_source']
		dz = self.los_parameters['dz']
		# Units of arcsecond
		cone_angle = self.los_parameters['cone_angle']

		r_los = self.cone_angle_to_radius(z+dz/2,z_lens,z_source,cone_angle)
		# Draw the radii of the los halos.
		r_draws = r_los*np.sqrt(np.random.rand(n_los))
		theta = 2*np.pi*np.random.rand(n_los)

		# Create an array for the coordinates
		cart_pos = np.zeros((n_los,2))

		cart_pos[:,0] = r_draws * np.cos(theta)
		cart_pos[:,1] = r_draws * np.sin(theta)

		return cart_pos

	def mass_concentration(self,z,m_200,scatter_mult=1.0):
		"""Returns the concentration of halos at a certain mass given the
		parameterization of DG_19.

		Args:
			z (float): The redshift of the nfw halos
			m_200 (np.array): array of M_200 of the nfw halo units of M_sun
			scatter_mult (float): an additional scaling to the scatter. Likely
				only useful for los rendering to force scatter to 0.

		Returns:
			(np.array): The concentration for each halo.
		"""
		# Get the concentration parameters
		c_0 = self.los_parameters['c_0']
		zeta = self.los_parameters['conc_zeta']
		beta = self.los_parameters['conc_beta']
		m_ref = self.los_parameters['conc_m_ref']
		dex_scatter = self.los_parameters['dex_scatter']*scatter_mult

		# The peak calculation is done by colossus. The cosmology must have
		# already been set. Note these functions expect M_sun/h units (which
		# you get by multiplying by h
		# https://www.astro.ljmu.ac.uk/~ikb/research/h-units.html)
		h = self.cosmo.h
		peak_heights = peaks.peakHeight(m_200*h,z)
		peak_height_ref = peaks.peakHeight(m_ref*h,0)

		# Now get the concentrations and add scatter
		concentrations = c_0*(1+z)**(zeta)*(peak_heights/peak_height_ref)**(
			-beta)
		if isinstance(concentrations,np.ndarray):
			conc_scatter = np.random.randn(len(concentrations))*dex_scatter
		elif isinstance(concentrations,float):
			conc_scatter = np.random.randn()*dex_scatter
		concentrations = 10**(np.log10(concentrations)+conc_scatter)

		return concentrations

	def convert_to_lenstronomy(self,z,z_masses,z_cart_pos):
		"""Converts the subhalo masses and position to truncated NFW profiles
		for lenstronomy

		Args:
			z (float): The redshift for each of the halos
			z_masses (np.array): The masses of each of the halos that
				were drawn
			z_cart_pos (np.array): A n_los x 2D array of the position of the
				halos that were drawn
		Returns:
			([string,...],[dict,...]): A tuple containing the list of models
			and the list of kwargs for the truncated NFWs.
		"""
		z_source = self.source_parameters['z_source']
		# First, draw a concentration for all our LOS structure from our mass
		# concentration relation
		concentration = self.mass_concentration(z,z_masses)

		# Now convert our mass and concentration into the lenstronomy
		# parameters
		z_r_200 = nfw_functions.r_200_from_m(z_masses,z,self.cosmo)
		z_r_scale = z_r_200/concentration
		z_rho_nfw = nfw_functions.rho_nfw_from_m_c(z_masses,concentration,
			self.cosmo,r_scale=z_r_scale)

		# Convert to lenstronomy units
		z_r_scale_ang, alpha_Rs = nfw_functions.convert_to_lenstronomy_NFW(
			z_r_scale,z,z_rho_nfw,z_source,self.cosmo)
		kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,self.cosmo)
		cart_pos_ang = z_cart_pos / np.expand_dims(kpc_per_arcsecond,-1)

		# Populate the parameters for each lens
		model_list = []
		kwargs_list = []

		for i in range(len(z_masses)):
			model_list.append('NFW')
			kwargs_list.append({'alpha_Rs':alpha_Rs[i], 'Rs':z_r_scale_ang[i],
				'center_x':cart_pos_ang[i,0],'center_y':cart_pos_ang[i,1]})

		return (model_list,kwargs_list)

	def draw_los(self):
		"""Draws masses, concentrations,and positions for the los substructure
		of a main lens halo.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each los halo returned, the second is the lenstronomy kwargs
			for that halo, and the third is a list of redshift values for
			each profile.
		Notes:
			The returned lens model list includes terms to correct for
			the average deflection angle introduced from the los halos.
		"""
		# Distribute line of sight substructure according to
		# https://arxiv.org/pdf/1909.02573.pdf. This also includes a
		# correction for the average deflection angle introduced by
		# the addition of the substructure.
		los_model_list = []
		los_kwargs_list = []
		los_z_list = []

		# Pull the paramters we need
		z_min = self.los_parameters['z_min']
		z_source = self.source_parameters['z_source']
		dz = self.los_parameters['dz']

		# Add halos from the starting reshift to the source redshift.
		# Note most of the calculations are done at z + dz/2, so you
		# want to stop at z_source-dz.
		z_range = np.arange(z_min,z_source-dz,dz)
		# Round the z_range to improve caching hits.
		z_range = list(np.round(z_range,2))

		# Iterate through each z and add the halos.
		for z in z_range:
			# Draw the masses and positions at this redshift from our
			# model
			z_masses = self.draw_nfw_masses(z)
			# Don't add anything to the model if no masses were drawn
			if z_masses.size == 0:
				continue
			z_cart_pos = self.sample_los_pos(z,len(z_masses))
			# Convert the mass and positions to lenstronomy models
			# and kwargs and append to our lists.
			model_list, kwargs_list = self.convert_to_lenstronomy(
				z,z_masses,z_cart_pos)
			los_model_list += model_list
			los_kwargs_list += kwargs_list

			# TODO - correction for average line of sight

			los_z_list += [z+dz/2]*len(model_list)

		return (los_model_list, los_kwargs_list, los_z_list)

	def calculate_average_alpha(self,num_pix):
		""" Calculates the average deflection maps from the los at each
		redshift specified by the los parameters and returns corresponding
		lenstronomy objects.

		Args:
			num_pix (int): The number of pixels to sample for our
				interpolation maps.

		Returns:
			(tuple): A tuple of two lists: the first is the interpolation
			profile type for each redshift slice and the second is the
			lenstronomy kwargs for that profile.

		Notes:
			The average los deflection angles of the lenstronomy objects will
			be the negative of the average (since we want to subtract the
			average effect not add it). Pixel scale will be set such that
			at each redshift a box of 5*r_los is captured.
		"""
		# Pull the parameters we need.
		z_min = self.los_parameters['z_min']
		z_source = self.source_parameters['z_source']
		z_lens = self.main_deflector_parameters['z_lens']
		dz = self.los_parameters['dz']
		dz *= self.los_parameters['alpha_dz_factor']
		delta_los = max(0, self.los_parameters['delta_los'])
		cone_angle = self.los_parameters['cone_angle']
		m_min = self.los_parameters['m_min']
		# Units of M_sun
		m_max = self.los_parameters['m_max']
		z_range = np.arange(z_min,z_source-dz,dz)
		# Round the z_range to improve caching hits. Add the dz/2 shift that
		# gets output by draw_los.
		z_range = list(np.round(z_range,2)+dz/2)

		# The lists where we'll store the lenstornomy variables
		interp_model_list = []
		interp_kwargs_list = []
		interp_z_list = []

		for zi, z in enumerate(z_range):
			# First we make the grid on which we'll conduct the interpolation.
			r_los = self.cone_angle_to_radius(z+dz/2,z_lens,z_source,
				cone_angle)
			kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,self.cosmo)
			r_los = r_los/kpc_per_arcsecond

			# Set the pixel scale to capture a box of dimension 5*r_los.
			pixel_scale = 5*r_los/num_pix
			x_grid, y_grid = util.make_grid(numPix=num_pix,
				deltapix=pixel_scale)
			x_axes, y_axes = util.get_axes(x_grid, y_grid)

			# Then we create the 2d rendering disk we want to convolve our
			# NFW with
			disk_bool = np.zeros(x_grid.shape)
			disk_bool[np.sqrt(x_grid**2+y_grid**2)<r_los] = 1
			disk_bool = util.array2image(disk_bool)

			# Next we calculate the NFW parameters for our NFW of average
			# mass.
			pl_slope, pl_norm = self.power_law_dn_dm(z+dz/2,m_min,m_max)
			m_average = (power_law.power_law_integrate(m_min,m_max,pl_slope+1)/
				power_law.power_law_integrate(m_min,m_max,pl_slope))
			c_average = self.mass_concentration(z,m_average,scatter_mult=0.0)

			# Convert our parameters to the lenstronomy definition
			r_200_avg = nfw_functions.r_200_from_m(m_average,z,self.cosmo)
			r_scale_avg = r_200_avg/c_average
			rho_nfw_avg = nfw_functions.rho_nfw_from_m_c(m_average,c_average,
				self.cosmo,r_scale=r_scale_avg)
			r_scale_ang_avg, alpha_Rs_avg = (
				nfw_functions.convert_to_lenstronomy_NFW(r_scale_avg,z,
				rho_nfw_avg,z_source,self.cosmo))

			# Get our deflection angle for this NFW profile
			ax,ay = NFW().derivatives(x_grid,y_grid,r_scale_ang_avg,
				alpha_Rs_avg)
			ax = util.array2image(ax)
			ay = util.array2image(ay)

			# Convolve our deflection angles with the disk
			ax_conv = fftconvolve(ax,disk_bool,mode='same')
			ay_conv = fftconvolve(ay,disk_bool,mode='same')

			# Finally, we just need to calculate how many of our
			# m_average NFW we expect per pixel. So far our convolution
			# is the deflection angle assuming 1 NFW / pixel.
			# First we calculate the number of m_avg nfw per kpc^3
			m_total = power_law.power_law_integrate(m_min,m_max,pl_slope+1)
			m_total *= pl_norm * delta_los
			n_total = m_total/m_average

			# Now we multiply by the length of our redshift slice to get per
			# kpc^2 and then multiply by kpc_per_arcsecond to get per
			# arcsecond^2
			dz_in_kpc = self.cosmo.comovingDistance(z-dz/2,z+dz/2)/self.cosmo.h
			dz_in_kpc /= (1+z-dz/2)
			# Mpc to kpc
			dz_in_kpc *= 1000
			n_total *= dz_in_kpc
			n_total *= kpc_per_arcsecond**2

			# Final convert from per arcsecond^2 to per pixel
			n_total *= pixel_scale**2

			ax_conv *= n_total
			ay_conv *= n_total

			# And now we can return the parameters of our interpol
			# profile
			interp_model_list.append('INTERPOL')
			interp_kwargs_list.append({'grid_interp_x':x_axes,
				'grid_interp_y':y_axes, 'f_x':-ax_conv,
				'f_y':-ay_conv})
			interp_z_list.append(z)

		return (interp_model_list,interp_kwargs_list,interp_z_list)
