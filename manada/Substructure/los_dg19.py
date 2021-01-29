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

# Define the parameters we expect to find for the DG_19 model
# TODO Fill this once we have all the parameters
draw_nfw_masses_DG_19_parameters = ['m_min','m_max','z_min','dz','cone_angle',
	'r_max','r_min']


class LOSDG19(LOSBase):
	"""Class for rendering the line of sight structure according to DG19.

	Args:
		los_parameters (dict): A dictionary containing the type of
			los distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		source_parameters (dict): A dictionary containing the type of the
			source and the value for each of its parameters.
		cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	Notes:
		This class uses caching to improve performance over many draws. Do not
		manually change the values of the passed in dictionary values as the
		cached functions will not register that those parameters have changed.
		For new parameter values use the update_parameters function.
	"""

	def __init__(self,los_parameters,main_deflector_parameters,
		source_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(los_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

		# Check that all the needed parameters are present
		self.check_parameterization(draw_nfw_masses_DG_19_parameters)

	@staticmethod
	@numba.njit
	def nu_f_nu(nu):
		"""Calculates :math:`\nu f(\nu)` for the Sheth Tormen 2001 model.

		Args:
			nu (np.array): An array of nu values to at which to
				calculate nu_f_nu

		Returns:
			(np.array): The value of :math:`\nu f(\nu)`

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
	def power_law_dn_dm(self,z,n_dm=100):
		"""Returns the best fit power law parameters for the physical number
		density at a given redshift and mass range.

		Args:
			z (float): The redshift at which to calculate the power law
				parameters.
			n_dm (int): The number of dm samples to consider in the fit.

		Returns:
			(tuple): The power law slope and the norm for the power law in
				units of 1/M_sun
		"""
		m_min = self.los_parameters['m_min']
		m_max = self.los_parameters['m_max']
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

	def update_parameters(self,los_parameters=None,
		main_deflector_parameters=None,source_parameters=None,
		cosmology_parameters=None):
		"""Updated the class parameters and clears the power_law cache if
		needed.

		Args:
			los_parameters (dict): A dictionary containing the type of
				los distribution and the value for each of its parameters.
			main_deflector_parameters (dict): A dictionary containing the type
				of main deflector and the value for each of its parameters.
			source_parameters (dict): A dictionary containing the type of the
				source and the value for each of its parameters.
			cosmology_parameters (str,dict, or
				colossus.cosmology.cosmology.Cosmology): Either a name
				of colossus cosmology, a dict with 'cosmology name': name of
				colossus cosmology, an instance of colussus cosmology, or a
				dict with H0 and Om0 ( other parameters will be set to
				defaults).

		Notes:
			Use this function to update parameter values instead of
			starting a new class. This function will preserve the cache if
			the parameters being changed to not affect the cache, and
			therefore will offer enormous improvements in performance.
		"""
		# Clear cache if min or max mass have changed.
		if ((los_parameters['m_min'] != self.los_parameters['m_min']) or
			(los_parameters['m_max'] != self.los_parameters['m_max'])):
			self.power_law_dn_dm.cache_clear()

		super().update_parameters(los_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

	def two_halo_boost(self,z,z_lens,dz,lens_m200,r_max,r_min,n_quads=1000):
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

	def volume_element(self,z,z_lens,z_source,dz,cone_angle):
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
		r_los = self.cone_angle_to_radius(z+dz/2,z_lens,z_source,cone_angle)

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
		m_min = self.los_parameters['m_min']
		m_max = self.los_parameters['m_max']
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
		# Get the parameters of the power law fit to the Sheth Tormen mass
		# function
		pl_slope, pl_norm = self.power_law_dn_dm(z)

		# Scale the norm by the total volume and the two point correlation.
		dV = self.volume_element(z,z_lens,z_source,dz,cone_angle)
		halo_boost = self.two_halo_boost(z,z_lens,dz,lens_m200,r_max,r_min)
		pl_norm *= dV * halo_boost

		# Draw from our power law and return the masses.
		masses = power_law.power_law_draw(m_min,m_max,pl_slope,pl_norm)
		return masses

	def draw_los(self):
		"""Draws masses, concentrations,and positions for the los substructure
		of a main lens halo.

		Returns:
			(tuple): A tuple of two lists: the first is the profile type for
			each los halo returned and the second is the lenstronomy kwargs
			for that halo.
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

		# Add halos from the starting reshift to the source redshift.
		z_range = np.arange(self.los_parameters['z_min'],
			self.source_parameters['z_source'],self.los_parameters['dz'])
		# Round the z_range to improve caching hits.
		z_range = list(np.round(z_range,2))

		# Iterate through each z and add the halos.
		for z in z_range:
			# Draw the masses and positions at this redshift from our
			# model
			z_masses = self.draw_nfw_masses(z)
			z_cart_pos = self.sample_los_pos(z,len(z_masses))
			# Convert the mass and positions to lenstronomy models
			# and kwargs and append to our lists.
			model_list, kwargs_list = self.convert_to_lenstronomy(
				z_masses,z_cart_pos)
			los_model_list += model_list
			los_kwargs_list += kwargs_list

			# TODO - correction for average line of sight

			los_z_list += [z]*len(model_list)

		return (los_model_list, los_kwargs_list, los_z_list)
