# -*- coding: utf-8 -*-
"""
Provides classes for specifying a sersic light distribution.

This module contains the class required to provide a sersic light distribution
as the source for paltas.
"""
from .source_base import SourceBase
from ..Utils.cosmology_utils import absolute_to_apparent, kpc_per_arcsecond
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util.data_util import magnitude2cps
import numpy as np
from lenstronomy.LensModel.Profiles import sersic_utils
from scipy.special import gammainc, gamma
from scipy.optimize import fsolve


class SingleSersicSource(SourceBase):
	"""Class to generate single Sersic profile light models

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters: dictionary with source-specific parameters.

	Notes:

	Required Parameters

	- magnitude - AB absolute magnitude of the source
	- output_ab_zeropoint - AB magnitude zeropoint of the detector
	- R_sersic - Sersic radius in units of arcseconds
	- n_sersic - Sersic index
	- e1 - x-direction ellipticity eccentricity
	- e2 - xy-direction ellipticity eccentricity
	- center_x - x-coordinate source center in units of arcseconds
	- center_y - y-coordinate source center in units of arcseconds
	- z_source - source redshift
	"""

	required_parameters = ('magnitude','output_ab_zeropoint','R_sersic',
		'n_sersic','e1','e2','center_x','center_y','z_source')

	def draw_source(self):
		"""Return lenstronomy LightModel kwargs

		Returns:
			(list,list,list) A list containing the model name(s),
			a list containing the model kwargs dictionaries, and a list
			containing the redshifts of each model. Redshifts list can
			be None.
		"""
		# Just extract each of the sersic parameters.
		sersic_params ={
			k: v
			for k, v in self.source_parameters.items()
			if k in self.required_parameters}
		sersic_params.pop('z_source')
		sersic_params.pop('output_ab_zeropoint')

		# mag to amp conversion
		sersic_params.pop('magnitude')
		mag_apparent = absolute_to_apparent(self.source_parameters['magnitude'],
			self.source_parameters['z_source'],self.cosmo)
		sersic_params['amp'] = SingleSersicSource.mag_to_amplitude(
			mag_apparent,self.source_parameters['output_ab_zeropoint'],
			sersic_params)
		return (
			['SERSIC_ELLIPSE'],
			[sersic_params],[self.source_parameters['z_source']])

	@staticmethod
	def mag_to_amplitude(mag_apparent,mag_zeropoint,kwargs_list):
		"""Converts a user defined magnitude to the corresponding amplitude
		that lenstronomy will use
	
		Args:
			mag_apparent (float): The desired apparent magnitude
			mag_zeropoint (float): The magnitude zero-point of the detector
			kwargs_list (dict): A dict of kwargs for SERSIC_ELLIPSE, amp
				parameter not included

		Returns: 
			(float): amplitude lenstronomy should use to get desired magnitude
			desired magnitude
		"""

		sersic_model = LightModel(['SERSIC_ELLIPSE'])
		# norm=True sets amplitude = 1
		flux_norm = sersic_model.total_flux([kwargs_list], norm=True)[0]
		flux_true = magnitude2cps(mag_apparent, mag_zeropoint)
		
		return flux_true/flux_norm

	@staticmethod
	def get_total_sersic_flux_r(r,R_sersic,n_sersic,amp_sersic):
		"""Returns the total sersic flux within a radius r.

		Args:
			r (float):  The radius to calculate the flux to in the
				same units as R_sersic.
			R_sersic (float): The sersic half-light radius in the same
				units as r.
			n_sersic (float): The sersic index
			amp_sersic (float): The amplitude (normalization) of the sersic
				luminosity. Should have units of flux per units of r^2. So if
				flux has units of kpc then it should have units flux/kpc^2.

		Returns:
			(float): The total flux within the radius r in counts per second.
		"""
		# Calculate the total flux from the analytic expression
		b_n = sersic_utils.SersicUtil.b_n(n_sersic)
		total = R_sersic**2*2*np.pi*n_sersic*np.exp(b_n)/(b_n**(2*n_sersic))
		total *= gammainc(2*n_sersic,b_n*(r/R_sersic)**(1/n_sersic))
		total *= gamma(2*n_sersic)
		total *= amp_sersic
		return total

	@staticmethod
	def get_total_sersic_flux(R_sersic,n_sersic,amp_sersic):
		"""Returns the total sersic flux.

		Args:
			R_sersic (float): The sersic half-light radius in the same
				units as r.
			n_sersic (float): The sersic index
			amp_sersic (float): The amplitude (normalization) of the sersic
				luminosity.

		Returns:
			(float): The total flux in counts per second.
		"""
		# Calculate the total flux from the analytic expression
		b_n = sersic_utils.SersicUtil.b_n(n_sersic)
		total = R_sersic**2*2*np.pi*n_sersic*np.exp(b_n)/(b_n**(2*n_sersic))
		total *= gamma(2*n_sersic)
		total *= amp_sersic
		return total


class DoubleSersicData(SingleSersicSource):
	"""Class to generate a bulge + disk sersic light model for use as the lens
	light of the deflector.

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters: dictionary with source-specific parameters.

	Notes:

	Required Parameters

	- magnitude - AB absolute total magnitude of the source
	- f_bulge - the fraction of the flux to assign to the bulge
	- output_ab_zeropoint - AB magnitude zeropoint of the detector
	- n_bulge - sersic index of the bulge. Should be close to 4
	- n_disk - sersic index of the disk. Should be close to 1
	- r_disk_bulge - the ratio of the disk to bulge half-light radius
	- e1 - x-direction ellipticity eccentricity of both components
	- e2 - xy-direction ellipticity eccentricity of both components
	- center_x - x-coordinate source center in units of arcseconds
	- center_y - y-coordinate source center in units of arcseconds
	- z_source - light source redshift (should be the same as main deflector)
	"""

	required_parameters = ('magnitude', 'f_bulge', 'output_ab_zeropoint',
		'n_bulge','n_disk','r_disk_bulge','e1_bulge','e2_bulge','e1_disk',
		'e2_disk','center_x','center_y','z_source')

	def get_bulge_disk_mag(self):
		"""Returns the apparent magnitude for the bulge and the disk

		Returns:
			(float,float): The apparent magnitude of the bulge and the
				apparent magnitude of the disk as a tuple.
		"""
		# Convert the magnitude to apparent magnitude=
		mag_apparent = absolute_to_apparent(
			self.source_parameters['magnitude'],
			self.source_parameters['z_source'],self.cosmo)

		# Divide the magnitude into the two components
		mag_bulge = mag_apparent - 2.5 * np.log10(
			self.source_parameters['f_bulge'])
		mag_disk = mag_apparent - 2.5 * np.log10(
			1-self.source_parameters['f_bulge'])

		return mag_bulge, mag_disk

	def get_total_half_light(self):
		"""Returns the half-light radius for the sum of the two
		components.

		Returns:
			(float): The half-light radius for the sum of the two
			components in units of kpc.
		Notes:
			The total half-light radius is set by the SDS galaxy size
			distribution for early type galaxies
			https://arxiv.org/abs/astro-ph/0301527. The half light radius
			of each component is set to respect the user provided ratio and
			the SDS constraint for the total.
		"""
		# Define the fit parameters
		a = 0.6
		b = -5.06
		M_0 = -20.52
		sigma_1, sigma_2 = 0.45, 0.27

		# Extract the absolute magnitude
		M = self.source_parameters['magnitude']

		# Start with the mean relation
		log_R_half = -0.4*a*M+b
		ln_R_half = np.log(10**log_R_half)

		# Add scatter
		ln_R_half += np.random.randn() * (
			sigma_2 + (sigma_1-sigma_2)/(1+10**(-0.8*(M-M_0))))

		return np.exp(ln_R_half)

	def get_bulge_disk_half_light(self,R_total):
		"""Returns the half-light radius for the disk and the bulge.

		Args:
			R_total (float): The total half-light radius for the full
				system in units of kpc.
		Returns:
			(float,float): A tuple with half light radius of the bulge
			and disk components in units of kpc.
		Notes:
			The total half-light radius is set by the SDS galaxy size
			distribution for early type galaxies
			https://arxiv.org/abs/astro-ph/0301527. The half light radius
			of each component is set to respect the user provided ratio and
			the SDS constraint for the total.
		"""
		# Extract the the flux ratios, the ratio of the half-light radii for
		# both systems, and the sersic index for both systems.
		f_bulge = self.source_parameters['f_bulge']
		r_disk_bulge = self.source_parameters['r_disk_bulge']
		n_bulge = self.source_parameters['n_bulge']
		n_disk = self.source_parameters['n_disk']

		# Build the function that returns the fraction of the total flux
		# contained within the half-light radius R_sersic for a disk
		# of half-light radius R_disk and a bulge of half-light
		# ratio R_disk/r_disk_bluge.
		def flux_fraction(R_disk):
			# Deal with negative guesses. Just return 0.5 (the return for
			# R_disk of 0) plus R_disk.
			if R_disk < 0:
				return R_disk - 0.5

			# Get the fluxes at R_total
			flux_disk = self.get_total_sersic_flux_r(R_total,R_disk,
				n_disk,1-f_bulge)
			flux_bulge = self.get_total_sersic_flux_r(R_total,
				R_disk/r_disk_bulge,n_bulge,f_bulge)

			# Get the total flux
			flux_disk_total = self.get_total_sersic_flux(R_disk,n_disk,
				1-f_bulge)
			flux_bulge_total = self.get_total_sersic_flux(R_disk/r_disk_bulge,
				n_bulge,f_bulge)

			# Return the ratio minus a half
			return ((flux_disk+flux_bulge)/(flux_disk_total+flux_bulge_total)
				- 0.5)

		# Solve for the disk radius that would give us the correct half-light
		# radius.
		R_disk_solve = fsolve(flux_fraction,R_total*r_disk_bulge)[0]
		R_bulge_solve = R_disk_solve / r_disk_bulge

		return R_bulge_solve,R_disk_solve

	def draw_source(self):
		"""Returns lenstronomy LightModel kwargs

		Returns:
			(list,list,list) A list containing the model name(s),
			a list containing the model kwargs dictionaries, and a list
			containing the redshifts of each model. Redshifts list can
			be None.
		"""
		# Get the apparent magnitude of the bulge and the disk
		mag_bulge, mag_disk = self.get_bulge_disk_mag()

		# Get the total half-light radius
		r_total = self.get_total_half_light()

		# Get the half-light radius of the disk and the bulge
		r_half_bulge, r_half_disk = self.get_bulge_disk_half_light(r_total)

		# Convert the radii to angular coordinates.
		kpc_p_a = kpc_per_arcsecond(self.source_parameters['z_source'],
			self.cosmo)
		r_half_bulge /= kpc_p_a
		r_half_disk /= kpc_p_a

		# Make the kwargs for both sersics.
		kwargs_bulge = {'R_sersic':r_half_bulge,
			'n_sersic':self.source_parameters['n_bulge'],
			'e1':self.source_parameters['e1_bulge'],
			'e2':self.source_parameters['e2_bulge']}
		kwargs_disk = {'R_sersic':r_half_disk,
			'n_sersic':self.source_parameters['n_disk'],
			'e1':self.source_parameters['e1_disk'],
			'e2':self.source_parameters['e2_disk']}
		light_model_kwargs = [kwargs_bulge,kwargs_disk]

		# Add the shared parameters.
		for kwargs in light_model_kwargs:
			kwargs['center_x'] = self.source_parameters['center_x']
			kwargs['center_y'] = self.source_parameters['center_y']

		# Get the amplitude for each component
		amp_bulge = SingleSersicSource.mag_to_amplitude(mag_bulge,
			self.source_parameters['output_ab_zeropoint'],kwargs_bulge)
		kwargs_bulge['amp'] = amp_bulge
		amp_disk = SingleSersicSource.mag_to_amplitude(mag_disk,
			self.source_parameters['output_ab_zeropoint'],kwargs_disk)
		kwargs_disk['amp'] = amp_disk

		# Populate our remaining lists.
		light_model_list = ['SERSIC_ELLIPSE']*2
		light_z_list = [self.source_parameters['z_source']]*2

		return light_model_list,light_model_kwargs,light_z_list
