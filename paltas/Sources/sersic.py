# -*- coding: utf-8 -*-
"""
Provides classes for specifying a sersic light distribution.

This module contains the class required to provide a sersic light distribution
as the source for paltas.
"""
from .source_base import SourceBase
from ..Utils.cosmology_utils import absolute_to_apparent
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util.data_util import magnitude2cps


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
	def mag_to_amplitude(mag_apparent,mag_zero_point,kwargs_list):
		"""Converts a user defined magnitude to the corresponding amplitude
		that lenstronomy will use
	
		Args:
			mag_apparent (float): The desired apparent magnitude
			mag_zero_point (float): The magnitude zero-point of the detector
			kwargs_list (dict): A dict of kwargs for SERSIC_ELLIPSE, amp
				parameter not included

		Returns: 
			(float): amplitude lenstronomy should use to get desired magnitude
			desired magnitude
		"""

		sersic_model = LightModel(['SERSIC_ELLIPSE'])
		# norm=True sets amplitude = 1
		flux_norm = sersic_model.total_flux([kwargs_list], norm=True)[0]
		flux_true = magnitude2cps(mag_apparent, mag_zero_point)
		
		return flux_true/flux_norm


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
	- n_bulge - sersic index of the bulge. Should be close to 1
	- n_disk - sersic index of the disk. Should be close to 4
	- r_disk_bulge - the ratio of the disk to bulge half-light radius
	- e1 - x-direction ellipticity eccentricity of both components
	- e2 - xy-direction ellipticity eccentricity of both components
	- center_x - x-coordinate source center in units of arcseconds
	- center_y - y-coordinate source center in units of arcseconds
	- z_source - light source redshift (should be the same as main deflector)
	"""

	required_parameters = ('magnitude', 'f_bulge', 'output_ab_zeropoint',
		'n_bulge','n_disk','r_disk_bulge','e1','e2','center_x','center_y',
		'z_source')

	def draw_source(self):
		"""Return lenstronomy LightModel kwargs

		Returns:
			(list,list,list) A list containing the model name(s),
			a list containing the model kwargs dictionaries, and a list
			containing the redshifts of each model. Redshifts list can
			be None.
		"""

		# Get the magnitude of the bulge and the disk
		M_disk =2
		# Just extract each of the sersic parameters.
		sersic_params ={
			k: v
			for k, v in self.source_parameters.items()
			if k in self.required_parameters}
		sersic_params.pop('z_source')
		sersic_params.pop('output_ab_zeropoint')

		# mag to amp conversion
		sersic_params.pop('magnitude')
		sersic_params['amp'] = SingleSersicSource.mag_to_amplitude(
			self.source_parameters['magnitude'],
			self.source_parameters['output_ab_zeropoint'], sersic_params)
		return (
			['SERSIC_ELLIPSE'],
			[sersic_params],[self.source_parameters['z_source']])
