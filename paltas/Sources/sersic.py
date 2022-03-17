# -*- coding: utf-8 -*-
"""
Provides classes for specifying a sersic light distribution.

This module contains the class required to provide a sersic light distribution
as the source for paltas.
"""
from .source_base import SourceBase
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
	- center_x - x-coordinate lens center in units of arcseconds
	- center_y - y-coordinate lens center in units of arcseconds
	- z_source - source redshift
	"""

	required_parameters = ('magnitude', 'output_ab_zeropoint', 'R_sersic', 
		'n_sersic', 'e1', 'e2', 'center_x', 'center_y', 'z_source')

	def draw_source(self):
		"""Return lenstronomy LightModel kwargs

		Returns:
			(list,list) A list containing the model names(s), and
			a list containing the model kwargs dictionaries.
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
		sersic_params['amp'] = SingleSersicSource.mag_to_amplitude(
			self.source_parameters['magnitude'],
			self.source_parameters['output_ab_zeropoint'], sersic_params)
		return (
			['SERSIC_ELLIPSE'],
			[sersic_params])

	@staticmethod
	def mag_to_amplitude(mag, mag_zero_point, kwargs_list):
		"""Converts a user defined magnitude to the corresponding amplitude
		that lenstronomy will use
	
		Args:
			mag (float): user defined desired magnitude
			kwargs_list (dict): dict of kwargs for SERSIC_ELLIPSE, amp 
			parameter not included

		Returns: 
			(float): amplitude lenstronomy should use to get desired magnitude
			desired magnitude
		"""

		sersic_model = LightModel(['SERSIC_ELLIPSE'])
		# norm=True sets amplitude = 1
		flux_norm = sersic_model.total_flux([kwargs_list], norm=True)[0]
		flux_true = magnitude2cps(mag, mag_zero_point)
		
		return flux_true/flux_norm
