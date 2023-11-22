# -*- coding: utf-8 -*-
"""
Define the base class to draw line of sight substructure for a lens

This module contains the base class that all the los classes will build
from. Because the steps for rendering halos can vary between different
models, the required functions are very sparse.
"""
import paltas


class LOSBase(paltas.BaseComponent):
	"""Base class for rendering the los of a main halo.

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
	"""

	init_kwargs = (
		'los_parameters', 'main_deflector_parameters',
		'source_parameters', 'cosmology_parameters')

	def draw_los(self):
		"""Draws masses, concentrations,and positions for the los substructure
		of a main lens halo.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each los halo returned, the second is the lenstronomy kwargs
			for that halo, and the third is a list of redshift values for
			each profile.
		"""
		raise NotImplementedError

	def calculate_average_alpha(self,n_draws=100):
		"""Calculates the average deflection maps from the los at each
		redshift specified by the los parameters and returns corresponding
		lenstronomy objects.

		Args:
			num_pix (int): The number of pixels to sample for our
				interpolation maps.

		Returns:
			(tuple): A tuple of three lists: the first is the interpolation
			profile type for each redshift slice and the second is the
			lenstronomy kwargs for each profile, and the third is a list of 
			redshift values for each profile.
		"""
		raise NotImplementedError

	def draw(self, result, *, kwargs_numerics, numpix, **kwargs):
		result.add_lenses(*self.draw_los())
		result.add_lenses(*self.calculate_average_alpha(
			numpix * kwargs_numerics['supersampling_factor']))
