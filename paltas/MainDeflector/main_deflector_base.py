# -*- coding: utf-8 -*-
"""
Provide the base class for specifying the main deflector of a lensing system.

This module contains the base class that all the main deflector classes will
build from. Because the steps for rendering a main deflector can vary between
different models, the required functions are very sparse.
"""
import paltas


class MainDeflectorBase(paltas.BaseComponent):
	"""Base class for rendering the main halo.

	Args:
		main_deflector_parameters (dict): A dictionary containing the type of
			los distribution and the value for each of its parameters.
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	"""

	init_kwargs = ('main_deflector_parameters', 'cosmology_parameters')

	def draw_main_deflector(self):
		"""Draws the lenstronomy profile names and kwargs for the components
		of the main deflector.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each component of the main deflector, the second is the
			lenstronomy kwargs for each component, and the third is the
			redshift for each component.
		"""
		raise NotImplementedError

	def draw(self, result, **kwargs):
		result.add_lenses(*self.draw_main_deflector())
