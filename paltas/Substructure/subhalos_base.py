# -*- coding: utf-8 -*-
"""
Define the base class to draw subhalos for a lens

This module contains the base class that all the subhalo classes will build
from. Because the steps for rendering subhalos can vary between different
models, the required functions are very sparse.
"""
import paltas


class SubhalosBase(paltas.BaseComponent):
	""" Base class for rendering the subhalos of a main halo.

	Args:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
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
		'subhalo_parameters', 'main_deflector_parameters', 
		'source_parameters', 'cosmology_parameters')

	def draw_subhalos(self):
		"""Draws masses, concentrations,and positions for the subhalos of a
		main lens halo.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each subhalo returned, the second is the lenstronomy kwargs for
			that subhalo, and the third is the redshift for each subhalo.
		"""
		raise NotImplementedError

	def draw(self, result, **kwargs):
		result.add_lenses(*self.draw_subhalos())
