# -*- coding: utf-8 -*-
"""
Define the base class to draw subhalos for a lens

This module contains the base class that all the subhalo classes will build
from. Because the steps for rendering subhalos can vary between different
models, the required functions are very sparse.
"""
from ..Utils.cosmology_utils import get_cosmology


class SubhalosBase():
	""" Base class for rendering the subhalos of a main halo.

	Args:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		source_parameters (dict): A dictionary containing the type of the
			source and the value for each of its parameters.
		cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	"""

	def __init__(self,subhalo_parameters,main_deflector_parameters,
		source_parameters,cosmology_parameters):

		# Save the parameters
		self.subhalo_parameters = subhalo_parameters
		self.main_deflector_parameters = main_deflector_parameters
		self.source_parameters = source_parameters

		# Turn our cosmology parameters into a colossus cosmology instance
		self.cosmo = get_cosmology(cosmology_parameters)

	def check_parameterization(self,required_params):
		""" Check that all the required parameters are present in the
		subhalo_parameters.

		Args:
			required_params ([str,...]): A list of strings containing the
				required parameters.
		"""
		if not all(elem in self.subhalo_parameters.keys() for
			elem in required_params):
			raise ValueError('Not all of the required parameters for the' +
				'parameterization are present.')

	def draw_subhalos(self):
		"""Draws masses, concentrations,and positions for the subhalos of a
		main lens halo.

		Returns:
			(tuple): A tuple of two lists: the first is the profile type for
			each subhalo returned and the second is the lenstronomy kwargs for
			that subhalo.
		"""
		raise NotImplementedError
