# -*- coding: utf-8 -*-
"""
Provide the base class for specifying the main deflector of a lensing system.

This module contains the base class that all the main deflector classes will
build from. Because the steps for rendering a main deflector can vary between
different models, the required functions are very sparse.
"""
from ..Utils.cosmology_utils import get_cosmology
import copy


class MainDeflectorBase():
	"""Base class for rendering the main halo.

	Args:
		main_deflector_parameters (dict): A dictionary containing the type of
			los distribution and the value for each of its parameters.
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	"""

	required_parameters = tuple()

	def __init__(self,main_deflector_parameters,cosmology_parameters):

		# Save the parameters as a copy to avoid any misunderstanding on the
		# user end.
		self.main_deflector_parameters = copy.deepcopy(
			main_deflector_parameters)

		# Turn our cosmology parameters into a colossus cosmology instance
		self.cosmo = get_cosmology(cosmology_parameters)

		# Check that all the needed parameters are present
		self.check_parameterization(self.__class__.required_parameters)

	def check_parameterization(self,required_params):
		""" Check that all the required parameters are present in the
		los_parameters.

		Args:
			required_params ([str,...]): A list of strings containing the
				required parameters.
		"""
		if not all(elem in self.main_deflector_parameters.keys() for
			elem in required_params):
			raise ValueError('Not all of the required parameters for the ' +
				'parameterization are present.')

	def update_parameters(self,main_deflector_parameters=None,
		cosmology_parameters=None):
		"""Updated the class parameters

		Args:
			los_parameters (dict): A dictionary containing the type of
				los distribution and the value for each of its parameters.
			main_deflector_parameters (dict): A dictionary containing the type
				of main deflector and the value for each of its parameters.
			source_parameters (dict): A dictionary containing the type of the
				source and the value for each of its parameters.
			cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
				Either a name of colossus cosmology, a dict with 'cosmology name':
				name of colossus cosmology, an instance of colussus cosmology, or
				a dict with H0 and Om0 ( other parameters will be set to
				defaults).
		"""
		if main_deflector_parameters is not None:
			self.main_deflector_parameters = copy.copy(
				main_deflector_parameters)
		if cosmology_parameters is not None:
			self.cosmo = get_cosmology(cosmology_parameters)

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
