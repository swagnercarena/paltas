# -*- coding: utf-8 -*-
"""
Define the base class to draw line of sight substructure for a lens

This module contains the base class that all the los classes will build
from. Because the steps for rendering halos can vary between different
models, the required functions are very sparse.
"""
from ..Utils.cosmology_utils import get_cosmology
import copy


class LOSBase():
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

	required_parameters = tuple()

	def __init__(self,los_parameters,main_deflector_parameters,
		source_parameters,cosmology_parameters):

		# Save the parameters as a copy to avoid any misunderstanding on the
		# user end.
		self.los_parameters = copy.deepcopy(los_parameters)
		self.main_deflector_parameters = copy.deepcopy(
			main_deflector_parameters)
		self.source_parameters = copy.deepcopy(source_parameters)

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
		if not all(elem in self.los_parameters.keys() for
			elem in required_params):
			raise ValueError('Not all of the required parameters for the ' +
				'parameterization are present.')

	def update_parameters(self,los_parameters=None,
		main_deflector_parameters=None,source_parameters=None,
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
		if los_parameters is not None:
			self.los_parameters = copy.copy(los_parameters)
		if main_deflector_parameters is not None:
			self.main_deflector_parameters = copy.copy(
				main_deflector_parameters)
		if source_parameters is not None:
			self.source_parameters = copy.copy(source_parameters)
		if cosmology_parameters is not None:
			self.cosmo = get_cosmology(cosmology_parameters)

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
		""" Calculates the average convergence from the los at each redshift
		specified by the los parameters and returns corresponding lenstronomy
		objects.

		Returns:
			(tuple): A tuple of two lists: the first is the interpolation
			profile type for each redshift slice and the second is the
			lenstronomy kwargs for that profile.
		"""
		raise NotImplementedError
