# -*- coding: utf-8 -*-
"""
Provides the base class for specifying the source of a lensing system.

This module contains the base class that all the source classes will build
from. Because the steps for rendering a source can vary between different
models, the required functions are very sparse.
"""
import copy
from ..Utils.cosmology_utils import get_cosmology


class SourceBase:
	"""
	Base class for producing lenstronomy LightModel arguments

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters (dict): dictionary with source-specific parameters
	"""

	required_parameters = tuple()

	def __init__(self, cosmology_parameters, source_parameters):
		self.cosmo = get_cosmology(cosmology_parameters)
		self.source_parameters = copy.deepcopy(source_parameters)

		# Check that all the required parameters are present
		self.check_parameterization(self.__class__.required_parameters)

	def check_parameterization(self, required_params):
		""" Check that all the required parameters are present in the
		source_parameters.

		Args:
			required_params ([str,...]): A list of strings containing the
				required parameters.
		"""
		if not all(elem in self.source_parameters.keys() for
			elem in required_params):
			raise ValueError('Not all of the required parameters for the ' +
				'parameterization are present.')

	def update_parameters(self, cosmology_parameters=None,source_parameters=None):
		"""Updated the class parameters

		Args:
			cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
				Either a name of colossus cosmology, a dict with 'cosmology name':
				name of colossus cosmology, an instance of colussus cosmology, or
				a dict with H0 and Om0 ( other parameters will be set to
				defaults).
			source_parameters (dict): A dictionary containing all the parameters
				needed to draw sources.
		"""
		if source_parameters is not None:
			self.source_parameters.update(source_parameters)
		if cosmology_parameters is not None:
			self.cosmo = get_cosmology(cosmology_parameters)

	def draw_source(self):
		"""Return lenstronomy LightModel names and kwargs

		Returns:
			(list,list) A list containing the model name(s), and
			a list containing the model kwargs dictionaries.
		"""
		raise NotImplementedError
