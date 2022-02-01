# -*- coding: utf-8 -*-
"""
Provides the base class for specifying the point source of a lensing system.

This module contains the base class that all the point source classes will 
build from. Because the steps for rendering a source can vary between different
models, the required functions are very sparse.
"""
import copy


class PointSourceBase:
	"""
	Base class for producing lenstronomy PointSource arguments

	Args:
		point_source_parameters (dict): dictionary with source-specific 
			parameters.

	Notes:
		Has no required parameters by default.
	"""

	required_parameters = tuple()

	def __init__(self, point_source_parameters):
		self.point_source_parameters = copy.deepcopy(point_source_parameters)

		# Check that all the required parameters are present
		self.check_parameterization(self.__class__.required_parameters)

	def check_parameterization(self, required_params):
		""" Check that all the required parameters are present in the
		point_source_parameters.

		Args:
			required_params ([str,...]): A list of strings containing the
				required parameters.
		"""
		if not all(elem in self.point_source_parameters.keys() for
			elem in required_params):
			raise ValueError('Not all of the required parameters for the ' +
				'parameterization are present.')

	def update_parameters(self, point_source_parameters=None):
		"""Update the class parameters

		Args:
			point_source_parameters (dict): A dictionary containing all the 
				parameters needed to draw point sources.
		"""
		if point_source_parameters is not None:
			self.point_source_parameters.update(point_source_parameters)

	def draw_point_source(self):
		"""Return lenstronomy PointSource names and kwargs

		Returns:
			(list,list) A list containing the model name(s), and
			a list containing the model kwargs dictionaries.
		"""
		raise NotImplementedError
