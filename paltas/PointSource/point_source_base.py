# -*- coding: utf-8 -*-
"""
Provides the base class for specifying the point source of a lensing system.

This module contains the base class that all the point source classes will 
build from. Because the steps for rendering a source can vary between different
models, the required functions are very sparse.
"""
import paltas


class PointSourceBase(paltas.BaseComponent):
	"""
	Base class for producing lenstronomy PointSource arguments

	Args:
		point_source_parameters (dict): dictionary with source-specific 
			parameters.

	Notes:
		Has no required parameters by default.
	"""

	init_kwargs = ('point_source_parameters',)

	def draw_point_source(self):
		"""Return lenstronomy PointSource names and kwargs

		Returns:
			(list,list) A list containing the model name(s), and
			a list containing the model kwargs dictionaries.
		"""
		raise NotImplementedError

	def draw(self, result, **kwargs):
		result.add_point_sources(*self.draw_point_source())
