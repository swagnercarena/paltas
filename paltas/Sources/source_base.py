# -*- coding: utf-8 -*-
"""
Provides the base class for specifying the source of a lensing system.

This module contains the base class that all the source classes will build
from. Because the steps for rendering a source can vary between different
models, the required functions are very sparse.
"""
import copy
from ..Utils.cosmology_utils import get_cosmology

import paltas


class SourceBase(paltas.BaseComponent):
	"""
	Base class for producing lenstronomy LightModel arguments

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters (dict): dictionary with source-specific parameters
	"""

	# Historical oversight: cosmology parameters as first arg...
	init_kwargs = ('cosmology_parameters', 'source_parameters')
	main_param_dict_name = 'source_parameters'

	def draw_source(self):
		"""Return lenstronomy LightModel names and kwargs

		Returns:
			(list,list,list) A list containing the model name(s),
			a list containing the model kwargs dictionaries, and a list
			containing the redshifts of each model. Redshifts list can
			be None.
		"""
		raise NotImplementedError

	def draw(self, result, lens_light=False, **kwargs):
		if lens_light:
			result.add_lens_light(*self.draw_source())
		else:
			result.add_sources(*self.draw_source())
