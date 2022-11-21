# -*- coding: utf-8 -*-
"""
Provides the base class for specifying the source of a lensing system.

This module contains the base class that all the source classes will build
from. Because the steps for rendering a source can vary between different
models, the required functions are very sparse.
"""
import copy

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
	is_lens_light = False

	def draw_source(self):
		"""Return lenstronomy LightModel names and kwargs

		Returns:
			(list,list,list) A list containing the model name(s),
			a list containing the model kwargs dictionaries, and a list
			containing the redshifts of each model. Redshifts list can
			be None.
		"""
		raise NotImplementedError

	def draw(self, result, **kwargs):
		if self.is_lens_light:
			result.add_lens_light(*self.draw_source())
		else:
			result.add_sources(*self.draw_source())


def make_lens_light_class(cls):
	"""Return a lens light class corresponding to a source class"""
	if cls is None:
		return None
	lens_cls = copy.copy(cls)
	lens_cls.is_lens_light = True

	# Make the class eat lens light parameters
	lens_cls.init_kwargs = ([
		'lens_light_parameters' if x == 'source_parameters' else x
		for x in cls.init_kwargs])
	lens_cls.main_param_dict_name = 'lens_light_parameters'

	# Any code pointing to source_parameters should go to lens_light_parameters
	lens_cls.source_parameters = property(lambda self: self.lens_light_parameters)
	
	# Have to redefine init, lens_light_parameters may be passed as kwarg..	
	old_init = lens_cls.__init__
	def new_init(self, cosmology_parameters, lens_light_parameters):
		return old_init(self, cosmology_parameters, lens_light_parameters)
	lens_cls.__init__ = new_init

	return lens_cls