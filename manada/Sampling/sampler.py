# -*- coding: utf-8 -*-
"""
Define the sampler class which reads goes from the distribution dictionary to
drawing samples on the lens parameters.

This module contains the class used to sample parameters for our train and test
set from the input distributions.
"""
import warnings
# Definte the components we need the sampler to consider.
# TODO: add point source & lens light
lensing_components = ['subhalo','los','main_deflector','source','lens_light',
	'point_source','lens_equation_solver','cosmology','psf','detector','drizzle']

# Global filters on the python warnings. Using this since filter
# behaviour is a bit weird.
CROSSOBJECTWARNING = True


class Sampler():
	"""Class for drawing lens parameter values from input distribution
	dictionaries

	Args:
		configuration_dictionary (dict): An instance of the configuration
			dictionary that will be used to decide how to sample parameter
			values.
	Notes:
		For examples on how to configure the dict object, see the
		ConfigurationDict.ipynb example notebook.
	"""

	def __init__(self,configuration_dictionary):
		self.config_dict = configuration_dictionary

	@staticmethod
	def draw_from_dict(draw_dict):
		"""Populates a dict with samples drawn from the specified distributions
		in the input dict.

		Args:
			draw_dict (dict): The dictionary containing keys mapping to values
				or distributions for each parameter.

		Returns:
			(dict): A dict with a drawn value for each parameter.

		Notes:
			Multivariate distribution for parameters should have a key of the
			form 'param_1,param_2,param_3'.
		"""
		param_dict = {}

		# Iterate through the keys in the draw_dict and populate the values of
		# param_dict correctly.
		for key in sorted(draw_dict):
			# If the key implies that multiple parameters will be drawn from
			# the distribution, draw the value and then iterate through the
			# parameters.
			if ',' in key:
				# Get the parameters, removing whitespace
				params = key.replace(' ','').split(',')
				# Draw the values
				draw = draw_dict[key]()
				# Check for consistency
				if len(params) != len(draw):
					raise ValueError('Parameters of length %d do'%(len(params))
						+ ' not match draw of length %d'%(len(draw)))
				# Populate the keys
				for i, param in enumerate(params):
					param_dict[param] = draw[i]
			# If it's a univariate function just call it.
			elif callable(draw_dict[key]):
				param_dict[key] = draw_dict[key]()
			# If it's a fixed value just populate it.
			else:
				param_dict[key] = draw_dict[key]

		return param_dict

	def sample(self):
		"""Samples from the distributions given in the configuration
		dictionary

		Returns:
			(dict): A dictionary containing the parameter values that will
			be sampled.
		"""
		# Pull the global warning variable and initialize our dict
		global CROSSOBJECTWARNING
		full_param_dict = {}

		# For each possible component of our lensing add the parameters
		for component in lensing_components:
			if component in self.config_dict:
				draw_dict = self.config_dict[component]['parameters']
				param_dict = self.draw_from_dict(draw_dict)
				full_param_dict[component+'_parameters'] = param_dict

		# Populate parameters from distributions that span accross objects.
		if 'cross_object' in self.config_dict:
			draw_dict = self.config_dict['cross_object']['parameters']
			cross_dict = self.draw_from_dict(draw_dict)
			# Go through the params and update the full dict
			for cross_param in cross_dict:
				component, param = cross_param.split(':')
				param_dict = full_param_dict[component+'_parameters']
				# Warn the user the first time an overwrite happens
				if param in param_dict and CROSSOBJECTWARNING:
					warnings.warn('Parameter in cross dict specified elsewhere!'
						+ ' Will be overwritten')
					CROSSOBJECTWARNING = False
				full_param_dict[component+'_parameters'][param] = (
					cross_dict[cross_param])

		# Populate the cross objects
		return full_param_dict
