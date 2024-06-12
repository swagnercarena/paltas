# -*- coding: utf-8 -*-
"""
Provide the implementation of a perturber that consists
of two or fewer components.

This module contains the classes to render perturbers consisting of simple
combinations of lenstronomy profiles.
"""
from .main_deflector_base import MainDeflectorBase
from lenstronomy.LensModel.profile_list_base import lens_class


class Perturber(MainDeflectorBase):
        """Class for rendering a perturber that includes a PEMD profile, an external shear and SIS model.

	Args:
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	Notes:
		Uses the lenstronomy EPL class, which is equivalent to PEMD but is pure
		python.

	Required Parameters

	- gamma - power law slope
	- theta_E - Einstein radius of the main deflector profile in units of arcseconds
	- e1 - x-direction ellipticity eccentricity
	- e2 - xy-direction ellipticity eccentricity
	- center_x - x-coordinate lens center in units of arcseconds
	- center_y - y-coordinate lens center in units of arcseconds
	- gamma1 - x-direction shear
	- gamma2 - xy-direction shear
	- ra_0 - ra origin of shear in units of arcseconds
	- dec_0 - dec origin of shear in units of arcseconds
	- z_lens - main deflector redshift
 	- p_theta_E - Einstein radius of the perturber profile in units of arcseconds
 	- p_center_x - x-coordinate perturber center in units of arcseconds
  	- p_center_y - y-coordinate perturber center in units of arcseconds
	"""
	# Define the parameters we expect to find for the DG_19 model
	required_parameters = ('gamma','theta_E','e1','e2','center_x','center_y','gamma1','gamma2','ra_0','dec_0','z_lens','p_theta_E','p_center_x','p_center_y')

	def __init__(self,main_deflector_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(main_deflector_parameters,cosmology_parameters)

	def draw_main_deflector(self):
		"""Draws the lenstronomy profile names and kwargs for the components
		of the main deflector.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each component of the perturber, the second is the
			lenstronomy kwargs for each component, and the third is the
			redshift for each component.
		"""
		# The lists of model parameters, kwargs, and redshifts can all be
		# pulled fairly directly from the main_deflecctor_parameters
		md_model_list = ['EPL_NUMBA','SHEAR','SIS']
		md_kwargs_list = []
		md_z_list = [self.main_deflector_parameters['z_lens']] * len(
			md_model_list)

		# Use lenstronomy to sort the parameters
		for model in md_model_list:
			# The list of parameters linked to that lenstronomy model
			if model == 'SIS':
				p_names = ['p_theta_E','p_center_x','p_center_y']
			else:
				p_names = (
					lens_class(model).param_names
				)
			model_kwargs = {}
			for param in p_names:
				model_kwargs[param] = (self.main_deflector_parameters[param])
			md_kwargs_list += [model_kwargs]

		return md_model_list, md_kwargs_list, md_z_list
