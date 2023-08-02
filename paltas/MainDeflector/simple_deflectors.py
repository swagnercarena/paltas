# -*- coding: utf-8 -*-
"""
Provide the implementation of a few main deflector profiles that consist
of two or fewer components.

This module contains the classes to render main deflectors consisting of simple
combinations of lenstronomy profiles.
"""
from .main_deflector_base import MainDeflectorBase
from lenstronomy.LensModel.profile_list_base import ProfileListBase


class PEMD(MainDeflectorBase):
	"""Class for rendering a main deflector the includes a PEMD profile.

	Args:
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	Notes:
		Uses the lenstronomy EPL_NUMBA class, which is equivalent to PEMD but is
		pure python.

	Required Parameters

	- gamma - power law slope
	- theta_E - Einstein radius of the profile in units of arcseconds
	- e1 - x-direction ellipticity eccentricity
	- e2 - xy-direction ellipticity eccentricity
	- center_x - x-coordinate lens center in units of arcseconds
	- center_y - y-coordinate lens center in units of arcseconds
	- z_lens - main deflector redshift
	"""
	# Define the parameters we expect to find for the DG_19 model
	required_parameters = ('gamma','theta_E','e1','e2','center_x',
		'center_y','z_lens')

	def __init__(self,main_deflector_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(main_deflector_parameters,cosmology_parameters)

	def draw_main_deflector(self):
		"""Draws the lenstronomy profile names and kwargs for the components
		of the main deflector.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each component of the main deflector, the second is the
			lenstronomy kwargs for each component, and the third is the
			redshift for each component.
		"""
		# The lists of model parameters, kwargs, and redshifts can all be
		# pulled fairly directly from the main_deflecctor_parameters
		md_model_list = ['EPL_NUMBA']
		md_kwargs_list = []
		md_z_list = [self.main_deflector_parameters['z_lens']] * len(
			md_model_list)

		# Use lenstronomy to sort the parameters
		for model in md_model_list:
			# The list of parameters linked to that lenstronomy model
			p_names = ProfileListBase._import_class(model,None,None).param_names
			model_kwargs = {}
			for param in p_names:
				model_kwargs[param] = (self.main_deflector_parameters[param])
			md_kwargs_list += [model_kwargs]

		return md_model_list, md_kwargs_list, md_z_list


class PEMDShear(MainDeflectorBase):
	"""Class for rendering a main deflector the includes a PEMD profile and
	external shear.

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
	- theta_E - Einstein radius of the profile in units of arcseconds
	- e1 - x-direction ellipticity eccentricity
	- e2 - xy-direction ellipticity eccentricity
	- center_x - x-coordinate lens center in units of arcseconds
	- center_y - y-coordinate lens center in units of arcseconds
	- gamma1 - x-direction shear
	- gamma2 - xy-direction shear
	- ra_0 - ra origin of shear in units of arcseconds
	- dec_0 - dec origin of shear in units of arcseconds
	- z_lens - main deflector redshift
	"""
	# Define the parameters we expect to find for the DG_19 model
	required_parameters = ('gamma','theta_E','e1','e2','center_x',
		'center_y','gamma1','gamma2','ra_0','dec_0','z_lens')

	def __init__(self,main_deflector_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(main_deflector_parameters,cosmology_parameters)

	def draw_main_deflector(self):
		"""Draws the lenstronomy profile names and kwargs for the components
		of the main deflector.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each component of the main deflector, the second is the
			lenstronomy kwargs for each component, and the third is the
			redshift for each component.
		"""
		# The lists of model parameters, kwargs, and redshifts can all be
		# pulled fairly directly from the main_deflecctor_parameters
		md_model_list = ['EPL_NUMBA','SHEAR']
		md_kwargs_list = []
		md_z_list = [self.main_deflector_parameters['z_lens']] * len(
			md_model_list)

		# Use lenstronomy to sort the parameters
		for model in md_model_list:
			# The list of parameters linked to that lenstronomy model
			p_names = ProfileListBase._import_class(model,None,None,kwargs_synthesis=None).param_names
			model_kwargs = {}
			for param in p_names:
				model_kwargs[param] = (self.main_deflector_parameters[param])
			md_kwargs_list += [model_kwargs]

		return md_model_list, md_kwargs_list, md_z_list


class PEMDShearFourMultipole(PEMDShear):
	"""Class for rendering a main deflector the includes a PEMD profile,
	external shear, and order 2, 3, and 4 multipole.
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
	- theta_E - Einstein radius of the profile in units of arcseconds
	- e1 - x-direction ellipticity eccentricity
	- e2 - xy-direction ellipticity eccentricity
	- center_x - x-coordinate lens center in units of arcseconds
	- center_y - y-coordinate lens center in units of arcseconds
	- gamma1 - x-direction shear
	- gamma2 - xy-direction shear
	- ra_0 - ra origin of shear in units of arcseconds
	- dec_0 - dec origin of shear in units of arcseconds
	- z_lens - main deflector redshift
	- mult2_a - multipole strength for order 2 multpole
	- mult2_phi - multipole order 2 orientation in radians
	- mult2_center_x - x-coordinate multipole order 2 center in units of
	arcseconds
	- mult2_center_y - y-coordinate multipole order 2 center in units of
	arcseconds
	- mult3_a - multipole strength for order 2 multpole
	- mult3_phi - multipole order 2 orientation in radians
	- mult3_center_x - x-coordinate multipole order 2 center in units of
	arcseconds
	- mult3_center_y - y-coordinate multipole order 2 center in units of
	arcseconds
	- mult4_a - multipole strength for order 2 multpole
	- mult4_phi - multipole order 2 orientation in radians
	- mult4_center_x - x-coordinate multipole order 2 center in units of
	arcseconds
	- mult4_center_y - y-coordinate multipole order 2 center in units of
	arcseconds
	"""
	# Define the parameters we expect to find for the DG_19 model
	required_parameters = PEMDShear.required_parameters + ('mult2_a',
		'mult2_phi','mult2_center_x','mult2_center_y','mult3_a','mult3_phi',
		'mult3_center_x','mult3_center_y','mult4_a','mult4_phi',
		'mult4_center_x','mult4_center_y')

	def __init__(self,main_deflector_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(main_deflector_parameters,cosmology_parameters)

	def draw_main_deflector(self):
		"""Draws the lenstronomy profile names and kwargs for the components
		of the main deflector.
		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each component of the main deflector, the second is the
			lenstronomy kwargs for each component, and the third is the
			redshift for each component.
		"""
		# Get the first set of lists directly from the inherited class
		md_model_list, md_kwargs_list, md_z_list = (
			super().draw_main_deflector())
		# Now add the three multipole models.
		for mult_i in range(2,5):
			# Only add multipole models with non zero strength.
			if self.main_deflector_parameters['mult%d_a'%(mult_i)] > 0:
				md_model_list += ['MULTIPOLE']
				# Initialize the model kwargs for this multipole model
				model_kwargs = {'m':mult_i,
					'a_m':self.main_deflector_parameters['mult%d_a'%(mult_i)],
					'phi_m':self.main_deflector_parameters[
						'mult%d_phi'%(mult_i)],
					'center_x':self.main_deflector_parameters[
						'mult%d_center_x'%(mult_i)],
					'center_y':self.main_deflector_parameters[
						'mult%d_center_y'%(mult_i)]}
				md_kwargs_list += [model_kwargs]
				md_z_list += [self.main_deflector_parameters['z_lens']]

		return md_model_list, md_kwargs_list, md_z_list
