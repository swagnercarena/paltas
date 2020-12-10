# -*- coding: utf-8 -*-
"""
Draw subhalos for a specific subhalo distribution

This module contains the functions needed to draw subhalos given a specific
subhalo parameterization and transform those parameters into lens models and
kwargs to pass into lenstronomy.
"""
from manada.Substructure import nfw_functions
from colossus.cosmology import cosmology


def draw_subhalos(subhalo_parameters,main_deflector_parameters,
	cosmology_parameters):
	"""
	Given the parameters of the subhalo mass distribution the main deflector
	lens parameters draw masses, concentrations,and positions for the
	subhalos of a main lens halo.

	Parameters:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		cosmology_parameters (dict): Either a dictionary containing the
			cosmology parameters or a string to be passed to collosus.

	Returns:
		(tuple): A tuple of two lists: the first is the profile type for each
		subhalo returned and the second is the kwargs for that subhalo.
	"""
	# Initialize the lists that will contain our mass profile types and
	# assosciated kwargs. If no subhalos are drawn, these will remain empty
	subhalo_model_list = []
	subhalo_kwargs_list = []

	# Initialize the cosmology
	if 'cosmology_name' in cosmology_parameters:
		cosmo = cosmology.setCosmology(cosmology_parameters['cosmology_name'])
	else:
		# Leave some parameters to their default values so the user only has
		# to specify H0 and Om0.
		col_params = {'flat': True, 'H0':cosmology_parameters['H0'],
		'Om0':cosmology_parameters['Om0'], 'Ob0': 0.049, 'sigma8': 0.81,
		'ns': 0.95}
		cosmo = cosmology.setCosmology('temp_cosmo', col_params)

	# Draw the model_list and kwargs depending on the type of subhalo
	# distribution
	# Distribute subhalos according to https://arxiv.org/pdf/1909.02573.pdf
	if subhalo_parameters['distribution'] == 'DG_19':
		# DG_19 assumes NFWs distributed throughout the main deflector.
		# For these NFWs we need positions, masses, and concentrations that
		# we will then translate to Lenstronomy parameters.
		subhalo_masses = nfw_functions.draw_nfw_masses_DG_19(
			subhalo_parameters,main_deflector_parameters,cosmo)
		subhalo_cart_pos = nfw_functions.sample_cored_nfw_DG_19(
			subhalo_parameters,main_deflector_parameters,cosmo,
			len(subhalo_masses))
		model_list, kwargs_list = nfw_functions.convert_to_lenstronomy_DG_19(
			subhalo_parameters,main_deflector_parameters,cosmo,subhalo_masses,
			subhalo_cart_pos)
		subhalo_model_list += model_list
		subhalo_kwargs_list += kwargs_list
	else:
		raise ValueError('Provided subhalo distribution %s not within' +
			'recognized distributions. Please implement this distribution' +
			'or use an already implemented distribution')

	return (subhalo_model_list, subhalo_kwargs_list)
