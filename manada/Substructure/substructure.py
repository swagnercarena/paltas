import numpy as np


def draw_subhalos(subhalo_parameters,mass_cocentration_parameters,
	main_deflector_parameters,cosmology):
	""" Given the parameters of the subhalo mass function, the
		mass-concentration relation, the main deflector lens parameters
		draw masses, concentrations,and positions for athe subhalos of a main
		lens halo.

	Parameters:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo mass profile and the value for each of its parameters.
		mass_cocentration_parameters (dict): A dictionary containing the type of
			mass-concentration relation and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		cosmology (dict): Either a dictionary containing the cosmology
			parameters or a string to be passed to collosus.
	Returns
	-------
	tuple
		A tuple of two lists: the first is the profile type for each subhalo returned
			and the second is the kwargs for that subhalo.
	"""
	# Initialize the lists that will contain our mass profile types and
	# assosciated kwargs. If no subhalos are drawn, these will remain empty
	subhalo_model_list = []
	subhalo_kwargs_list = []

	# Draw the masses

	return (subhalo_model_list, subhalo_kwargs_list)
