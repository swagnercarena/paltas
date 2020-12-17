# -*- coding: utf-8 -*-
"""
Add to the functionality of colossus

Useful functions for extra cosmology calculations.
"""
from colossus.cosmology import cosmology
import numpy as np


def get_cosmology(cosmology_parameters):
	"""Returns colossus cosmology

	Args:
		cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).

	Returns:
		(colossus.cosmology.cosmology.Cosmology): A corresponding instance
		of the colossus cosmology class.
	"""
	if isinstance(cosmology_parameters, cosmology.Cosmology):
		return cosmology_parameters
	if isinstance(cosmology_parameters, str):
		return cosmology.setCosmology(cosmology_parameters)
	if isinstance(cosmology_parameters, dict):
		if 'cosmology_name' in cosmology_parameters:
			return get_cosmology(cosmology_parameters['cosmology_name'])
		else:
			# Leave some parameters to their default values so the user only
			# has to specify H0 and Om0.
			col_params = dict(flat=True, H0=cosmology_parameters['H0'],
				Om0=cosmology_parameters['Om0'],Ob0=0.049,sigma8=0.81,ns=0.95)
			return cosmology.setCosmology('temp_cosmo', col_params)


def kpc_per_arcsecond(z,cosmo):
	"""Calculates the physical kpc per arcsecond at a given redshift and
	cosmology

	Args:
		z (float): The redshift to calculate the distance at
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.

	Returns:
		(float): The kpc per arcsecond
	"""
	h = cosmo.h
	kpc_per_arcsecond = (cosmo.angularDiameterDistance(z) *np.pi/180/3600 /
		h * 1e3)
	return kpc_per_arcsecond
