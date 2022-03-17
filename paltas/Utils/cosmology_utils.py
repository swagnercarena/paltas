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
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or
			a dict with H0 and Om0 ( other parameters will be set to
			defaults).

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
				Om0=cosmology_parameters['Om0'],
				Ob0=cosmology_parameters['Ob0'],
				sigma8=cosmology_parameters['sigma8'],
				ns=cosmology_parameters['ns'])
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


def ddt(sample,cosmo):
	"""Calculates time delay distance given lens redshift, source redshift, and
	cosmology.

	Args: 
		sample (dict): Dictionary containing dictionaries of parameters for each
			model component. Generated using Sampler .sample() method
		cosmo (colossus.cosmology.Cosmology): An instance of the colossus
			cosmology object
	
	Returns:
		(float): Time delay distance
	"""
	z_lens = sample['main_deflector_parameters']['z_lens']
	z_source = sample['source_parameters']['z_source']
	D_d = cosmo.angularDiameterDistance(z_lens)
	D_s = cosmo.angularDiameterDistance(z_source)
	D_ds =  (1/ (1+z_source)) * cosmo.comovingDistance(z_min=z_lens,
		z_max=z_source)
	# convert from Mpc/h to Mpc 
	return (1+z_lens) * D_d * D_s / (D_ds*cosmo.h)
