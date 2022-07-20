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


def absolute_to_apparent(mag_absolute,z_light,cosmo,
	include_k_correction=True):
	"""Converts from absolute magnitude to apparent magnitude.

	Args:
		mag_apparent (float): The absolute magnitude
		z_light (float): The redshift of the light
		cosmo (colossus.cosmology.Cosmology): An instance of the colossus
			cosmology object
		include_k_correction (bool): If true apply an approximate k
			correction for a galaxy-like source.

	Returns:
		(float): The absolute magnitude of the light
	"""
	# Use the luminosity distance for the conversion
	lum_dist = cosmo.luminosityDistance(z_light)
	# Convert from Mpc/h to pc
	lum_dist *= 1e6/cosmo.h

	mag_apparent = mag_absolute + 5 *np.log10(lum_dist/10)

	# Calculate the k_correction if requested
	if include_k_correction:
		mag_apparent += get_k_correction(z_light)

	return mag_apparent


def get_k_correction(z_light):
	"""Get the k correction for a galaxy source of light at a given redshift.

	Args:
		z_light (float): The redshift of the light

	Returns:
		(float): The k correction such that m_apparent = M_absolute + DM +
		K_corr.

	Notes:
		This code assumes the galaxy has a flat spectral wavelength density (and
		therefore 1/nu^2 spectral frequency density) and that the bandpass used
		for the absolute and apparent magntidue is the same.
	"""

	return 2.5 * np.log(1+z_light)
