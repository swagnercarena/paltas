# -*- coding: utf-8 -*-
"""
Add to the functionality of colossus

Useful functions for extra cosmology calculations.
"""
import numpy as np


def kpc_per_arcsecond(z,cosmo):
	"""
	Calculate the physical kpc per arcsecond at a given redshift and cosmology

	Parameters:
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
