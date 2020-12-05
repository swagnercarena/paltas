# -*- coding: utf-8 -*-
"""
Draw subhalo masses and concentrations for NFW subhalos

This module contains the functions needed to turn the parameters of NFW
subhalo distributions into masses, concentrations, and positions for those
NFW subhalos.
"""
import numpy as np
from manada.Utils import power_law

draw_nfw_masses_DG_19_parameters = ['sigma_sub','shmf_plaw_index','m_pivot',
	'm_min','m_max']


def host_scaling_function_DG_19(host_m200, z_lens, k1=0.88, k2=1.7):
	""" Scaling for the subhalo mass function based on the mass of the host
		halo. Derived from galacticus in https://arxiv.org/pdf/1909.02573.pdf.

	Parameters:
		host_m200 (float): The mass of the host halo in units of M_sun
		z_lens (flat): The redshift of the host halo / main deflector
		k1 (flaot): Amplitude of halo mass dependence
		k2 (flaot): Amplitude of the redshift scaling
	Returns:
		(float): The normalization scaling for the subhalo mass function
	Notes:
		Default values of k1 and k2 are derived from galacticus.
	"""
	# Equation from DG_19
	log_f = k1 * np.log10(host_m200/1e-13) + k2 * np.log10(z_lens+0.5)
	return 10**log_f


def draw_nfw_masses_DG_19(subhalo_parameters,main_deflector_parameters,cosmo):
	""" Draw from the https://arxiv.org/pdf/1909.02573.pdf mass function and
		return a list of the masses.

	Parameters:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object that will be used to calculate redshift
			dependent quantities.
	Returns:
		(list): The masses of the drawn halos
	"""

	# Check that we have all the parameters we need
	if not all(elem in draw_nfw_masses_DG_19_parameters for elem in
		subhalo_parameters.keys()):
		raise ValueError('Not all of the required parameters for the DG_19' +
			'parameterization are present.')

	# Pull the parameters we need from the input dictionaries
	# Units of m_sun times inverse kpc^2
	sigma_sub = subhalo_parameters['sigma_sub']
	shmf_plaw_index = subhalo_parameters['shmf_plaw_index']
	# Units of m_sun
	m_pivot = subhalo_parameters['m_pivot']
	# Units of m_sun
	host_m200 = main_deflector_parameters['M200']
	# Units of m_sun
	m_min = subhalo_parameters['m_min']
	# Units of m_sun
	m_max = subhalo_parameters['m_max']
	z_lens = main_deflector_parameters['z_lens']

	# Calculate the overall norm of the power law. This includes host scaling,
	# sigma_sub, and the area of interest.
	f_host = host_scaling_function_DG_19(host_m200,z_lens)

	# In DG_19 subhalos are rendered up until 3*theta_E
	r_E = (cosmo.angularDiameterDistance(z_lens)*
		main_deflector_parameters['theta_E'])
	dA = np.pi * r_E**2

	# We can also fold in the pivot mass into the norm for simplicity (then
	# all we have to do is sample from a power law).
	norm = f_host*dA*sigma_sub*m_pivot**(-shmf_plaw_index-1)

	# Draw from our power law and return the masses.
	masses = power_law.power_law_draw(m_min,m_max,shmf_plaw_index,norm)
	return masses
