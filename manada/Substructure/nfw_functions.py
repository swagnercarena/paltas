# -*- coding: utf-8 -*-
"""
Draw subhalo masses and concentrations for NFW subhalos

This module contains the functions needed to turn the parameters of NFW
subhalo distributions into masses, concentrations, and positions for those
NFW subhalos.
"""
import numpy as np
from manada.Utils import power_law, cosmology_utils
import numba
from scipy.interpolate import interp1d
from colossus.halo.concentration import peaks

draw_nfw_masses_DG_19_parameters = ['sigma_sub','shmf_plaw_index','m_pivot',
	'm_min','m_max','c_0','conc_xi','conc_beta','conc_m_ref',
	'dex_scatter']


# -------------------------------Mass Functions-------------------------------

@numba.njit()
def host_scaling_function_DG_19(host_m200, z_lens, k1=0.88, k2=1.7):
	"""
	Scaling for the subhalo mass function based on the mass of the host
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
	log_f = k1 * np.log10(host_m200/1e13) + k2 * np.log10(z_lens+0.5)
	return 10**log_f


def draw_nfw_masses_DG_19(subhalo_parameters,main_deflector_parameters,cosmo):
	"""
	Draw from the https://arxiv.org/pdf/1909.02573.pdf mass function and
	return an array of the masses.

	Parameters:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.
	Returns:
		(np.array): The masses of the drawn halos
	"""

	# Check that we have all the parameters we need
	if not all(elem in subhalo_parameters.keys() for
		elem in draw_nfw_masses_DG_19_parameters):
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

	# In DG_19 subhalos are rendered up until 3*theta_E.
	# Colossus return in MPC per h per radian so must be converted to kpc per
	# arc second
	kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z_lens,cosmo)
	r_E = (kpc_per_arcsecond*main_deflector_parameters['theta_E'])
	dA = np.pi * (3*r_E)**2

	# We can also fold in the pivot mass into the norm for simplicity (then
	# all we have to do is sample from a power law).
	norm = f_host*dA*sigma_sub*m_pivot**(-shmf_plaw_index-1)

	# Draw from our power law and return the masses.
	masses = power_law.power_law_draw(m_min,m_max,shmf_plaw_index,norm)
	return masses


def mass_concentration_DG_19(subhalo_parameters,z,m_200,cosmo):
	"""
	Return the concentration of halos at a certain mass given the
	parameterization of DG_19.

	Parameters:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		z (float): The redshift of the nfw halo
		m_200 (np.array): array of M_200 of the nfw halo units of M_sun
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.
	"""
	# Get the concentration parameters
	c_0 = subhalo_parameters['c_0']
	xi = subhalo_parameters['conc_xi']
	beta = subhalo_parameters['conc_beta']
	m_ref = subhalo_parameters['conc_m_ref']
	dex_scatter = subhalo_parameters['dex_scatter']

	# The peak calculation is done by colossus. The cosmology must have
	# already been set. Note these functions expect M_sun/h units (which
	# you get by multiplying by h
	# https://www.astro.ljmu.ac.uk/~ikb/research/h-units.html)
	h = cosmo.h
	peak_heights = peaks.peakHeight(m_200*h,z)
	peak_height_ref = peaks.peakHeight(m_ref*h,0)

	# Now get the concentrations and add scatter
	concentrations = c_0*(1+z)**(xi)*(peak_heights/peak_height_ref)**(-beta)
	if isinstance(concentrations,np.ndarray):
		conc_scatter = np.random.randn(len(concentrations))*dex_scatter
	elif isinstance(concentrations,float):
		conc_scatter = np.random.randn()*dex_scatter
	concentrations = 10**(np.log10(concentrations)+conc_scatter)

	return concentrations


# -----------------------------Position Functions-----------------------------

@numba.njit()
def cored_nfw_integral(r_tidal,rho_nfw,r_scale,r_upper):
	"""
	Integrate the cored nfw profile from 0 to r_upper

	Parameters:
		r_tidal (float): The tidal radius within which the NFW profile
			will be replaced by a uniform profile.
		rho_nfw (float): The amplitude of the nfw density outside the
			cored radius.
		r_scale (float): The scale radius of the nfw
		r_upper (np.array): An array containing the upper bounds
			to be evaluated.
	Returns:
		(np.array): The value of the integral for each r_upper given.
	"""
	# Convert to natural units for NFW
	x_tidal = r_tidal / r_scale
	x_upper = r_upper / r_scale

	# Get the value of the NFW in the core region
	uniform_value = rho_nfw/(x_tidal*(1+x_tidal)**2)

	# Array to save the integral outputs to
	integral_values = np.zeros(r_upper.shape)

	# Add the cored component
	integral_values += uniform_value * np.minimum(r_tidal,r_upper)

	# Add the nfw component where x_upper > x_tidal
	lower_bound = 1/(x_tidal+1) + np.log(x_tidal) - np.log(x_tidal+1)
	upper_bound = 1/(x_upper+1) + np.log(x_upper) - np.log(x_upper+1)
	nfw_integral = upper_bound - lower_bound
	add_nfw = r_upper > r_tidal
	integral_values[add_nfw] += nfw_integral[add_nfw]*rho_nfw*r_scale

	return integral_values


def cored_nfw_draws(r_tidal,rho_nfw,r_scale,r_max,n_subs,n_cdf_samps=1000):
	"""
	Return radial samples from a cored nfw profile

	Parameters:
		r_tidal (float): The tidal radius within which the NFW profile
			will be replaced by a uniform profile.
		rho_nfw (float): The amplitude of the nfw density outside the
			cored radius.
		r_scale (float): The scale radius of the nfw
		r_max (float): The maximum value of r to sample
		n_subs (int): The number of subhalo positions to sample
		n_cdf_samps (int): The number of samples to use to numerically
			invert the cdf for sampling.
	Returns:
		(np.array): A n_subs array giving sampled radii.
	"""
	# First we have to numerically calculate the inverse cdf
	r_values = np.linspace(0,r_max,n_cdf_samps)
	cdf_values = cored_nfw_integral(r_tidal,rho_nfw,r_scale,r_values)
	# Normalize
	cdf_values /= np.max(cdf_values)
	# Use scipy to create our inverse cdf
	inverse_cdf = interp1d(cdf_values,r_values)

	# Now draw from the inverse cdf
	cdf_draws = np.random.rand(n_subs)
	r_draws = inverse_cdf(cdf_draws)

	return r_draws


def r_200_from_m(m_200,cosmo):
	"""
	Calculate r_200 for our NFW profile given our m_200 mass.

	Parameters:
		m_200 (np.array): The mass contained within r_200
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.
	Returns:
		(np.array): The r_200 radius corresponding to the given mass.
	"""
	# Get the critical density
	h = cosmo.h
	# rho_c is returned in units of M_sun*h^2/kpc^3
	rho_c = cosmo.rho_c(0)*h**2

	# Return r_200 given that critical density
	return (3*m_200/(4*np.pi*rho_c*200))**(1.0/3.0)


def rho_nfw_from_m_c(m_200,c,cosmo,r_scale=None):
	"""
	Calculate the amplitude of the nfw profile given the physical parameters.

	Parameters:
		m_200 (np.array): The mass of the nfw halo in units of M_sun
		c (np.array): The concentration of the nfw_halo
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.
		r_scale (np.array): The scale radius in units of kpc
	Returns:
		(np.array): The amplitude for the nfw.
	"""
	# If r_scale is not provided, calculate it
	if r_scale is None:
		r_200 = r_200_from_m(m_200,cosmo)
		r_scale = r_200/c

	# Calculate the density to match the mass and concentration.
	rho_0 = m_200/(4*np.pi*r_scale**3*(np.log(1+c)-c/(1+c)))

	return rho_0


def rejection_sampling_DG_19(r_samps,r_200,r_3E):
	"""
	Given the radial sampling of the positions and DG_19 constraints
	conduct rejection sampling and return the cartesian positions.

	Parameters:
		r_samps (np.array): Samples of the radial coordinates for
			the subhalos.
		r_200 (float): The r_200 of the host halo which will be used
			as the maximum z magnitude
		r_3E (float): 3 times the einstein radius, which will be used
			to bound the x and y coordinates.
	Returns:
		([np.array,...]): A list of two numpy arrays: the boolean
		array of accepted samples and a n_subsx3 array of x,y,z
		coordinates.
	"""
	# Sample theta and phi values for all of the radii samples
	theta = np.random.rand(len(r_samps)) * 2 * np.pi
	phi = np.random.rand(len(r_samps))*np.pi

	# Initialize the x,y,z array
	cart_pos = np.zeros(r_samps.shape+(3,))

	# Get the x, y, and z coordinates
	cart_pos[:,0] += r_samps*np.sin(phi)*np.cos(theta)
	cart_pos[:,1] += r_samps*np.sin(phi)*np.sin(theta)
	cart_pos[:,2] += r_samps*np.cos(phi)

	# Test which samples are outside the DG_19 bounds
	r2_inside = np.sqrt(cart_pos[:,0]**2+cart_pos[:,1]**2)<r_3E
	z_inside = np.abs(cart_pos[:,2])<r_200
	keep = np.logical_and(r2_inside,z_inside)

	return [keep,cart_pos]


def sample_cored_nfw_DG_19(r_tidal,subhalo_parameters,
	main_deflector_parameters,cosmo,n_subs):
	"""
	Given the a tidal radius that defines a core region and the parameters
	of the main deflector, sample positions for NFW subhalos bounded
	as described in https://arxiv.org/pdf/1909.02573.pdf

	Parameters:
		r_tidal (float): The tidal radius within which the NFW profile will
			be replaced by a uniform profile.
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.
		n_subs (int): The number of subhalo positions to sample
	Returns:
		(np.array): A n_subs x 3 array giving the x,y,z position of the
		subhalos in units of kpc.
	Notes:
		The code works through rejection sampling, which can be inneficient
		for certain configurations. If this is a major issue, it may be worth
		introducing more analytical components.
	"""

	# Check that we have all the parameters we need
	if not all(elem in subhalo_parameters.keys() for
		elem in draw_nfw_masses_DG_19_parameters):
		raise ValueError('Not all of the required parameters for the DG_19' +
			'parameterization are present.')

	# Create an array that will store our coordinates
	cart_pos = np.zeros((n_subs,3))

	host_m200 = main_deflector_parameters['M200']
	z_lens = main_deflector_parameters['z_lens']
	host_c = mass_concentration_DG_19(subhalo_parameters,z_lens,host_m200,
		cosmo)
	host_r_200 = r_200_from_m(host_m200,cosmo)
	host_r_scale = host_r_200/host_c
	host_rho_nfw = rho_nfw_from_m_c(host_m200,host_c,cosmo,
		r_scale=host_r_scale)

	# Tranform the einstein radius to physical units (TODO this should
	# be a function). Multiply by 3 since that's what's relevant for
	# DG_19 parameterization.
	kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z_lens,cosmo)
	r_3E = (kpc_per_arcsecond*main_deflector_parameters['theta_E'])*3

	# The largest radius we should bother sampling is set by the diagonal of
	# our cylinder.
	r_max = np.sqrt(r_3E**2+host_r_200**2)

	n_accepted_draws = 0
	r_subs = cored_nfw_draws(r_tidal,host_rho_nfw,host_r_scale,r_max,n_subs)
	keep_ind, cart_draws = rejection_sampling_DG_19(r_subs,host_r_200,r_3E)

	# Save the cartesian coordinates we want to keep
	cart_pos[n_accepted_draws:n_accepted_draws+np.sum(keep_ind)] = (
		cart_draws[keep_ind])
	n_accepted_draws += np.sum(keep_ind)

	# Get the fraction of rejection to see how much we should sample
	rejection_frac = 1-np.mean(keep_ind)

	# Keep drawing until we have enough r_subs.
	while n_accepted_draws<n_subs:
		r_subs = cored_nfw_draws(r_tidal,host_rho_nfw,host_r_scale,r_max,
			int(np.round(n_subs*rejection_frac)))
		keep_ind, cart_draws = rejection_sampling_DG_19(r_subs,host_r_200,
			r_3E)
		use_keep = np.minimum(n_subs-n_accepted_draws,np.sum(keep_ind))
		# Save the cartesian coordinates we want to keep
		cart_pos[n_accepted_draws:n_accepted_draws+use_keep] = (
			cart_draws[keep_ind][:use_keep])
		n_accepted_draws += use_keep

	return cart_pos
