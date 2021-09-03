# -*- coding: utf-8 -*-
"""
Define functions useful for conversion between different NFW conventions

This module contains the functions used to move between NFW conventions and
to transform NFW parameters into lenstronomy inputs.
"""
import numpy as np
from ..Utils import cosmology_utils
import numba
from scipy.interpolate import interp1d
import lenstronomy.Util.constants as const


@numba.njit()
def cored_nfw_integral(r_tidal,rho_nfw,r_scale,r_upper):
	"""Integrates the cored nfw mass profile from 0 to r_upper

	Args:
		r_tidal (float): The tidal radius within which the NFW profile
			will be replaced by a uniform profile. Units of kpc
		rho_nfw (float): The amplitude of the nfw density outside the
			cored radius. Units of M_sun/kpc^3
		r_scale (float): The scale radius of the nfw in units of kpc
		r_upper (np.array): An array containing the upper bounds
			to be evaluated in units of kpc.

	Returns:
		(np.array): The value of the integral for each r_upper given.

	Notes:
		The cored NFW profile here assume uniform distribution within the
		tidal radius and then an NFW radial distribution outside the
		tidal radius.
	"""
	# Convert to natural units for NFW
	x_tidal = r_tidal / r_scale
	x_upper = r_upper / r_scale

	# Get the value of the NFW in the core region such that at x_tidal we
	# match the nfw profile.
	linear_scaling = rho_nfw/(x_tidal*(1+x_tidal)**2)

	# Array to save the integral outputs to
	integral_values = np.zeros(r_upper.shape)

	# Add the cored component
	integral_values += 1.0/3.0*linear_scaling * np.minimum(r_tidal,r_upper)**3

	# Add the nfw component where x_upper > x_tidal
	lower_bound = 1/(x_tidal+1) + np.log(x_tidal+1)
	upper_bound = 1/(x_upper+1) + np.log(x_upper+1)
	nfw_integral = upper_bound - lower_bound
	add_nfw = r_upper > r_tidal
	integral_values[add_nfw] += nfw_integral[add_nfw]*rho_nfw*r_scale**3

	# 4 pi factor from volume integral.
	integral_values *= 4 * np.pi

	return integral_values


def cored_nfw_draws(r_tidal,rho_nfw,r_scale,r_max,n_subs,n_cdf_samps=1000):
	"""Returns radial samples from a cored nfw mass profile

	Args:
		r_tidal (float): The tidal radius within which the NFW profile
			will be replaced by a uniform profile in units of kpc.
		rho_nfw (float): The amplitude of the nfw density outside the
			cored radius in units of M_sun / kpc^3.
		r_scale (float): The scale radius of the nfw in units of kpc.
		r_max (float): The maximum value of r to sample in units of kpc.
		n_subs (int): The number of subhalo positions to sample
		n_cdf_samps (int): The number of samples to use to numerically
			invert the cdf for sampling.

	Returns:
		(np.array): A n_subs array giving sampled radii in units of kpc.

	Notes:
		The cored NFW profile here assume uniform distribution within the
		tidal radius and then an NFW radial distribution outside the
		tidal radius.
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


@numba.njit()
def nfw_integral(rho_nfw,r_scale,r_upper):
	"""Integrates the nfw mass profile from 0 to r_upper.

	Args:
		rho_nfw (float): The amplitude of the nfw density outside the
			cored radius. Units of M_sun/kpc^3
		r_scale (float): The scale radius of the nfw in units of kpc
		r_upper (np.array): An array containing the upper bounds
			to be evaluated in units of kpc.

	Returns:
		(np.array): The value of the integral for each r_upper given.
	"""
	# Convert to natural units for NFW
	x_upper = r_upper / r_scale

	# Calculate the nfw integral
	integral_values=1/(x_upper+1) + np.log(x_upper+1)-1

	# Deal with the change of units to x in the integral and 4 pi from the
	# volume element.
	return integral_values*rho_nfw*r_scale**3*4*np.pi


def nfw_draws(rho_nfw,r_scale,r_max,n_subs,n_cdf_samps=1000):
	"""Returns radial samples from the nfw mass profile.


	Args:
		rho_nfw (float): The amplitude of the nfw density outside the
			cored radius in units of M_sun / kpc^3.
		r_scale (float): The scale radius of the nfw in units of kpc.
		r_max (float): The maximum value of r to sample in units of kpc.
		n_subs (int): The number of subhalo positions to sample
		n_cdf_samps (int): The number of samples to use to numerically
			invert the cdf for sampling.

	Returns:
		(np.array): A n_subs array giving sampled radii in units of kpc.
	"""
	# First we have to numerically calculate the inverse cdf
	r_values = np.linspace(0,r_max,n_cdf_samps)
	cdf_values = nfw_integral(rho_nfw,r_scale,r_values)
	# Normalize
	cdf_values /= np.max(cdf_values)
	# Use scipy to create our inverse cdf
	inverse_cdf = interp1d(cdf_values,r_values)

	# Now draw from the inverse cdf
	cdf_draws = np.random.rand(n_subs)
	r_draws = inverse_cdf(cdf_draws)

	return r_draws


@numba.njit()
def tnfw_integral(rho_nfw,r_scale,r_trunc,r_upper):
	"""Integrates the truncated nfw mass profile from 0 to r_upper. This is
	the same as the nfw profile but scaled by a factor of
	r_trunc^2/(r_trunc^2+r^2)

	Args:
		rho_nfw (float): The amplitude of the nfw density outside the
			cored radius. Units of M_sun/kpc^3
		r_scale (float): The scale radius of the nfw in units of kpc
		r_trunc (float): The truncation radius for the profile.
		r_upper (np.array): An array containing the upper bounds
			to be evaluated in units of kpc.

	Returns:
		(np.array): The value of the integral for each r_upper given.
	"""
	# Do the integral calculation in pieces
	integral_values = 1/(2*(r_upper+r_scale)*(r_scale**2+r_trunc**2)**2)
	integral_values *= r_scale**3 * r_trunc**2

	integral_values *= (-2*r_upper*(r_scale**2+r_trunc**2)+
		(r_upper+r_scale)*(4*r_scale*r_trunc*np.arctan(r_upper/r_trunc)+
			(r_scale-r_trunc)*(r_scale+r_trunc)*(
				2*np.log(r_scale)-2*(
					np.log(r_upper+r_scale)+np.log(r_trunc))+np.log(
					r_upper**2+r_trunc**2))))

	return integral_values*4*np.pi*rho_nfw


def tnfw_draws(rho_nfw,r_scale,r_trunc,r_max,n_subs,n_cdf_samps=1000):
	"""Returns radial samples from truncated nfw mass profile This is
	the same as the nfw profile but scaled by a factor of
	r_trunc^2/(r_trunc^2+r^2)


	Args:
		rho_nfw (float): The amplitude of the nfw density outside the
			cored radius in units of M_sun / kpc^3.
		r_scale (float): The scale radius of the nfw in units of kpc.
		r_trunc (float): The truncation radius for the profile.
		r_max (float): The maximum value of r to sample in units of kpc.
		n_subs (int): The number of subhalo positions to sample
		n_cdf_samps (int): The number of samples to use to numerically
			invert the cdf for sampling.

	Returns:
		(np.array): A n_subs array giving sampled radii in units of kpc.

	Notes:
		Set r_max to a value larger than r_trunc but not so large
		that you're wasting a lot of interpolation points for the inverse
		cdf.
	"""
	# First we have to numerically calculate the inverse cdf
	r_values = np.linspace(0,r_max,n_cdf_samps)
	cdf_values = tnfw_integral(rho_nfw,r_scale,r_trunc,r_values)
	# Normalize
	cdf_values /= np.max(cdf_values)
	# Use scipy to create our inverse cdf
	inverse_cdf = interp1d(cdf_values,r_values)

	# Now draw from the inverse cdf
	cdf_draws = np.random.rand(n_subs)
	r_draws = inverse_cdf(cdf_draws)

	return r_draws


def r_200_from_m(m_200,z,cosmo):
	"""Calculates r_200 for our NFW profile given our m_200 mass.

	Args:
		m_200 (float, or np.array): The mass contained within r_200 in units
			of M_sun
		z (float, or np.array): The redshift of the halo
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.

	Returns:
		(float, or np.array): The r_200 radius corresponding to the given mass
		in units of kpc.

	Notes:
		This equation assumes that for a halo at redshift z, m200 is
		defined in terms of the critical density at that redshift. Therefore
		the output to the equation is in physical units.
	"""
	# Get the critical density
	h = cosmo.h
	# rho_c is returned in units of M_sun*h^2/kpc^3
	rho_c = cosmo.rho_c(z)*h**2

	# Return r_200 given that critical density
	return (3*m_200/(4*np.pi*rho_c*200))**(1.0/3.0)


def rho_nfw_from_m_c(m_200,c,cosmo,r_scale=None,z=None):
	"""Calculates the amplitude of the nfw profile given the physical
	parameters.

	Args:
		m_200 (float, or np.array): The mass of the nfw halo in units of M_sun
		c (float, or np.array): The concentration of the nfw_halo
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.
		r_scale (float, or np.array): The scale radius in units of kpc
		z (float, or np.array): The redshift at which to calculate
			r_200.

	Returns:
		(float, or np.array): The amplitude for the nfw in units of
		M_sun/kpc^3.
	"""
	# If r_scale is not provided, calculate it
	if r_scale is None:
		if z is None:
			raise ValueError('Must specify z if not specifying r_scale')
		r_200 = r_200_from_m(m_200,z,cosmo)
		r_scale = r_200/c

	# Calculate the density to match the mass and concentration.
	rho_0 = m_200/(4*np.pi*r_scale**3*(np.log(1+c)-c/(1+c)))

	return rho_0


def m_c_from_rho_r_scale(rho_nfw,r_scale,cosmo,z):
	"""Calculates the mass and concentration from the amplitude of the
	nfw profile and scale radius.

	Args:
		rho_nfw (np.array): The amplitude of the nfw in units of M_sun/kpc^3
		r_scale (np.array): The scale radius in units of kpc
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.
		z (float): The redshift of the NFW halos.

	Returns:
		[np.array,...]: A list of 2 np.array objects. This first is the
		mass and the second the concentration for each NFW.
	"""
	# The equations do not have a simple closed form, so we're going to
	# cheat and interpolate
	c_eval = np.logspace(-1,3,1000)
	h = cosmo.h
	# rho_c is returned in units of M_sun*h^2/kpc^3
	rho_c = cosmo.rho_c(z)*h**2

	rho_nfw_evals = rho_c*200/3*c_eval**3/(np.log(1+c_eval)-c_eval/(1+c_eval))
	# Interpolate the inverse
	inverse_c = interp1d(rho_nfw_evals,c_eval)
	c = inverse_c(rho_nfw)

	# With the concentration we can get the mass
	r_200 = r_scale * c
	m = (4*np.pi*rho_c*200*r_200**3)/3

	return [m,c]


def calculate_sigma_crit(z,z_source,cosmo):
	"""Calculates the critical density for the given configuration

	Args:
		z (np.array): The redshift of the nfw
		z_source (float): The redshift of the source
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.

	Returns:
		(np.array): The critical surface density in units of M_sun/kpc^2
	"""
	# Normalization factor for sigma_crit
	norm = const.c ** 2 / (4 * np.pi * const.G)*const.Mpc / const.M_sun
	# Get our three angular diameter distances
	mpc2kpc = 1e3
	dd = cosmo.angularDiameterDistance(z)
	ds = cosmo.angularDiameterDistance(z_source)
	cosmo_astro = cosmo.toAstropy()
	dds = cosmo_astro.angular_diameter_distance_z1z2(z,z_source).value
	return ds / (dd * dds) * norm/mpc2kpc**2


def convert_to_lenstronomy_NFW(r_scale,z,rho_nfw,z_source,cosmo):
	"""Converts physical NFW parameters to parameters used by lenstronomy

	Args:
		r_scale (np.array): The scale radius of the nfw in units of kpc
		z (np.array): The redshift of the nfw
		rho_nfw (np.array): The amplitude of the nfw halos in units of
			M_sun/kpc^3
		z_source (float): The redshift of the source
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.

	Returns:
		[np.array,...]: A list of 2 numpy arrays: The angular r_scale and the
		deflection angle at the scale radius.
	"""
	kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,cosmo)

	# For two of our parameters we just need to convert to arcseconds
	r_scale_ang = r_scale/kpc_per_arcsecond

	# For our deflection angle the calculation is a little more involved
	# and requires knowledge of the source redshift.
	sigma_crit = calculate_sigma_crit(z,z_source,cosmo)
	alpha_rs = rho_nfw * (4*r_scale**2*(1+np.log(0.5)))
	alpha_rs /= kpc_per_arcsecond * sigma_crit
	return [r_scale_ang, alpha_rs]


def convert_from_lenstronomy_NFW(r_scale_ang,alpha_rs,z,z_source,cosmo):
	"""Converts parameters used by lenstronomy to physical NFW parameters

	Args:
		r_scale_ang (np.array): The scale radius of the nfw in angular
			units
		alpha_rs (np.array): The deflection angle at the scale radius
		z (np.array): The redshift of the nfw
		z_source (float): The redshift of the source
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.

	Returns:
		[np.array,...]: A list of 2 numpy arrays: The scale radius in kpc
		and the amplitude of the nfw halo in units of M_sun/kpc^3.
	"""
	kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,cosmo)

	# For the scale radius we just need to convert from arcseconds
	r_scale = r_scale_ang * kpc_per_arcsecond

	# Deflection angle to rho_nfw requires sigma_crit and source
	# position
	sigma_crit = calculate_sigma_crit(z,z_source,cosmo)
	rho_nfw = alpha_rs / (4*r_scale**2*(1+np.log(0.5)))
	rho_nfw *= kpc_per_arcsecond * sigma_crit
	return [r_scale,rho_nfw]


def convert_to_lenstronomy_tNFW(r_scale,z,rho_nfw,r_trunc,z_source,cosmo):
	"""Converts physical tNFW parameters to parameters used by lenstronomy

	Args:
		r_scale (np.array): The scale radius of the nfw in units of kpc
		z (np.array): The redshift of the nfw
		rho_nfw (np.array): The amplitude of the nfw halos in units of
			M_sun/kpc^3
		r_trunc (np.array): The truncation radius for each nfw in units of
			kpc.
		z_source (float): The redshift of the source
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.

	Returns:
		[np.array,...]: A list of 3 numpy arrays: The angular r_scale, the
		deflection angle at the scale radius, and the angular truncation
		radius in units of kpc.
	"""
	kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,cosmo)

	# For two of our parameters we just need to convert to arcseconds
	r_scale_ang = r_scale/kpc_per_arcsecond
	r_trunc_ang = r_trunc/kpc_per_arcsecond

	# For our deflection angle the calculation is a little more involved
	# and requires knowledge of the source redshift. We will take advantage
	# of lenstronomy for this calculation.
	sigma_crit = calculate_sigma_crit(z,z_source,cosmo)
	alpha_rs = rho_nfw * (4*r_scale**2*(1+np.log(0.5)))
	alpha_rs /= kpc_per_arcsecond * sigma_crit
	return [r_scale_ang, alpha_rs, r_trunc_ang]


def convert_from_lenstronomy_tNFW(r_scale_ang,alpha_rs,r_trunc_ang,z,
	z_source,cosmo):
	"""Converts parameters used by lenstronomy to physical tNFW parameters

	Args:
		r_scale_ang (np.array): The scale radius of the nfw in angular
			units
		alpha_rs (np.array): The deflection angle at the scale radius
		r_trunc_ang (np.array): The truncation radius for each nfw in angular
			units.
		z (np.array): The redshift of the nfw
		z_source (float): The redshift of the source
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
			colossus cosmology object.

	Returns:
		[np.array,...]: A list of 3 numpy arrays: The scale radius in kpc,
		the amplitude of the nfw halo in units of M_sun/kpc^3, and the
		truncation radius in units of kpc.
	"""
	kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,cosmo)

	# For the scale radius we just need to convert from arcseconds
	r_scale = r_scale_ang * kpc_per_arcsecond
	r_trunc = r_trunc_ang * kpc_per_arcsecond

	# Deflection angle to rho_nfw requires sigma_crit and source
	# position
	sigma_crit = calculate_sigma_crit(z,z_source,cosmo)
	rho_nfw = alpha_rs / (4*r_scale**2*(1+np.log(0.5)))
	rho_nfw *= kpc_per_arcsecond * sigma_crit
	return [r_scale,rho_nfw,r_trunc]
