# -*- coding: utf-8 -*-
"""
Draw subhalo masses and concentrations for NFW subhalos according to
https://arxiv.org/pdf/1909.02573.pdf

This module contains the functions needed to turn the parameters of NFW
subhalo distributions into masses, concentrations, and positions for those
NFW subhalos.
"""
from .subhalos_base import SubhalosBase
from . import nfw_functions
import numba
import numpy as np
from ..Utils import power_law, cosmology_utils
from colossus.halo.concentration import peaks


class SubhalosDG19(SubhalosBase):
	"""Class for rendering the subhalos of a main halos according to DG19.

	Args:
		subhalo_parameters (dict): A dictionary containing the type of
			subhalo distribution and the value for each of its parameters.
		main_deflector_parameters (dict): A dictionary containing the type of
			main deflector and the value for each of its parameters.
		source_parameters (dict): A dictionary containing the type of the
			source and the value for each of its parameters.
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).

	Notes:

	Required Parameters

	- sigma_sub - SHMF normalization in units of kpc^(-2)
	- shmf_plaw_index - SHMF mass function power-law slope
	- m_pivot - SHMF power-law pivot mass in unit of M_sun
	- m_min - SHMF minimum rendered mass in units of M_sun
	- m_max - SHMF maximum rendered mass in units of M_sun
	- c_0 - concentration normalization
	- conc_zeta - concentration redshift power law slope
	- conc_beta - concentration peak height power law slope
	- conc_m_ref - concentration peak height pivot mass
	- dex_scatter - scatter in concentration in units of dex
	- k1 - slope of SHMF host mass dependence
	- k2 - slope of SHMF host redshift dependence
	"""
	# Define the parameters we expect to find for the DG_19 model
	required_parameters = ('sigma_sub','shmf_plaw_index','m_pivot','m_min',
		'm_max','c_0','conc_zeta','conc_beta','conc_m_ref','dex_scatter',
		'k1','k2')

	def __init__(self,subhalo_parameters,main_deflector_parameters,
		source_parameters,cosmology_parameters):

		# Initialize the super class
		super().__init__(subhalo_parameters,main_deflector_parameters,
			source_parameters,cosmology_parameters)

	@staticmethod
	@numba.njit()
	def host_scaling_function(host_m200, z_lens, k1=0.88, k2=1.7):
		"""Returns scaling for the subhalo mass function based on the mass of
		the host halo.

		Derived from galacticus in https://arxiv.org/pdf/1909.02573.pdf.

		Args:
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

	def draw_nfw_masses(self):
		"""Draws from the https://arxiv.org/pdf/1909.02573.pdf subhalo mass
		function and returns an array of the masses.

		Returns:
			(np.array): The masses of the drawn halos in units of M_sun
		"""

		# Pull the parameters we need from the input dictionaries
		# Units of m_sun times inverse kpc^2
		sigma_sub = max(0, self.subhalo_parameters['sigma_sub'])
		shmf_plaw_index = self.subhalo_parameters['shmf_plaw_index']
		# Units of m_sun
		m_pivot = self.subhalo_parameters['m_pivot']
		# Units of m_sun
		host_m200 = self.main_deflector_parameters['M200']
		# Units of m_sun
		m_min = self.subhalo_parameters['m_min']
		# Units of m_sun
		m_max = self.subhalo_parameters['m_max']
		z_lens = self.main_deflector_parameters['z_lens']
		k1 = self.subhalo_parameters['k1']
		k2 = self.subhalo_parameters['k2']

		# Calculate the overall norm of the power law. This includes host
		# scaling, sigma_sub, and the area of interest.
		f_host = self.host_scaling_function(host_m200,z_lens,k1=k1,k2=k2)

		# In DG_19 subhalos are rendered up until 3*theta_E.
		# Colossus return in MPC per h per radian so must be converted to kpc
		# per arc second
		kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z_lens,
			self.cosmo)
		r_E = (kpc_per_arcsecond*self.main_deflector_parameters['theta_E'])
		dA = np.pi * (3*r_E)**2

		# We can also fold in the pivot mass into the norm for simplicity (then
		# all we have to do is sample from a power law).
		norm = f_host*dA*sigma_sub*m_pivot**(-shmf_plaw_index-1)

		# Draw from our power law and return the masses.
		masses = power_law.power_law_draw(m_min,m_max,shmf_plaw_index,norm)
		return masses

	def mass_concentration(self,z,m_200):
		"""Returns the concentration of halos at a certain mass given the
		parameterization of DG_19.

		Args:
			z (np.array): The redshift of the nfw halo
			m_200 (np.array): array of M_200 of the nfw halo units of M_sun

		Returns:
			(np.array): The concentration for each halo.
		"""
		# Get the concentration parameters
		c_0 = self.subhalo_parameters['c_0']
		zeta = self.subhalo_parameters['conc_zeta']
		beta = self.subhalo_parameters['conc_beta']
		m_ref = self.subhalo_parameters['conc_m_ref']
		dex_scatter = self.subhalo_parameters['dex_scatter']

		# The peak calculation is done by colossus. The cosmology must have
		# already been set. Note these functions expect M_sun/h units (which
		# you get by multiplying by h
		# https://www.astro.ljmu.ac.uk/~ikb/research/h-units.html)
		h = self.cosmo.h
		peak_heights = peaks.peakHeight(m_200*h,z)
		peak_height_ref = peaks.peakHeight(m_ref*h,0)

		# Now get the concentrations and add scatter
		concentrations = c_0*(1+z)**(zeta)*(peak_heights/peak_height_ref)**(
			-beta)
		if isinstance(concentrations,np.ndarray):
			conc_scatter = np.random.randn(len(concentrations))*dex_scatter
		elif isinstance(concentrations,float):
			conc_scatter = np.random.randn()*dex_scatter
		concentrations = 10**(np.log10(concentrations)+conc_scatter)

		return concentrations

	@staticmethod
	def rejection_sampling(r_samps,r_200,r_3E):
		"""Given the radial sampling of the positions and DG_19 constraints,
		conducts rejection sampling and return the cartesian positions.

		Args:
			r_samps (np.array): Samples of the radial coordinates for
				the subhalos in units of kpc.
			r_200 (float): The r_200 of the host halo which will be used
				as the maximum z magnitude in units of kpc.
			r_3E (float): 3 times the einstein radius, which will be used
				to bound the x and y coordinates in units of kpc.

		Returns:
			([np.array,...]): A list of two numpy arrays: the boolean
			array of accepted samples and a n_subsx3 array of x,y,z
			coordinates. All in units of kpc.
		"""
		# Sample theta and phi values for all of the radii samples
		theta = np.random.rand(len(r_samps)) * 2 * np.pi
		phi = np.arccos(1-2*np.random.rand(len(r_samps)))

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

		return (keep,cart_pos)

	def sample_cored_nfw(self,n_subs):
		"""Given the a tidal radius that defines a core region and the
		parameters of the main deflector, samples positions for NFW subhalos
		bounded as described in https://arxiv.org/pdf/1909.02573.pdf

		Args:
			n_subs (int): The number of subhalo positions to sample

		Returns:
			(np.array): A n_subs x 3 array giving the x,y,z position of the
			subhalos in units of kpc.

		Notes:
			The code works through rejection sampling, which can be inneficient
			for certain configurations. If this is a major issue, it may be
			worth introducing more analytical components.
		"""

		# Create an array that will store our coordinates
		cart_pos = np.zeros((n_subs,3))

		# Calculate the needed host properties
		host_m200 = self.main_deflector_parameters['M200']
		z_lens = self.main_deflector_parameters['z_lens']
		host_c = self.mass_concentration(z_lens,host_m200)
		host_r_200 = nfw_functions.r_200_from_m(host_m200,z_lens,self.cosmo)
		host_r_scale = host_r_200/host_c
		# DG_19 definition of the tidal radius
		r_tidal = host_r_200/2
		host_rho_nfw = nfw_functions.rho_nfw_from_m_c(host_m200,host_c,
			self.cosmo,r_scale=host_r_scale)

		# Tranform the einstein radius to physical units (TODO this should
		# be a function). Multiply by 3 since that's what's relevant for
		# DG_19 parameterization.
		kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z_lens,self.cosmo)
		r_3E = (kpc_per_arcsecond*self.main_deflector_parameters['theta_E'])*3

		# The largest radius we should bother sampling is set by the diagonal of
		# our cylinder.
		r_max = np.sqrt(r_3E**2+host_r_200**2)

		n_accepted_draws = 0
		r_subs = nfw_functions.cored_nfw_draws(r_tidal,host_rho_nfw,
			host_r_scale,r_max,n_subs)
		keep_ind, cart_draws = self.rejection_sampling(r_subs,host_r_200,r_3E)

		# Save the cartesian coordinates we want to keep
		cart_pos[n_accepted_draws:n_accepted_draws+np.sum(keep_ind)] = (
			cart_draws[keep_ind])
		n_accepted_draws += np.sum(keep_ind)

		# Get the fraction of rejection to see how much we should sample
		rejection_frac = max(1-np.mean(keep_ind),1e-1)

		# Keep drawing until we have enough r_subs.
		while n_accepted_draws<n_subs:
			r_subs = nfw_functions.cored_nfw_draws(r_tidal,host_rho_nfw,
				host_r_scale,r_max,int(np.round(n_subs*rejection_frac)))
			keep_ind, cart_draws = self.rejection_sampling(r_subs,host_r_200,
				r_3E)
			use_keep = np.minimum(n_subs-n_accepted_draws,np.sum(keep_ind))
			# Save the cartesian coordinates we want to keep
			cart_pos[n_accepted_draws:n_accepted_draws+use_keep] = (
				cart_draws[keep_ind][:use_keep])
			n_accepted_draws += use_keep

		return cart_pos

	@staticmethod
	def get_truncation_radius(m_200,r,m_pivot=1e7,r_pivot=50):
		"""Returns the truncation radius for a subhalo given the mass and
		radial position in the host NFW

		Args:
			m_200 (np.array): The mass of the subhalos in units of M_sun
			r (np.array): The radial position of the subhalos in units of kpc
			m_pivot (float): The pivot mass for the scaling in units of M_sun
			r_pivot (float): The pivot radius for the relation in unit of kpc

		Returns:
			(np.array): The truncation radii for the subhalos in units of kpc
		"""

		return 1.4*(m_200/m_pivot)**(1/3)*(r/r_pivot)**(2/3)

	def convert_to_lenstronomy(self,subhalo_masses,subhalo_cart_pos):
		"""Converts the subhalo masses and position to truncated NFW profiles
		for lenstronomy

		Args:
			subhalo_masses (np.array): The masses of each of the subhalos that
				were drawn
			subhalo_cart_pos (np.array): A n_subs x 3D array of the positions
				of the subhalos that were drawn
		Returns:
			([string,...],[dict,...]): A tuple containing the list of models
			and the list of kwargs for the truncated NFWs.
		"""
		# First, for each subhalo mass we'll also have to draw a concentration.
		# This requires a redshift. DG_19 used the predicted redshift of infall
		# from galacticus. For now, we'll use the redshift of the lens itself.
		# TODO: Use a different redshift
		z_lens = self.main_deflector_parameters['z_lens']
		z_source = self.source_parameters['z_source']
		subhalo_z = (np.ones(subhalo_masses.shape) *
			self.main_deflector_parameters['z_lens'])
		concentration = self.mass_concentration(subhalo_z,subhalo_masses)

		# We'll also need the radial position in the halo
		r_in_host = np.sqrt(np.sum(subhalo_cart_pos**2,axis=-1))

		# Now we can convert these masses and concentrations into NFW parameters
		# for lenstronomy.
		sub_r_200 = nfw_functions.r_200_from_m(subhalo_masses,subhalo_z,
			self.cosmo)
		sub_r_scale = sub_r_200/concentration
		sub_rho_nfw = nfw_functions.rho_nfw_from_m_c(subhalo_masses,
			concentration,self.cosmo,
			r_scale=sub_r_scale)
		sub_r_trunc = self.get_truncation_radius(subhalo_masses,r_in_host)

		# Convert to lenstronomy units
		sub_r_scale_ang, alpha_Rs, sub_r_trunc_ang = (
			nfw_functions.convert_to_lenstronomy_tNFW(
				sub_r_scale,subhalo_z,sub_rho_nfw,sub_r_trunc,z_source,
				self.cosmo))
		kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z_lens,
			self.cosmo)
		cart_pos_ang = subhalo_cart_pos / np.expand_dims(kpc_per_arcsecond,
			axis=-1)

		# Populate the parameters for each lens
		model_list = []
		kwargs_list = []
		for i in range(len(subhalo_masses)):
			model_list.append('TNFW')
			kwargs_list.append({'alpha_Rs':alpha_Rs[i],'Rs':sub_r_scale_ang[i],
				'center_x':(cart_pos_ang[i,0]+
					self.main_deflector_parameters['center_x']),
				'center_y':(cart_pos_ang[i,1]+
					self.main_deflector_parameters['center_y']),
				'r_trunc':sub_r_trunc_ang[i]})

		return (model_list,kwargs_list)

	def draw_subhalos(self):
		"""Draws masses, concentrations,and positions for the subhalos of a
		main lens halo.

		Returns:
			(tuple): A tuple of three lists: the first is the profile type for
			each subhalo returned, the second is the lenstronomy kwargs for
			that subhalo, and the third is the redshift for each subhalo.
		Notes:
			The redshift for each subhalo is the same as the host, so the
			returned redshift list is not necessary unless the output is
			being combined with los substructure.
		"""
		# Initialize the lists that will contain our mass profile types and
		# assosciated kwargs. If no subhalos are drawn, these will remain empty
		subhalo_model_list = []
		subhalo_kwargs_list = []

		# Distribute subhalos according to https://arxiv.org/pdf/1909.02573.pdf
		# DG_19 assumes NFWs distributed throughout the main deflector.
		# For these NFWs we need positions, masses, and concentrations that
		# we will then translate to Lenstronomy parameters.
		subhalo_masses = self.draw_nfw_masses()

		# It is possible for there to be no subhalos. In that regime
		# just return empty lists
		if subhalo_masses.size == 0:
			return (subhalo_model_list, subhalo_kwargs_list, [])

		subhalo_cart_pos = self.sample_cored_nfw(len(subhalo_masses))
		model_list, kwargs_list = self.convert_to_lenstronomy(
			subhalo_masses,subhalo_cart_pos)
		subhalo_model_list += model_list
		subhalo_kwargs_list += kwargs_list
		subhalo_z_list = [self.main_deflector_parameters['z_lens']]*len(
			subhalo_masses)

		return (subhalo_model_list, subhalo_kwargs_list, subhalo_z_list)
