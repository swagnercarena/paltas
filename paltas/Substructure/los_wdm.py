# -*- coding: utf-8 -*-
"""
Draw los masses and concentrations for NFW subhalos in a WDM universe
following a combination of https://arxiv.org/pdf/1909.02573.pdf,
https://arxiv.org/pdf/2003.01125.pdf, and https://arxiv.org/pdf/1507.01998.pdf.

This module contains the functions needed to turn the parameters of warm dark
matter los distributions into masses, concentrations, and positions for those
subhalos.
"""

from . los_dg19 import LOSDG19


class LOSWDM(LOSDG19):
	"""Class for rendering the line of sight structure with a warm dark matter
	half-mode mass.

	Args:
		los_parameters (dict): A dictionary containing the type of
			los distribution and the value for each of its parameters.
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

	- delta_los - mass function normalization
	- m_min - minimum rendered mass in units of M_sun
	- m_max - maximum rendered mass in units of M_sun
	- z_min - minimum redshift at which to render los halos
	- dz - redshift bin width
	- cone_angle - cone opening angle for rendering los halos
	- c_0 - concentration normalization
	- conc_zeta - concentration redshift power law slope
	- conc_beta - concentration peak height power law slope
	- conc_m_ref - concentration peak height pivot mass
	- dex_scatter - scatter in concentration in units of dex
	- alpha_dz_factor- deflection angle correction redshift bin width
	TODO
	"""
	# Define the parameters we expect to find for the model
	required_parameters = LOSDG19.required_parameters + ('m_hm',)

	def draw_nfw_masses(self,z):
		"""Draws from the Sheth Tormen mass function with an additional
		correction for two point correlation and then applied the
		suppresion from warm dark matter.

		Args:
			z (float): The redshift at which to draw the masses

		Returns:
			(np.array): An array with the drawn masses in units of M_sun.

		"""
		# Pull the parameters we need from the input dictionaries
		# Units of M_sun
		lens_m200 = self.main_deflector_parameters['M200']
		z_lens = self.main_deflector_parameters['z_lens']
		z_source = self.source_parameters['z_source']
		dz = self.los_parameters['dz']
		# Units of arcsecond
		cone_angle = self.los_parameters['cone_angle']
		# Units of Mpc
		r_max = self.los_parameters['r_max']
		# Units of Mpc
		r_min = self.los_parameters['r_min']
		# Units of M_sun
		m_min = self.los_parameters['m_min']
		# Units of M_sun
		m_max = self.los_parameters['m_max']
		delta_los = max(0, self.los_parameters['delta_los'])
		# Get the parameters of the power law fit to the Sheth Tormen mass
		# function
		pl_slope, pl_norm = self.power_law_dn_dm(z+dz/2,m_min,m_max)

		# Scale the norm by the total volume and the two point correlation.
		dV = self.volume_element(z,z_lens,z_source,dz,cone_angle)
		halo_boost = self.two_halo_boost(z,z_lens,dz,lens_m200,r_max,r_min)
		pl_norm *= dV * halo_boost * delta_los

		# Draw from our power law and return the masses.
		masses = power_law.power_law_draw(m_min,m_max,pl_slope,pl_norm)
		return masses

	def mass_concentration(self,z,m_200,scatter_mult=1.0):
		"""Returns the concentration of halos at a certain mass given the
		parameterization of warm dark matter model.

		Args:
			z (float): The redshift of the nfw halos
			m_200 (np.array): array of M_200 of the nfw halo units of M_sun
			scatter_mult (float): an additional scaling to the scatter. Likely
				only useful for los rendering to force scatter to 0.

		Returns:
			(np.array): The concentration for each halo.
		"""
		# Get the concentration parameters
		c_0 = self.los_parameters['c_0']
		zeta = self.los_parameters['conc_zeta']
		beta = self.los_parameters['conc_beta']
		m_ref = self.los_parameters['conc_m_ref']
		dex_scatter = self.los_parameters['dex_scatter']*scatter_mult

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
