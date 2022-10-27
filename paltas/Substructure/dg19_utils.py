import warnings
from colossus.lss import peaks, bias
import numpy as np


def mass_concentration(parameter_dict, cosmo, z, m_200, scatter_mult=1.):
	"""Returns the concentration of halos at a certain mass given the
	parameterization of DG_19.

	Args:
		parameter_dict: dictionary from which to pull:
			c_0
			conc_zeta
			conc_beta
			conc_m_ref
			dex_scatter 
		cosmo: colossus cosmology
		z (float): The redshift of the nfw halos
		m_200 (np.array): array of M_200 of the nfw halo units of M_sun
		scatter_mult (float): an additional scaling to the scatter. Likely
			only useful for los rendering to force scatter to 0.

	Returns:
		(np.array): The concentration for each halo.
	"""
	# Get the concentration parameters
	c_0 = parameter_dict['c_0']
	zeta = parameter_dict['conc_zeta']
	beta = parameter_dict['conc_beta']
	m_ref = parameter_dict['conc_m_ref']
	dex_scatter = parameter_dict['dex_scatter']*scatter_mult

	# The peak calculation is done by colossus. 
	# These functions expect M_sun/h units (which you get by multiplying by h
	# https://www.astro.ljmu.ac.uk/~ikb/research/h-units.html)
	h = cosmo.h
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


def halfmode_suppression(masses, param_dict):
	m_hm = param_dict.get('halfmode_mass', 0)

	if m_hm == 0 or not len(masses):
		# No suppression / pure cold dark matter
		return masses

	# Get shape parameters controlling half-mode suppression
	# These are fit to simulation, see discussion in Nadler et al. ApJ 917:7 
	alpha = param_dict.get('halfmode_alpha', 1)
	beta = param_dict.get('halfmode_beta', 1)
	gamma = param_dict.get('halfmode_gamma', -1.3)

	p_keep = (1 + (alpha * m_hm / masses)**beta) ** gamma
	if not (0 <= p_keep.min() <= p_keep.max() <= 1):
		warnings.warn("Strange values in p_keep... will be clipped")
	p_keep = p_keep.clip(0, 1)

	return masses[np.random.rand(len(masses)) < p_keep]
