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
