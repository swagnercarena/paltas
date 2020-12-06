import unittest
from manada.Substructure import nfw_functions
from manada.Utils import power_law
import numpy as np
from colossus.cosmology import cosmology


class NFWFunctionsTests(unittest.TestCase):

	def test_host_scaling_function_DG_19(self):
		# Just test that the function agrees with some pre-computed values
		host_m200_list = [1e10,1e11,2.5e11]
		z_lens_list = [0.3,0.4,0.6]
		pre_computed_list = [0.0015676639733375364,0.014528215260579059,
			0.04576724520325134]

		for i in range(len(host_m200_list)):
			self.assertAlmostEqual(nfw_functions.host_scaling_function_DG_19(
				host_m200_list[i],z_lens_list[i]),pre_computed_list[i])

	def test_draw_nfw_masses_DG_19(self):
		# Test that the mass function draws agree with the input parameters

		# Quickest test is to make sure an error is raised of all the needed
		# parameters are not passed in
		subhalo_parameters = {'sigma_sub':4e-2, 'shmf_plaw_index': -1.83,
			'm_pivot': 1e8, 'm_min': 1e6}
		main_deflector_parameters = {'M200': 1e13, 'z_lens':0.45,
			'theta_E':0.38}
		cosmo = cosmology.setCosmology('planck18')

		with self.assertRaises(ValueError):
			masses = nfw_functions.draw_nfw_masses_DG_19(subhalo_parameters,
				main_deflector_parameters,cosmo)

		# Add the parameter we need
		subhalo_parameters['m_max'] = 1e9

		# Calculate the norm by hand and make sure the statistics agree
		kpc_per_arcsecond = (cosmo.angularDiameterDistance(
			main_deflector_parameters['z_lens']) * cosmo.H0/100 * 1000
			*  np.pi/180/3600)
		r_E = kpc_per_arcsecond*main_deflector_parameters['theta_E']
		dA = np.pi * (3*r_E)**2
		f_host = nfw_functions.host_scaling_function_DG_19(
			main_deflector_parameters['M200'],
			main_deflector_parameters['z_lens'])
		e_counts =  power_law.power_law_integrate(subhalo_parameters['m_min'],
			subhalo_parameters['m_max'],subhalo_parameters['shmf_plaw_index'])
		norm_without_sigma_sub = dA*f_host*(
			subhalo_parameters['m_pivot']**(-
				subhalo_parameters['shmf_plaw_index']-1))*e_counts
		desired_count = 1e2
		subhalo_parameters['sigma_sub'] = (desired_count /
			norm_without_sigma_sub)

		total_subs = 0
		n_loops = 5000
		for _ in range(n_loops):
			masses = nfw_functions.draw_nfw_masses_DG_19(subhalo_parameters,
				main_deflector_parameters,cosmo)
			total_subs += len(masses)
		self.assertEqual(np.round(total_subs/n_loops),desired_count)

		# Now just give some parameters for an HE0435-1223 galaxy and make sure
		# what we return is reasonable as a sanity check on units.
		subhalo_parameters['sigma_sub'] = 4e-2
		total_subs = 0
		for _ in range(n_loops):
			masses = nfw_functions.draw_nfw_masses_DG_19(subhalo_parameters,
				main_deflector_parameters,cosmo)
			total_subs += len(masses)
		self.assertGreater(total_subs//n_loops,50)
		self.assertLess(total_subs//n_loops,200)
