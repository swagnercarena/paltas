import unittest
from manada.Substructure import nfw_functions
from manada.Utils import power_law
import numpy as np
from scipy.integrate import quad
from colossus.cosmology import cosmology


class NFWFunctionsTests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)

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

	def test_cored_nfw_integral(self):
		# Test that the cored nfw integral returns values that agree with the
		# numerical integral.
		r_tidal = 0.5
		r_scale = 2
		rho_nfw = 1
		r_upper = np.linspace(0,4,100)
		analytic_values = nfw_functions.cored_nfw_integral(r_tidal,rho_nfw,
			r_scale,r_upper)

		def cored_nfw_func(r,r_tidal,rho_nfw,r_scale):
			if r<r_tidal:
				x_tidal = r_tidal/r_scale
				return rho_nfw/(x_tidal*(1+x_tidal)**2)
			else:
				x = r/r_scale
				return rho_nfw/(x*(1+x)**2)

		for i in range(len(r_upper)):
			self.assertAlmostEqual(analytic_values[i],quad(cored_nfw_func,0,
				r_upper[i],args=(r_tidal,rho_nfw,r_scale))[0])

	def test_cored_nfw_draws(self):
		# Test that the draws follow the desired distribution
		r_tidal = 0.5
		r_scale = 2
		rho_nfw = 1.5
		r_max = 4
		n_subs = int(1e5)
		r_draws = nfw_functions.cored_nfw_draws(r_tidal,rho_nfw,r_scale,
			r_max,n_subs)

		n_test_points = 100
		r_test = np.linspace(0,r_max,n_test_points)
		analytic_integral = nfw_functions.cored_nfw_integral(r_tidal,rho_nfw,
			r_scale,r_test)

		for i in range(n_test_points):
			self.assertAlmostEqual(np.mean(r_draws<r_test[i]),
				analytic_integral[i]/np.max(analytic_integral),places=2)

	def test_r_200_from_m(self):
		# TODO!
		return
