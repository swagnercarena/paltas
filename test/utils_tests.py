import unittest
from manada.Utils import power_law, cosmology_utils
from scipy.integrate import quad
import numpy as np
from colossus.cosmology import cosmology
from astropy import units as u


class PowerLawTests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)

	def test_power_law_integrate(self):
		# Check that the analytic integral of the power law agrees with the
		# numerical equivalent.
		p_mins = np.logspace(3,6,4)
		p_maxs = np.logspace(7,9,4)
		slopes = np.linspace(-1.9,-1.7,5)

		def p_func(x,slope):
			return x**slope

		for p_min, p_max, slope in zip(p_mins,p_maxs,slopes):
			self.assertAlmostEqual(power_law.power_law_integrate(p_min,p_max,
				slope),quad(p_func,p_min,p_max,args=(slope))[0])

	def test_power_law_draw(self):
		# Test that the draws agree with the desired power law with desired
		# norm.
		p_min = 1e6
		p_max = 1e9
		slope = -1.9
		desired_count = 1e2
		norm = desired_count / power_law.power_law_integrate(p_min,p_max,
			slope)

		total_subs = 0
		n_loops = 5000
		for _ in range(n_loops):
			masses = power_law.power_law_draw(p_min,p_max,slope,norm)
			total_subs += len(masses)
		self.assertEqual(np.round(total_subs/n_loops),desired_count)

		desired_count = 1e6
		norm = desired_count / power_law.power_law_integrate(p_min,p_max,
			slope)
		masses = power_law.power_law_draw(p_min,p_max,slope,norm)

		# Check that the integration follows as we would expect
		test_points = np.logspace(6,9,20)
		for test_point in test_points:
			self.assertAlmostEqual(np.mean(masses<test_point),
				power_law.power_law_integrate(p_min,test_point,slope)/
				power_law.power_law_integrate(p_min,p_max,slope),
				places=2)

		# Since this test is important, repeat it one more time with slightly
		# different parameters.
		p_min = 1e5
		p_max = 1e10
		slope = -1.72
		norm = desired_count / power_law.power_law_integrate(p_min,p_max,
			slope)
		masses = power_law.power_law_draw(p_min,p_max,slope,norm)

		test_points = np.logspace(6,9,20)
		for test_point in test_points:
			self.assertAlmostEqual(np.mean(masses<test_point),
				power_law.power_law_integrate(p_min,test_point,slope)/
				power_law.power_law_integrate(p_min,p_max,slope),
				places=2)


class CosmologyTests(unittest.TestCase):

	def test_get_cosmology(self):
		# Check that for the four input types, get cosmology works as
		# intended.
		# Start with string
		cosmology_parameters = 'planck18'
		string_cosmo = cosmology_utils.get_cosmology(cosmology_parameters)
		sh = string_cosmo.h
		so = string_cosmo.Om0

		# Now dict containing the string in cosmology_name
		cosmology_parameters = {'cosmology_name':'planck18'}
		dict_string_cosmo = cosmology_utils.get_cosmology(cosmology_parameters)
		dsh = dict_string_cosmo.h
		dso = dict_string_cosmo.Om0

		# Directly pass the cosmology
		cosmo = cosmology.setCosmology('planck18')
		cosmo_cosmo = cosmology_utils.get_cosmology(cosmo)
		ch = cosmo_cosmo.h
		co = cosmo_cosmo.Om0

		# Pass the parameters in the form of a dict
		cosmology_parameters = {}
		cosmology_parameters['H0'] = 67.66
		cosmology_parameters['Om0'] = 0.3111
		direct_cosmo = cosmology_utils.get_cosmology(cosmo)
		dh = direct_cosmo.h
		do = direct_cosmo.Om0

		# Check that they're all equal.
		self.assertEqual(sh,dsh)
		self.assertEqual(sh,ch)
		self.assertEqual(sh,dh)

		self.assertEqual(so,dso)
		self.assertEqual(so,co)
		self.assertEqual(so,do)

	def test_kpc_per_arcsecond(self):
		# Just check that the calculation agree with what we would expect using
		# astropy.
		cosmo = cosmology.setCosmology('planck18')
		h = cosmo.H0/100
		for z_test in np.linspace(0.2,0.8,20):
			dd = cosmo.comovingDistance(z_max=z_test)/(1+z_test)
			dd *= 1/h * u.Mpc.to(u.kpc)/u.radian.to(u.arcsecond)
			self.assertAlmostEqual(cosmology_utils.kpc_per_arcsecond(z_test,
				cosmo),dd,places=4)

		# Repeat the test in array form
		z_test = np.linspace(0.2,0.8,20)
		dd = cosmo.comovingDistance(z_max=z_test)/(1+z_test)
		dd *= 1/h * u.Mpc.to(u.kpc)/u.radian.to(u.arcsecond)
		np.testing.assert_almost_equal(cosmology_utils.kpc_per_arcsecond(
			z_test,cosmo),dd,decimal=4)
