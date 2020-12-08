import unittest
from manada.Utils import power_law, cosmology_utils
from scipy.integrate import quad
import numpy as np
from colossus.cosmology import cosmology
from astropy import units as u


class PowerLawTest(unittest.TestCase):

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


class CosmologyTest(unittest.TestCase):

	def test_kpc_per_arcsecond(self):
		# Just check that the calculation agree with what we would expect using
		# astropy.
		cosmo = cosmology.setCosmology('planck18')
		h = cosmo.H0/100
		for z_test in np.linspace(0.2,0.8,20):
			dd = cosmo.comovingDistance(z_max=z_test)/(1+z_test)
			dd *= h * u.Mpc.to(u.kpc)/u.radian.to(u.arcsecond)
			# print(dd)
			self.assertAlmostEqual(cosmology_utils.kpc_per_arcsecond(z_test,
				cosmo),dd,places=4)
