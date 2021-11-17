import numpy as np
import unittest
from manada.Sampling.sampler import Sampler
from manada.Sampling import distributions
from scipy.stats import uniform, norm, loguniform, lognorm, multivariate_normal
from scipy.stats import truncnorm
import warnings


class SamplerTests(unittest.TestCase):

	def setUp(self):

		# Set up a distribution for our cross_object parameters
		mean = np.ones(2)
		cov = np.array([[1.0,0.7],[0.7,1.0]])
		min_values = np.zeros(2)
		tmn = distributions.TruncatedMultivariateNormal(mean,cov,min_values,
			None)
		self.config_dict = {
			'subhalo':{
				'class': None,
				'parameters':{
					'sigma_sub':uniform(loc=0,scale=5e-2).rvs,
					'shmf_plaw_index': norm(loc=-1.83,scale=0.1).rvs,
					'm_pivot': 1e8,'m_min': 1e6,'m_max': 1e10,
					'c_0':18,'conc_xi':-0.2,'conc_beta':0.8,
					'conc_m_ref': 1e8,'dex_scatter': 0.1
				}
			},
			'los':{
				'class': None,
				'parameters':{
					'm_min':1e6,'m_max':1e10,'z_min':0.01,
					'dz':0.01,'cone_angle':8.0,'r_min':0.5,'r_max':10.0,
					'c_0':18,'conc_xi':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
					'dex_scatter': 0.1,'delta_los':uniform(loc=0,scale=2.0).rvs
				}
			},
			'main_deflector':{
				'models': None,
				'parameters':{
					'M200': loguniform(a=1e11,b=5e13).rvs,
					'z_lens': 0.5,
					'gamma': lognorm(scale=2.01,s=0.1).rvs,
					'theta_E': lognorm(scale=1.1,s=0.05).rvs,
					'e1,e2': multivariate_normal(np.zeros(2),
						np.array([[1,0.5],[0.5,1]])).rvs,
					'center_x': norm(loc=0.0,scale=0.16).rvs,
					'center_y': norm(loc=0.0,scale=0.16).rvs,
					'gamma1': norm(loc=0.0,scale=0.05).rvs,
					'gamma2': norm(loc=0.0,scale=0.05).rvs}
			},
			'source':{
				'class_instance': None,
				'parameters':{
					'z_source':1.5,'catalog_i':200}
			},
			'cosmology':{
				'parameters':{
					'cosmology_name': 'planck18'
				}
			},
			'psf':{
				'parameters':{
					'psf_type':'GAUSSIAN',
					'fwhm': 0.1
				}
			},
			'detector':{
				'parameters':{
					'pixel_scale':0.08,'ccd_gain':2.5,'read_noise':4.0,
					'magnitude_zero_point':25.9463,
					'exposure_time':5400.0,'sky_brightness':22,
					'num_exposures':1, 'background_noise':None
				}
			},
			'drizzle':{
				'parameters':{
					'supersample_pixel_scale':0.020,'output_pixel_scale':0.030,
					'wcs_distortion':None,
					'offset_pattern':[(0,0),(0.5,0),(0.0,0.5),(-0.5,-0.5)]
				}
			},
			'point_source':{
				'class': None,
				'parameters':{
					'x_point_source':0.01,'y_point_source':0.01,'magnitude':24.8,
					'output_ab_zeropoint':25.127
				}
			},
			'lens_light':{
				'class': None,
				'parameters':{
					'z_source':1.5,
					'amp':truncnorm(-20.0/2.0,np.inf,loc=20.0,scale=2).rvs,
					'R_sersic':truncnorm(-1.0/0.2,np.inf,loc=1.0,scale=0.2).rvs,
					'n_sersic':truncnorm(-1.2/0.2,np.inf,loc=1.2,scale=0.2).rvs,
					'e1':norm(loc=0.0,scale=0.1).rvs,
					'e2':norm(loc=0.0,scale=0.1).rvs,
					'center_x':0.0,'center_y':0.0
				}
			},
			'cross_object':{
				'parameters':{
					'los:delta_los,subhalo:sigma_sub':tmn
				}
			}
		}

		self.s = Sampler(self.config_dict)

	def test_draw_from_dict(self):
		# Test that the draw dict returns the samples we desire.
		draw_dict = {
			'e1,e2': multivariate_normal(np.zeros(2),np.array([[1,0.5],
				[0.5,1]])).rvs,
			'theta_E': uniform().rvs,
			'gamma1':0.2,
			'name':'test'
		}

		param_dict = self.s.draw_from_dict(draw_dict)

		# Check that the correct values are returned for fixed parameters
		# and that for random parameters the values are within reasonable
		# bounds.
		self.assertEqual(param_dict['name'],'test')
		self.assertEqual(param_dict['gamma1'],0.2)
		self.assertGreater(param_dict['e1'],-10)
		self.assertGreater(param_dict['e2'],-10)
		self.assertLess(param_dict['e1'],10)
		self.assertLess(param_dict['e2'],10)
		self.assertGreater(param_dict['theta_E'],0)
		self.assertLess(param_dict['theta_E'],1)

	def test_sample(self):
		# Test that the samples drawn agree with the values put in.]
		warnings.simplefilter('ignore')
		sample = self.s.sample()

		# Sample a few times
		for _ in range(10):
			# First check that all the expected dicts are in the object
			expected_dicts = ['subhalo_parameters','los_parameters',
				'main_deflector_parameters','source_parameters',
				'cosmology_parameters','drizzle_parameters',
				'lens_light_parameters','point_source_parameters']
			for dict_name in expected_dicts:
				self.assertTrue(dict_name in sample)

			# Now for each dict, check some values
			self.assertEqual(sample['cosmology_parameters']['cosmology_name'],
				'planck18')
			self.assertEqual(sample['source_parameters']['catalog_i'],
				200)
			self.assertGreater(sample['main_deflector_parameters']['theta_E'],
				0)
			self.assertGreater(sample['main_deflector_parameters']['gamma'],
				0)
			self.assertGreater(sample['los_parameters']['delta_los'],
				0)
			self.assertGreater(sample['subhalo_parameters']['sigma_sub'],
				0)


class DistributionsTests(unittest.TestCase):

	def testMultivariateLogNormal(self):
		# Test that the values returned follow the expected statistics
		mean = np.ones(3)
		cov = np.array([[1,0.2,0.2],[0.2,1.0,0.2],[0.2,0.2,1.0]])
		mln = distributions.MultivariateLogNormal(mean,cov)

		# Just test that the regular call works
		draw = mln()
		self.assertTupleEqual(draw.shape,(1,3))

		# Now test the statistics for more draws
		n_draws = int(1e6)
		draws = mln(n_draws)
		self.assertTupleEqual(draws.shape,(n_draws,3))
		np.testing.assert_almost_equal(np.cov(np.log(draws).T),cov,
			decimal=2)
		np.testing.assert_almost_equal(np.mean(np.log(draws),axis=0),mean,
			decimal=2)

	def testTruncatedMultivariateNormal(self):
		# Test that the truncated multivariate normal returns reasonable
		# values.
		# Start with the case where it should just behave like a multivariate
		# normal
		mean = np.ones(2)
		cov = np.array([[1.0,0.7],[0.7,1.0]])
		min_values = None
		max_values = None

		tmn = distributions.TruncatedMultivariateNormal(mean,cov,min_values,
			max_values)
		# Check this works
		draw = tmn()
		self.assertTupleEqual(draw.shape,(2,))
		# Test stats for large number of draws
		draws = tmn(int(1e5))
		np.testing.assert_almost_equal(np.cov(draws.T),cov,decimal=2)
		np.testing.assert_almost_equal(np.mean(draws,axis=0),mean,decimal=2)

		# Now put some real limits and make sure they're respected
		min_values = np.array([0,-0.1])
		max_values = np.array([2,2.1])
		tmn = distributions.TruncatedMultivariateNormal(mean,cov,min_values,
			max_values)
		# Check that this works
		draw = tmn()
		self.assertTupleEqual(draw.shape,(2,))
		# Test the limits for large number of draws
		draws = tmn(int(1e5))
		self.assertTrue(np.prod(draws[:,0]>min_values[0]))
		self.assertTrue(np.prod(draws[:,1]>min_values[1]))
		self.assertTrue(np.prod(draws[:,0]<max_values[0]))
		self.assertTrue(np.prod(draws[:,1]<max_values[1]))
		np.testing.assert_almost_equal(np.mean(draws,axis=0),mean,decimal=2)

	def testEllipticitiesTranslation(self):
		# test mapping w/ constants
		dist = distributions.EllipticitiesTranslation(q_dist=0.1,phi_dist=0)
		e1,e2 = dist()
		self.assertAlmostEqual(e1,0.9/1.1)
		self.assertAlmostEqual(e2,0)
		# test mapping w/ distributions
		# cos(2phi) positive, sin(2phi) positive
		dist = distributions.EllipticitiesTranslation(q_dist=uniform(
			loc=0.,scale=0.1).rvs,phi_dist=uniform(loc=0,scale=np.pi/4).rvs)
		e1,e2 = dist()
		print('e1: ', e1)
		self.assertTrue(e1 >= 0)
		self.assertTrue(e2 >= 0)
			
	def testExternalShearTranslation(self):
		# test mapping w/ constants
		dist = distributions.ExternalShearTranslation(gamma_dist=1.5,
			phi_dist=np.pi/12)
		g1,g2 = dist()
		self.assertAlmostEqual(g1,1.5*np.sqrt(3)/2)
		self.assertAlmostEqual(g2,1.5*0.5)
		# test mapping w/ distributions
		g = uniform(loc=0, scale=0.5).rvs
		# sin(2phi) positive, cos(2phi) negative
		p = uniform(loc=np.pi/4,scale=np.pi/4).rvs
		dist = distributions.ExternalShearTranslation(gamma_dist=g,phi_dist=p)
		g1,g2 = dist()
		self.assertTrue(np.abs(g1)< 1 and np.abs(g2) < 1)
		self.assertTrue(g1 < 0 and g2 > 0)

	def testKappaTransformDistribution(self):
		# test mapping w/ constants
		dist = distributions.KappaTransformDistribution(n_dist=.2)
		kappa = dist()
		self.assertAlmostEqual(kappa,1 - 1/0.2)
		# test mapping w/ distributions
		dist = distributions.KappaTransformDistribution(n_dist=
			uniform(loc=0.8,scale=0.1).rvs)
		kappa = dist()
		self.assertTrue(kappa < 0)

	def testDuplicateXY(self):
		# test mapping w/ constants
		dist = distributions.DuplicateXY(x_dist=1,y_dist=2)
		x1,y1,x2,y2 = dist()
		self.assertTrue(x1==1 and x2==1 and y1==2 and y2==2)
		# test mapping w/ distributions
		dist = distributions.DuplicateXY(x_dist=uniform(loc=1,scale=1).rvs,
			y_dist=uniform(loc=-2,scale=1).rvs)
		x1,y1,x2,y2 = dist()
		self.assertTrue(x1>0 and x2>0 and y1<0 and y2<0)

	def testRedshiftsTruncNorm(self):
		# z_lens has min of 0.5
		# z_source has min of 0, centered at 0.6
		# check that z_source > 0.5 is enforced for multiple draws
		dist = distributions.RedshiftsTruncNorm(
			z_lens_min=10,z_lens_mean=1,z_lens_std=0.05,
			z_source_min=1,z_source_mean=0.6,z_source_std=0.6)
		for i in range(0,5):
			z_lens,z_source = dist()
			self.assertTrue(z_lens > 0.5 and z_source > 0.5)
