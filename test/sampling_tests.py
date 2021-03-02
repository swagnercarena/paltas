import numpy as np
import unittest
import os
from manada.Sampling.sampler import Sampler
from scipy.stats import uniform, norm, loguniform, lognorm, multivariate_normal


class SamplerTests(unittest.TestCase):

	def setUp(self):
		self.config_dict = {
			'subhalo':{
				'class': None,
				'params':{
					'sigma_sub':uniform(loc=0,scale=5e-2).rvs,
					'shmf_plaw_index': norm(loc=-1.83,scale=0.1).rvs,
					'm_pivot': 1e8,'m_min': 1e6,'m_max': 1e10,
					'c_0':18,'conc_xi':-0.2,'conc_beta':0.8,
					'conc_m_ref': 1e8,'dex_scatter': 0.1
				}
			},
			'los':{
				'class': None,
				'params':{
					'm_min':1e6,'m_max':1e10,'z_min':0.01,
					'dz':0.01,'cone_angle':8.0,'r_min':0.5,'r_max':10.0,
					'c_0':18,'conc_xi':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
					'dex_scatter': 0.1,'delta_los':uniform(loc=0,scale=2.0).rvs
				}
			},
			'main_deflector':{
				'models': None,
				'params':{
					'M200': loguniform(a=1e11,b=5e13).rvs,
					'z_lens': 0.5,
					'gamma': lognorm(scale=2.01,s=0.1).rvs,
					'theta_E': lognorm(scale=1.1,s=0.05).rvs,
					'e1': norm(loc=0.0,scale=0.05).rvs,
					'e2': norm(loc=0.0,scale=0.05).rvs,
					'center_x': norm(loc=0.0,scale=0.16).rvs,
					'center_y': norm(loc=0.0,scale=0.16).rvs,
					'gamma1': norm(loc=0.0,scale=0.05).rvs,
					'gamma2': norm(loc=0.0,scale=0.05).rvs}
			},
			'source':{
				'class': None,
				'parameters':{
					'z_source':1.5,'catalog_i':200}
			},
			'cosmology':{
				'parameters':{
					'cosmology_name': 'planck18'
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
		self.assertLess(param_dict['e1'],110)
		self.assertGreater(param_dict['theta_E'],0)
		self.assertLess(param_dict['theta_E'],1)
