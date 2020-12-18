import unittest
from manada.Substructure import nfw_functions
from manada.Substructure.subhalos_base import SubhalosBase
from manada.Substructure.subhalos_dg19 import SubhalosDG19
from manada.Utils import power_law
import numpy as np
from scipy.integrate import quad
from colossus.cosmology import cosmology
from colossus.halo.concentration import peaks
from colossus.halo import profile_nfw
from lenstronomy.Cosmo.lens_cosmo import LensCosmo


class NFWFunctionsTests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)

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
		# Compare the calculation from our function to the colossus output
		cosmo = cosmology.setCosmology('planck18')
		m_200 = np.logspace(7,10,20)
		c = 2.9

		# Colossus calculation
		h = cosmo.h
		z_lens =0.2
		rhos, rs = profile_nfw.NFWProfile.fundamentalParameters(M=m_200*h,
			c=c,z=z_lens,mdef='200c')

		# manada calculation
		r_200 = nfw_functions.r_200_from_m(m_200,z_lens,cosmo)

		np.testing.assert_almost_equal(r_200/c,rs/h)

	def test_rho_nfw_from_m_c(self):
		# Compare the calculation from our function to the colossus output
		cosmo = cosmology.setCosmology('planck18')
		m_200 = np.logspace(7,10,20)
		c = 2.9

		h = cosmo.h
		z = 0.2
		rhos, rs = profile_nfw.NFWProfile.fundamentalParameters(M=m_200*h,
			c=c,z=z,mdef='200c')

		# manada calculation
		rho_nfw = nfw_functions.rho_nfw_from_m_c(m_200,c,cosmo,z=z)

		np.testing.assert_almost_equal(rho_nfw,rhos*h**2)

	def test_calculate_sigma_crit(self):
		# Check that the sigma_crit calculation agrees with
		# lenstronomy
		cosmo = cosmology.setCosmology('planck18')
		cosmo_astropy = cosmo.toAstropy()
		z_lens = 0.2
		z_source = 1.3
		lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source,
			cosmo=cosmo_astropy)
		sigma_crit = nfw_functions.calculate_sigma_crit(z_lens,z_source,cosmo)
		mpc_2_kpc = 1e3
		self.assertAlmostEqual(sigma_crit/(lens_cosmo.sigma_crit/mpc_2_kpc**2),
			1,places=4)

	def test_convert_to_lenstronomy_NFW(self):
		# Compare the values we return to those returned by lenstronomy
		cosmo = cosmology.setCosmology('planck18')
		cosmo_astropy = cosmo.toAstropy()
		# Our calculations are always at z=0.
		z_lens = 0.2
		z_source = 1.5
		lens_cosmo = LensCosmo(z_lens=z_lens, z_source=z_source,
			cosmo=cosmo_astropy)
		m_200 = 1e9
		c = 4

		# Do the physical calculations in our code and in lenstronomy
		r_scale = nfw_functions.r_200_from_m(m_200,z_lens,cosmo)/c
		r_trunc = np.ones(r_scale.shape)
		rho_nfw = nfw_functions.rho_nfw_from_m_c(m_200,c,cosmo,r_scale=r_scale)
		rho0, Rs, r200 = lens_cosmo.nfwParam_physical(M=m_200, c=c)

		# Repeat for the angular calculations
		rs_angle_ls, alpha_rs_ls = lens_cosmo.nfw_physical2angle(M=m_200,c=c)
		r_scale_angle, alpha_rs, r_trunc_ang = (
			nfw_functions.convert_to_lenstronomy_NFW(r_scale,z_lens,rho_nfw,
			r_trunc,z_source,cosmo))

		mpc_2_kpc = 1e3
		self.assertAlmostEqual(r_scale,Rs*mpc_2_kpc)
		self.assertAlmostEqual(rho_nfw,rho0/(mpc_2_kpc**3),places=2)
		self.assertAlmostEqual(r_scale_angle,rs_angle_ls,places=2)
		self.assertAlmostEqual(alpha_rs,alpha_rs_ls)
		self.assertAlmostEqual(r_trunc_ang*r_scale/r_scale_angle,r_trunc)


class SubhalosBaseTests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)

		# Make the subhalo base class we'll use for testing.
		self.subhalo_parameters = {'sigma_sub':4e-2, 'shmf_plaw_index': -1.83,
			'm_pivot': 1e8, 'm_min': 1e6, 'm_max': 1e9, 'c_0':18,
			'conc_xi':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.0, 'distribution':'DG_19'}
		self.main_deflector_parameters = {'M200': 1e13, 'z_lens': 0.5,
			'theta_E':1, 'center_x':0.0, 'center_y': 0.0}
		self.source_parameters = {'z_source':1.5}
		self.cosmology_parameters = {'cosmology_name': 'planck18'}
		self.sb = SubhalosBase(self.subhalo_parameters,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)

	def test_init(self):
		# Check that the parameters are saved as desired.
		self.assertTrue(all(elem in self.sb.subhalo_parameters.keys()
			for elem in self.subhalo_parameters.keys()))
		self.assertTrue(all(elem in self.sb.main_deflector_parameters.keys()
			for elem in self.main_deflector_parameters.keys()))
		self.assertTrue(all(elem in self.sb.source_parameters.keys()
			for elem in self.source_parameters.keys()))
		# Chekc that we have a colossus cosmology
		self.assertTrue(isinstance(self.sb.cosmo,cosmology.Cosmology))

	def test_check_parameterization(self):
		# Test that it raises the error if something is missing,
		# doesn't raise it if everything (and more) is there
		parameters_extra = ['sigma_sub','shmf_plaw_index','m_pivot','m_min',
			'm_max','c_0','conc_xi','conc_beta','conc_m_ref','dex_scatter',
			'extra']
		parameters_less = ['sigma_sub','shmf_plaw_index','m_pivot','m_min',
			'm_max','c_0','conc_xi','conc_beta','conc_m_ref']

		# Make sure no error is raised if everything is there
		self.sb.check_parameterization(parameters_less)

		# Check that an error is raised if something is missing
		with self.assertRaises(ValueError):
			self.sb.check_parameterization(parameters_extra)

	def test_draw_subhalos(self):
		# Just check that the NotImplementedError is raised
		with self.assertRaises(NotImplementedError):
			self.sb.draw_subhalos()


class SubhalosDG19Tests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)

		self.subhalo_parameters = {'sigma_sub':4e-2, 'shmf_plaw_index': -1.83,
			'm_pivot': 1e8, 'm_min': 1e6, 'm_max': 1e9, 'c_0':18,
			'conc_xi':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.0, 'distribution':'DG_19'}
		self.main_deflector_parameters = {'M200': 1e13, 'z_lens': 0.5,
			'theta_E':0.38, 'center_x':0.0, 'center_y': 0.0}
		self.source_parameters = {'z_source':1.5}
		self.cosmology_parameters = {'cosmology_name': 'planck18'}
		self.sd = SubhalosDG19(self.subhalo_parameters,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)
		self.cosmo = self.sd.cosmo

	def test_host_scaling_function(self):
		# Just test that the function agrees with some pre-computed values
		host_m200_list = [1e10,1e11,2.5e11]
		z_lens_list = [0.3,0.4,0.6]
		pre_computed_list = [0.0015676639733375364,0.014528215260579059,
			0.04576724520325134]

		for i in range(len(host_m200_list)):
			self.assertAlmostEqual(SubhalosDG19.host_scaling_function(
				host_m200_list[i],z_lens_list[i]),pre_computed_list[i])

	def test_draw_nfw_masses(self):
		# Test that the mass function draws agree with the input parameters
		# Add the parameter we need
		subhalo_parameters = {'sigma_sub':4e-2, 'shmf_plaw_index': -1.83,
			'm_pivot': 1e8, 'm_min': 1e6, 'm_max': 1e9, 'c_0':18,
			'conc_xi':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.0}

		# Calculate the norm by hand and make sure the statistics agree
		kpc_per_arcsecond = (self.cosmo.angularDiameterDistance(
			self.main_deflector_parameters['z_lens']) / self.cosmo.h * 1000
			*  np.pi/180/3600)
		r_E = kpc_per_arcsecond*self.main_deflector_parameters['theta_E']
		dA = np.pi * (3*r_E)**2
		f_host = SubhalosDG19.host_scaling_function(
			self.main_deflector_parameters['M200'],
			self.main_deflector_parameters['z_lens'])
		e_counts =  power_law.power_law_integrate(subhalo_parameters['m_min'],
			subhalo_parameters['m_max'],subhalo_parameters['shmf_plaw_index'])
		norm_without_sigma_sub = dA*f_host*(
			subhalo_parameters['m_pivot']**(-
				subhalo_parameters['shmf_plaw_index']-1))*e_counts
		desired_count = 1e2
		self.sd.subhalo_parameters['sigma_sub'] = (desired_count /
			norm_without_sigma_sub)

		total_subs = 0
		n_loops = 5000
		for _ in range(n_loops):
			masses = self.sd.draw_nfw_masses()
			total_subs += len(masses)
		self.assertEqual(np.round(total_subs/n_loops),desired_count)

		# Now just give some parameters for an HE0435-1223 galaxy and make sure
		# what we return is reasonable as a sanity check on units.
		self.sd.subhalo_parameters['sigma_sub'] = 4e-2
		total_subs = 0
		for _ in range(n_loops):
			masses = self.sd.draw_nfw_masses()
			total_subs += len(masses)
		self.assertGreater(total_subs//n_loops,100)
		self.assertLess(total_subs//n_loops,500)

	def test_mass_concentration(self):
		# Test that the mass concentration relationship has thr right scatter
		n_haloes = 10000
		z = np.ones(n_haloes)*0.2
		m_200 = np.logspace(6,10,n_haloes)
		concentrations = self.sd.mass_concentration(z,m_200)

		h = self.cosmo.h
		peak_heights = peaks.peakHeight(m_200*h,z)
		peak_heights_ref = peaks.peakHeight(1e8*h,0)
		np.testing.assert_almost_equal(concentrations,18*1.2**(-0.2)*(
			peak_heights/peak_heights_ref)**(-0.8))

		# Test that scatter works as desired
		self.sd.subhalo_parameters['dex_scatter'] = 0.1
		m_200 = np.logspace(6,10,n_haloes)
		concentrations = self.sd.mass_concentration(z,m_200)
		scatter = np.log10(concentrations) - np.log10(18*1.2**(-0.2)*(
			peak_heights/peak_heights_ref)**(-0.8))
		self.assertAlmostEqual(np.std(scatter),
			self.sd.subhalo_parameters['dex_scatter'],places=2)
		self.assertAlmostEqual(np.mean(scatter),0.0,places=2)

		# Check that things don't crash if you pass in a float for the
		# mass
		z = 0.2
		m_200 =1e9
		concentrations = self.sd.mass_concentration(z,m_200)

	def test_rejection_sampling(self):
		# Test the the numba rejection sampling works as expected
		# Start with bounds that accept everything
		n_samps = int(1e6)
		r_samps = np.ones(n_samps)
		r_200 = 2
		r_3E = 2
		keep, cart_pos = SubhalosDG19.rejection_sampling(r_samps,r_200,r_3E)

		# Check the the stats agree with what we put in
		self.assertEqual(np.mean(keep),1.0)
		np.testing.assert_almost_equal(np.sqrt(np.sum(cart_pos**2,axis=-1)),
			r_samps)
		phi = np.arccos(cart_pos[:,2]/r_samps)
		theta = np.arctan(cart_pos[:,1]/cart_pos[:,0])
		self.assertAlmostEqual(np.mean(phi),np.pi/2,places=2)
		self.assertAlmostEqual(np.mean(theta),0,places=2)
		for i in range(3):
			self.assertAlmostEqual(np.max(cart_pos[:,i])-np.min(cart_pos[:,i]),
				2,places=4)

		# Set boundaries and make sure that the keep draws are within the
		# boundaries
		n_samps = int(1e4)
		r_samps = np.random.rand(n_samps)
		r_200 = 0.9
		r_3E = 0.2
		keep, cart_pos = SubhalosDG19.rejection_sampling(r_samps,r_200,r_3E)

		self.assertLess(np.mean(keep),1)
		cart_pos = cart_pos[keep]
		self.assertEqual(np.mean(np.sqrt(cart_pos[:,0]**2+
			cart_pos[:,1]**2)<r_3E),1)
		self.assertEqual(np.mean(np.abs(cart_pos[:,2])<r_200),1)

	def test_sample_cored_nfw(self):
		# Test that the sampling code returns the desired number of
		# values
		n_subs = int(1e5)
		cart_pos = self.sd.sample_cored_nfw(n_subs)

		# Basic check that the array is long enough and that no value
		# hasn't been overwritten
		self.assertEqual(len(cart_pos),n_subs)
		self.assertEqual(np.sum(cart_pos==0),0)

		# We can't check against an analytic distribution, but we can
		# check that the likelihood of the radius decreases with increase
		# radius
		r_vals = np.sqrt(np.sum(cart_pos**2,axis=-1))
		r_bins, _ = np.histogram(r_vals,bins=100)
		self.assertGreater(r_bins[0],r_bins[5])
		self.assertGreater(r_bins[5],r_bins[-1])
		self.assertGreater(r_bins[0],r_bins[2])

	def test_get_truncation_radius(self):
		# Test that the returned radius agrees with some precomputed values
		m_200 = np.logspace(6,9,10)
		r = np.linspace(10,300,10)
		pre_comp_vals = np.array([0.22223615,0.74981341,1.4133784,2.32006694,
			3.57303672,5.30341642,7.68458502,10.94766194,15.40115448,
			21.45666411])
		r_trunc = SubhalosDG19.get_truncation_radius(m_200,r)

		np.testing.assert_almost_equal(r_trunc,pre_comp_vals,decimal=5)

	def test_draw_subhalos(self):
		# Check that the draw returns parameters that can be passsed
		# into lenstronomy. This will also test convert_to_lenstronomy.
		subhalo_model_list, subhalo_kwargs_list = self.sd.draw_subhalos()

		for subhalo_model in subhalo_model_list:
			self.assertEqual(subhalo_model,'TNFW')

		required_keys_DG_19 = ['alpha_Rs','Rs','center_x','center_y','r_trunc']
		for subhalo_kwargs in subhalo_kwargs_list:
			self.assertTrue(all(elem in subhalo_kwargs.keys() for
				elem in required_keys_DG_19))
