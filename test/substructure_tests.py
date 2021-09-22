import unittest
from manada.Substructure import nfw_functions
from manada.Substructure.subhalos_base import SubhalosBase
from manada.Substructure.subhalos_dg19 import SubhalosDG19
from manada.Substructure.los_base import LOSBase
from manada.Substructure.los_dg19 import LOSDG19
from manada.Utils import cosmology_utils
from colossus.lss.mass_function import modelSheth99, massFunction
from manada.Utils import power_law
import numpy as np
from scipy.integrate import quad
from colossus.cosmology import cosmology
from colossus.halo.concentration import peaks
from colossus.lss import bias
from colossus.halo import profile_nfw
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
import lenstronomy.Util.util as util


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
				return rho_nfw/(x_tidal*(1+x_tidal)**2)*r**2/r_tidal**2
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
		n_subs = int(2e5)
		r_draws = nfw_functions.cored_nfw_draws(r_tidal,rho_nfw,r_scale,
			r_max,n_subs)

		n_test_points = 100
		r_test = np.linspace(0,r_max,n_test_points)
		analytic_integral = nfw_functions.cored_nfw_integral(r_tidal,rho_nfw,
			r_scale,r_test)

		for i in range(n_test_points):
			self.assertAlmostEqual(np.mean(r_draws<r_test[i]),
				analytic_integral[i]/np.max(analytic_integral),places=2)

		# Make sure that within a square they are uniformly distributed
		cart_pos = np.zeros((n_subs,3))
		theta = np.random.rand(n_subs) * 2 * np.pi
		phi = np.arccos(1-2*np.random.rand(n_subs))
		cart_pos[:,0] += r_draws*np.sin(phi)*np.cos(theta)
		cart_pos[:,1] += r_draws*np.sin(phi)*np.sin(theta)
		cart_pos[:,2] += r_draws*np.cos(phi)

		d_max = np.sqrt(2)*r_tidal
		xwhere = np.abs(cart_pos[:,0]) < d_max/2
		ywhere = np.abs(cart_pos[:,1]) < d_max/2
		zwhere = np.abs(cart_pos[:,2]) < d_max/2
		where = xwhere * ywhere * zwhere
		cart_pos = cart_pos[where]
		for i in range(3):
			self.assertAlmostEqual(2*np.mean(cart_pos[:,i]<-d_max/4),
				np.mean(cart_pos[:,i]>0),places=1)

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

	def test_m_c_from_rho_r_scale(self):
		# Just test that the inverse works well
		cosmo = cosmology.setCosmology('planck18')
		n_samps = 100
		m_200 = np.logspace(7,10,n_samps)
		c = np.linspace(1,40,n_samps)
		z_lens = 1.2

		r200 = nfw_functions.r_200_from_m(m_200,z_lens,cosmo)
		rho_nfw = nfw_functions.rho_nfw_from_m_c(m_200,c,cosmo,z=z_lens)

		m_inv,c_inv = nfw_functions.m_c_from_rho_r_scale(rho_nfw,r200/c,cosmo,
			z_lens)

		np.testing.assert_almost_equal(m_200/m_inv,1,decimal=2)
		np.testing.assert_almost_equal(c,c_inv,decimal=2)

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
		n_samps = 100
		z_halo = np.linspace(0.2,1.2,n_samps)
		z_source = 1.5
		m_200 = np.logspace(8,9,n_samps)
		c = np.linspace(4,5,n_samps)

		# Do the vectorized calculation using our code
		r_scale = nfw_functions.r_200_from_m(m_200,z_halo,cosmo)/c
		rho_nfw = nfw_functions.rho_nfw_from_m_c(m_200,c,cosmo,r_scale=r_scale)
		r_scale_angle, alpha_rs = nfw_functions.convert_to_lenstronomy_NFW(
			r_scale,z_halo,rho_nfw,z_source,cosmo)

		# Do the physical calculation in lenstronomy
		for i in range(n_samps):
			lens_cosmo = LensCosmo(z_lens=z_halo[i], z_source=z_source,
				cosmo=cosmo_astropy)
			rho0, Rs, r200 = lens_cosmo.nfwParam_physical(M=m_200[i], c=c[i])
			rs_angle_ls, alpha_rs_ls = lens_cosmo.nfw_physical2angle(M=m_200[i],
				c=c[i])

			mpc_2_kpc = 1e3
			self.assertAlmostEqual(r_scale[i],Rs*mpc_2_kpc)
			self.assertAlmostEqual(rho_nfw[i]/(rho0/(mpc_2_kpc**3)),1)
			self.assertAlmostEqual(r_scale_angle[i],rs_angle_ls,places=2)
			self.assertAlmostEqual(alpha_rs[i],alpha_rs_ls)

	def test_convert_from_lenstronomy_NFW(self):
		# Just make sure this is a valid inverse transform
		cosmo = cosmology.setCosmology('planck18')
		# Our calculations are always at z=0.
		n_samps = 100
		z_halo = np.linspace(0.2,1.2,n_samps)
		z_source = 1.5
		m_200 = np.logspace(8,9,n_samps)
		c = np.linspace(4,5,n_samps)

		# Do the vectorized calculation using our code
		r_scale = nfw_functions.r_200_from_m(m_200,z_halo,cosmo)/c
		rho_nfw = nfw_functions.rho_nfw_from_m_c(m_200,c,cosmo,r_scale=r_scale)
		r_scale_angle, alpha_rs = nfw_functions.convert_to_lenstronomy_NFW(
			r_scale,z_halo,rho_nfw,z_source,cosmo)
		r_scale_inv, rho_nfw_inv = (
			nfw_functions.convert_from_lenstronomy_NFW(r_scale_angle,
				alpha_rs,z_halo,z_source,cosmo))

		np.testing.assert_almost_equal(r_scale,r_scale_inv)
		np.testing.assert_almost_equal(rho_nfw,rho_nfw_inv)

	def test_convert_to_lenstronomy_tNFW(self):
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
			nfw_functions.convert_to_lenstronomy_tNFW(r_scale,z_lens,rho_nfw,
			r_trunc,z_source,cosmo))

		mpc_2_kpc = 1e3
		self.assertAlmostEqual(r_scale,Rs*mpc_2_kpc)
		self.assertAlmostEqual(rho_nfw,rho0/(mpc_2_kpc**3),places=2)
		self.assertAlmostEqual(r_scale_angle,rs_angle_ls,places=2)
		self.assertAlmostEqual(alpha_rs,alpha_rs_ls)
		self.assertAlmostEqual(r_trunc_ang*r_scale/r_scale_angle,r_trunc)

	def test_convert_from_lenstronomy_tNFW(self):
		# Just make sure this is a valid inverse transform
		cosmo = cosmology.setCosmology('planck18')
		# Our calculations are always at z=0.
		n_samps = 100
		z_halo = np.linspace(0.2,1.2,n_samps)
		z_source = 1.5
		m_200 = np.logspace(8,9,n_samps)
		c = np.linspace(4,5,n_samps)

		# Do the vectorized calculation using our code
		r_scale = nfw_functions.r_200_from_m(m_200,z_halo,cosmo)/c
		r_trunc = np.ones(r_scale.shape)
		rho_nfw = nfw_functions.rho_nfw_from_m_c(m_200,c,cosmo,r_scale=r_scale)
		r_scale_angle, alpha_rs, r_trunc_ang = (
			nfw_functions.convert_to_lenstronomy_tNFW(r_scale,z_halo,rho_nfw,
				r_trunc,z_source,cosmo))
		r_scale_inv, rho_nfw_inv, r_trunc_inv = (
			nfw_functions.convert_from_lenstronomy_tNFW(r_scale_angle,
				alpha_rs,r_trunc_ang,z_halo,z_source,cosmo))

		np.testing.assert_almost_equal(r_scale,r_scale_inv)
		np.testing.assert_almost_equal(rho_nfw,rho_nfw_inv)
		np.testing.assert_almost_equal(r_trunc,r_trunc_inv)


class SubhalosBaseTests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)

		# Make the subhalo base class we'll use for testing.
		self.subhalo_parameters = {'sigma_sub':4e-2, 'shmf_plaw_index': -1.83,
			'm_pivot': 1e8, 'm_min': 1e6, 'm_max': 1e9, 'c_0':18,
			'conc_zeta':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.0}
		self.main_deflector_parameters = {'M200': 1e13, 'z_lens': 0.5,
			'theta_E':1, 'center_x':0.0, 'center_y': 0.0}
		self.source_parameters = {'z_source':1.5}
		self.cosmology_parameters = {'cosmology_name': 'planck18'}
		self.sb = SubhalosBase(self.subhalo_parameters,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)
		self.cosmo = self.sb.cosmo

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
			'm_max','c_0','conc_zeta','conc_beta','conc_m_ref','dex_scatter',
			'extra']
		parameters_less = ['sigma_sub','shmf_plaw_index','m_pivot','m_min',
			'm_max','c_0','conc_zeta','conc_beta','conc_m_ref']

		# Make sure no error is raised if everything is there
		self.sb.check_parameterization(parameters_less)

		# Check that an error is raised if something is missing
		with self.assertRaises(ValueError):
			self.sb.check_parameterization(parameters_extra)

	def test_update_parameters(self):
		# Check that the parameters are updated
		subhalo_parameters = {'sigma_sub':1e-2, 'shmf_plaw_index': -1.83,
			'm_pivot': 1e8, 'm_min': 1e6, 'm_max': 1e9, 'c_0':18,
			'conc_zeta':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.0, 'distribution':'DG_19'}
		main_deflector_parameters = {'M200': 1e12, 'z_lens': 0.4,
			'theta_E':0.32, 'center_x':0.1, 'center_y': 0.1}
		source_parameters = {'z_source':1.9}
		cosmology_parameters = {'cosmology_name': 'WMAP7'}
		self.sb.update_parameters(subhalo_parameters=subhalo_parameters,
			main_deflector_parameters=main_deflector_parameters,
			source_parameters=source_parameters,
			cosmology_parameters=cosmology_parameters)

		self.assertFalse(self.sb.subhalo_parameters['sigma_sub'] ==
			self.subhalo_parameters['sigma_sub'])
		self.assertFalse(self.sb.main_deflector_parameters['M200'] ==
			self.main_deflector_parameters['M200'])
		self.assertFalse(self.sb.source_parameters['z_source'] ==
			self.source_parameters['z_source'])
		self.assertFalse(self.sb.cosmo.h == self.cosmo.h)

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
			'conc_zeta':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.0, 'k1':0.88,'k2':1.7}
		self.main_deflector_parameters = {'M200': 1.1e13, 'z_lens': 0.5,
			'theta_E':2.38, 'center_x':0.0, 'center_y': 0.0}
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
			'conc_zeta':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
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

		# Now make sure that getting rid of mass and redshift power law
		# removes the host_scaling_function contribution
		self.sd.subhalo_parameters['k1'] = 0
		self.sd.subhalo_parameters['k2'] = 0
		total_subs = 0
		n_loops = 5000
		for _ in range(n_loops):
			masses = self.sd.draw_nfw_masses()
			total_subs += len(masses)
		self.assertEqual(np.round(total_subs/n_loops),np.round(
			desired_count/f_host))

		# Now just give some parameters for an HE0435-1223 galaxy and make sure
		# what we return is reasonable as a sanity check on units.
		self.sd.subhalo_parameters['k1'] = 0.88
		self.sd.subhalo_parameters['k2'] = 1.7
		self.sd.subhalo_parameters['sigma_sub'] = 4e-2
		self.sd.main_deflector_parameters['theta_E'] = 0.38
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
		# check that the likelihood of the radius roughly follows our
		# expectations.
		r_vals = np.sqrt(np.sum(cart_pos**2,axis=-1))
		r_bins, _ = np.histogram(r_vals,bins=100)
		self.assertLess(r_bins[0],r_bins[2])
		self.assertGreater(r_bins[5],r_bins[-1])

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
		subhalo_model_list, subhalo_kwargs_list, subhalo_z_list = (
			self.sd.draw_subhalos())

		for subhalo_model in subhalo_model_list:
			self.assertEqual(subhalo_model,'TNFW')

		required_keys_DG_19 = ['alpha_Rs','Rs','center_x','center_y','r_trunc']
		for subhalo_kwargs in subhalo_kwargs_list:
			self.assertTrue(all(elem in subhalo_kwargs.keys() for
				elem in required_keys_DG_19))

		self.assertListEqual(subhalo_z_list,[0.5]*len(subhalo_model_list))

		# Check that things still work in the limit of no substructure
		self.sd.subhalo_parameters['sigma_sub'] = 1e-9
		subhalo_model_list, subhalo_kwargs_list, subhalo_z_list = (
			self.sd.draw_subhalos())
		self.assertEqual(len(subhalo_model_list),0)
		self.assertEqual(len(subhalo_kwargs_list),0)
		self.assertEqual(len(subhalo_z_list),0)

		# Check that things still work in the limit of negative substructure
		# values (which should just behave like 0).
		self.sd.subhalo_parameters['sigma_sub'] = -0.1
		subhalo_model_list, subhalo_kwargs_list, subhalo_z_list = (
			self.sd.draw_subhalos())
		self.assertEqual(len(subhalo_model_list),0)
		self.assertEqual(len(subhalo_kwargs_list),0)
		self.assertEqual(len(subhalo_z_list),0)


class LOSBaseTests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)

		self.los_parameters = {'test_1':0.2, 'test_2':0.3}
		self.main_deflector_parameters = {'M200': 1e13, 'z_lens': 0.5,
			'theta_E':0.38, 'center_x':0.0, 'center_y': 0.0}
		self.source_parameters = {'z_source':1.5}
		self.cosmology_parameters = {'cosmology_name': 'planck18'}
		self.lb = LOSBase(self.los_parameters,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)
		self.cosmo = self.lb.cosmo

	def test_check_parameterization(self):
		parameters_extra = ['test_1','test_2','test_3']
		parameters_less = ['test_1','test_2']

		# Make sure no error is raised if everything is there
		self.lb.check_parameterization(parameters_less)

		# Check that an error is raised if something is missing
		with self.assertRaises(ValueError):
			self.lb.check_parameterization(parameters_extra)

	def test_update_parameters(self):
		# Check that updating the parameters works
		los_parameters = {'test_1':0.1, 'test_2':0.1}
		main_deflector_parameters = {'M200': 1e12, 'z_lens': 0.4,
			'theta_E':0.32, 'center_x':0.1, 'center_y': 0.1}
		source_parameters = {'z_source':1.9}
		cosmology_parameters = {'cosmology_name': 'WMAP7'}
		self.lb.update_parameters(los_parameters=los_parameters,
			main_deflector_parameters=main_deflector_parameters,
			source_parameters=source_parameters,
			cosmology_parameters=cosmology_parameters)

		self.assertFalse(self.lb.los_parameters['test_1'] ==
			self.los_parameters['test_1'])
		self.assertFalse(self.lb.main_deflector_parameters['M200'] ==
			self.main_deflector_parameters['M200'])
		self.assertFalse(self.lb.source_parameters['z_source'] ==
			self.source_parameters['z_source'])
		self.assertFalse(self.lb.cosmo.h == self.cosmo.h)

	def test_draw_los(self):
		# Just check that the NotImplementedError is raised
		with self.assertRaises(NotImplementedError):
			self.lb.draw_los()

	def test_calculate_average_alpha(self):
		# Just check that the NotImplementedError is raised
		with self.assertRaises(NotImplementedError):
			self.lb.calculate_average_alpha()


class LOSDG19Tests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)

		self.los_parameters = {'m_min':1e6, 'm_max':1e10,'z_min':0.01,
			'dz':0.01,'cone_angle':4.0,'r_min':0.5,'r_max':10.0,'c_0':18,
			'conc_zeta':-0.2,'conc_beta':0.8,'conc_m_ref': 1e8,
			'dex_scatter': 0.0,'delta_los':1.7,'alpha_dz_factor':1.0}
		self.main_deflector_parameters = {'M200': 1e13, 'z_lens': 0.5,
			'theta_E':0.38, 'center_x':0.0, 'center_y': 0.0}
		self.source_parameters = {'z_source':1.5}
		self.cosmology_parameters = {'cosmology_name': 'planck18'}
		self.ld = LOSDG19(self.los_parameters,
			self.main_deflector_parameters,self.source_parameters,
			self.cosmology_parameters)
		self.cosmo = self.ld.cosmo

	def test_nu_f_nu(self):
		# Compare a few values of nu_f_nu with colossus output.
		z = self.source_parameters['z_source']
		delta_c = peaks.collapseOverdensity(z=z,corrections=True)
		nu = np.linspace(1e-2,10,100)
		sigma = delta_c/nu
		nfn_eval = self.ld.nu_f_nu(nu)
		# Convert for our slightly more precise value of A
		nfn_eval *= 0.3222/0.32218
		np.testing.assert_almost_equal(modelSheth99(sigma,z),
			nfn_eval)

		# Make sure that it integrates to 1
		def f_nu(nu):
			return self.ld.nu_f_nu(nu)/nu
		self.assertAlmostEqual(quad(f_nu,0,1e2)[0],1,places=4)

	def test_dn_dm(self):
		# Test that the outputs of the mass function agrees with colossus
		z = self.source_parameters['z_source']
		m = np.logspace(4,10,100)
		dndm_eval = self.ld.dn_dm(m,z)
		dndlogm_eval = dndm_eval*m

		# Convert the colossus mass function to our units
		col_mf = massFunction(m*self.cosmo.h,z,model='sheth99',q_out='dndlnM',
			deltac_args={'corrections':True})
		# /kpc^3 to /Mpc^3
		dndlogm_eval *= 1e9
		# Convert from rho_m(0) to rho_m(z)
		col_mf *= (1+z)**(3)
		# Convert for our slightly more precise value of A
		col_mf /= 0.3222/0.32218
		# col_mf returns in units of comoving (Mpc/h)**(-3)
		col_mf *= self.cosmo.h**3
		np.testing.assert_almost_equal(dndlogm_eval,col_mf)

		# Repeat for a different redshift
		z = 0.1
		dndm_eval = self.ld.dn_dm(m,z)
		dndlogm_eval = dndm_eval*m

		# Convert the colossus mass function to our units
		col_mf = massFunction(m*self.cosmo.h,z,model='sheth99',q_out='dndlnM',
			deltac_args={'corrections':True})
		# /kpc^3 to /Mpc^3
		dndlogm_eval *= 1e9
		# Convert from rho_m(0) to rho_m(z)
		col_mf *= (1+z)**(3)
		# Convert for our slightly more precise value of A
		col_mf /= 0.3222/0.32218
		# col_mf returns in units of comoving (Mpc/h)**(-3)
		col_mf *= self.cosmo.h**3
		np.testing.assert_almost_equal(dndlogm_eval,col_mf)

	def test_power_law_dn_dm(self):
		# Make sure that the power law fit is a decent explanation of the
		# actual mass function.
		m = np.logspace(6,10,100)
		z=0.1
		dn_dm = self.ld.dn_dm(m,z)
		slope, norm = self.ld.power_law_dn_dm(z,1e6,1e10)
		y = norm*m**slope
		np.testing.assert_almost_equal(dn_dm/y,y/y,decimal=1)

	def test_two_halo_boost(self):
		# Make sure that the two_halo_boost term agrees with some analytical
		# test cases
		# First check that the two halo boost term is 1 far from the
		# main deflector
		dz = 0.01
		z_lens = 1.1
		lens_m200 = 1e13
		r_max = 10
		r_min = 0.5
		n_quads = 1000

		# Behind the light cone
		for z in np.linspace(0,1.08,10):
			self.assertEqual(self.ld.two_halo_boost(z,z_lens,dz,lens_m200,
				r_max,r_min,n_quads=n_quads),1)

		# In front of the light cone
		for z in np.linspace(1.11,4,30):
			self.assertEqual(self.ld.two_halo_boost(z,z_lens,dz,lens_m200,
				r_max,r_min,n_quads=n_quads),1)

		# Check that the values near the lens redshift work
		z = 1.1
		# Conduct the calculation by hand
		z_range = np.linspace(z,z+dz,n_quads)
		r_cmv = -self.cosmo.comovingDistance(z_range,z_lens)
		r_cmv = r_cmv[r_cmv>r_min*self.cosmo.h]
		xi_halo = self.cosmo.correlationFunction(r_cmv,z_lens)
		xi_halo *= bias.haloBias(lens_m200*self.cosmo.h,z_lens,mdef='200c',
			model='tinker10')
		self.assertEqual(self.ld.two_halo_boost(z,z_lens,dz,lens_m200,r_max,
			r_min,n_quads=n_quads),1+np.mean(xi_halo))

		z=1.095
		r_min = 0.01
		z_range = np.linspace(z,z+dz,n_quads)
		r_cmv = self.cosmo.comovingDistance(z_range,z_lens)
		r_cmv[z_range>z_lens] *= -1
		r_cmv = r_cmv[r_cmv>r_min*self.cosmo.h]
		xi_halo = self.cosmo.correlationFunction(r_cmv,z_lens)
		xi_halo *= bias.haloBias(lens_m200*self.cosmo.h,z_lens,mdef='200c',
			model='tinker10')
		self.assertEqual(self.ld.two_halo_boost(z,z_lens,dz,lens_m200,r_max,
			r_min,n_quads=n_quads),1+np.mean(xi_halo))
		# Make sure it's boosting
		self.assertGreater(self.ld.two_halo_boost(z,z_lens,dz,lens_m200,r_max,
			r_min,n_quads=n_quads),1)

	def test_cone_angle_to_radius(self):
		# Test that the radius grows and shrinks as expected.
		z_lens = 1.1
		z_source = 3.5
		cone_angle = 2.0
		z_first = np.arange(0.1,z_lens,0.01)
		r_first = np.zeros(z_first.shape)

		for i in range(len(z_first)):
			# Get the radius in comoving coordinates
			r_first[i] = self.ld.cone_angle_to_radius(z_first[i],z_lens,
				z_source,cone_angle)*(1+z_first[i])

		# Check that our cone is growing at a constant rate
		ratio_first = self.cosmo.comovingDistance(0.0,z_first)/r_first
		self.assertAlmostEqual(np.std(ratio_first),0.0,places=3)
		self.assertGreater(np.mean(ratio_first),0.0)
		np.testing.assert_almost_equal(r_first/(1+z_first),
			cosmology_utils.kpc_per_arcsecond(z_first,self.cosmo)*cone_angle*
			0.5)

		# Repeat the same, but now make sure that the cone is shrinking
		z_second = np.arange(z_lens,z_source,0.01)
		r_second = np.zeros(z_second.shape)
		for i in range(len(z_second)):
			# Get the radius in comoving coordinates
			r_second[i] = self.ld.cone_angle_to_radius(z_second[i],z_lens,
				z_source,cone_angle)*(1+z_second[i])

		r_cmv = self.cosmo.comovingDistance(z_lens,z_second)
		ratio_second = (r_cmv[1:]-r_cmv[:-1])/(r_second[1:]-r_second[:-1])
		self.assertAlmostEqual(np.std(ratio_second),0.0,places=2)
		self.assertLess(np.mean(ratio_second),0.0)

		# Confirm the hand calc
		hand_calc = 0.8
		hand_calc *= (self.cosmo.comovingDistance(z_source)/
			self.cosmo.comovingDistance(z_lens,z_source))
		hand_calc *= (self.cosmo.comovingDistance(z_lens,z_second)/
			self.cosmo.comovingDistance(z_second))
		hand_calc = 1-hand_calc
		hand_calc *= cosmology_utils.kpc_per_arcsecond(z_second,
			self.cosmo)*cone_angle*0.5
		np.testing.assert_almost_equal(r_second/(1+z_second),hand_calc)

		# One last check, if the angle_bufffer is 1 then z_source should
		# return a radius of 0
		angle_buffer = 1.0
		self.assertAlmostEqual(self.ld.cone_angle_to_radius(z_source,z_lens,
			z_source,cone_angle,angle_buffer=angle_buffer),0.0)

	def test_volume_element(self):
		# Check that the returned volume element is what you would expect
		z_lens = 1.1
		z_source = 4.0
		dz = 0.01
		cone_angle = 2.0
		z_range = np.arange(0.01,z_source,0.01)
		r_z = np.zeros(z_range.shape)
		v_z = np.zeros(z_range.shape)

		for i in range(len(z_range)):
			r_z[i] = self.ld.cone_angle_to_radius(z_range[i]+dz/2,z_lens,
				z_source,cone_angle)
			v_z[i] = self.ld.volume_element(z_range[i],z_lens,z_source,dz,
				cone_angle)

		dz_in_kpc = self.cosmo.comovingDistance(z_range,z_range+dz)/(
			1+z_range)/self.cosmo.h*1e3
		np.testing.assert_almost_equal(v_z,1*np.pi*r_z**2*dz_in_kpc)

	def test_draw_nfw_masses(self):
		# Check that the number of masses drawn behave according to the
		# expectation for the power law.
		z = 0.1

		# Manually calculate the expected counts
		pl_slope, pl_norm = self.ld.power_law_dn_dm(z,
			self.los_parameters['m_min'],self.los_parameters['m_max'])
		dV = self.ld.volume_element(z,self.main_deflector_parameters['z_lens'],
			self.source_parameters['z_source'],self.los_parameters['dz'],
			self.los_parameters['cone_angle'])
		power_law_norm = power_law.power_law_integrate(
			self.los_parameters['m_min'],self.los_parameters['m_max'],pl_slope)
		n_expected = pl_norm*dV*power_law_norm*self.los_parameters['delta_los']
		n_loops = 1000
		total = 0
		for _ in range(n_loops):
			total += len(self.ld.draw_nfw_masses(z))
		self.assertAlmostEqual(np.round(total/n_loops),np.round(n_expected))

		z = 0.49
		pl_slope, pl_norm = self.ld.power_law_dn_dm(
			z+self.los_parameters['dz']/2,
			self.los_parameters['m_min'],self.los_parameters['m_max'])
		dV = self.ld.volume_element(z,self.main_deflector_parameters['z_lens'],
			self.source_parameters['z_source'],self.los_parameters['dz'],
			self.los_parameters['cone_angle'])
		power_law_norm = power_law.power_law_integrate(
			self.los_parameters['m_min'],self.los_parameters['m_max'],pl_slope)
		halo_boost = self.ld.two_halo_boost(z,
			self.main_deflector_parameters['z_lens'],self.los_parameters['dz'],
			self.main_deflector_parameters['M200'],self.los_parameters['r_max'],
			self.los_parameters['r_min'])
		n_expected = pl_norm*dV*power_law_norm*halo_boost*self.los_parameters[
			'delta_los']
		n_loops = 1000
		total = 0
		for _ in range(n_loops):
			total += len(self.ld.draw_nfw_masses(z))
		self.assertLess(np.abs(total/n_loops-n_expected),1.0)

	def test_sample_los_pos(self):
		# Test that the positions for the los halos are within the los cone.
		z = 0.1
		n_los = int(1e6)

		cart_pos = self.ld.sample_los_pos(z,n_los)
		r_los = self.ld.cone_angle_to_radius(z+self.los_parameters['dz']/2,
			self.main_deflector_parameters['z_lens'],
			self.source_parameters['z_source'],
			self.los_parameters['cone_angle'])

		# Check the angle distribution
		angles = np.arctan(cart_pos[:,1]/cart_pos[:,0])
		self.assertAlmostEqual(np.mean(angles<0),np.mean(angles>0),places=2)

		# Make sure that within a square they are uniformly distributed
		d_max = np.sqrt(2)*r_los
		xwhere = np.abs(cart_pos[:,0]) < d_max/2
		ywhere = np.abs(cart_pos[:,1]) < d_max/2
		where = xwhere * ywhere
		cart_pos = cart_pos[where]
		self.assertAlmostEqual(2*np.mean(cart_pos[:,0]<-d_max/4),
			np.mean(cart_pos[:,0]>0),places=2)
		self.assertAlmostEqual(2*np.mean(cart_pos[:,1]<-d_max/4),
			np.mean(cart_pos[:,1]>0),places=2)

	def test_mass_concentration(self):
		# Confirm that the concentrations drawn follows expectations
		n_haloes = 10000
		z = np.ones(n_haloes)*0.2
		m_200 = np.logspace(6,10,n_haloes)
		concentrations = self.ld.mass_concentration(z,m_200)

		h = self.cosmo.h
		peak_heights = peaks.peakHeight(m_200*h,z)
		peak_heights_ref = peaks.peakHeight(1e8*h,0)
		np.testing.assert_almost_equal(concentrations,18*1.2**(-0.2)*(
			peak_heights/peak_heights_ref)**(-0.8))

		# Test that scatter works as desired
		self.ld.los_parameters['dex_scatter'] = 0.1
		m_200 = np.logspace(6,10,n_haloes)
		concentrations = self.ld.mass_concentration(z,m_200)
		scatter = np.log10(concentrations) - np.log10(18*1.2**(-0.2)*(
			peak_heights/peak_heights_ref)**(-0.8))
		self.assertAlmostEqual(np.std(scatter),
			self.ld.los_parameters['dex_scatter'],places=2)
		self.assertAlmostEqual(np.mean(scatter),0.0,places=2)

		# Check that things don't crash if you pass in a float for the
		# mass
		z = 0.2
		m_200 =1e9
		concentrations = self.ld.mass_concentration(z,m_200)

	def test_convert_to_lenstronomy(self):
		# Test that the lenstronomy parameters returned match our
		# expectation
		z = 0.02
		n_halos = 1000
		z_masses = 10**(np.random.rand(n_halos)*4+6)
		z_cart_pos = np.random.rand(n_halos*2).reshape((-1,2))

		kpc_per_arcsecond = cosmology_utils.kpc_per_arcsecond(z,self.cosmo)
		z_cart_ang = z_cart_pos / np.expand_dims(kpc_per_arcsecond,axis=-1)

		model_list, kwargs_list = self.ld.convert_to_lenstronomy(z,z_masses,
			z_cart_pos)

		self.assertEqual(len(model_list),n_halos)
		self.assertEqual(len(kwargs_list),n_halos)

		for i, model in enumerate(model_list):
			self.assertEqual(model,'NFW')
			self.assertAlmostEqual(kwargs_list[i]['center_x'],z_cart_ang[i,0])
			self.assertAlmostEqual(kwargs_list[i]['center_y'],z_cart_ang[i,1])

	def test_draw_los(self):
		# Check that the draw returns parameters that can be passsed
		# into lenstronomy. This will also test convert_to_lenstronomy.
		los_model_list, los_kwargs_list, los_z_list = self.ld.draw_los()

		for model in los_model_list:
			self.assertEqual(model,'NFW')

		required_keys = ['alpha_Rs','Rs','center_x','center_y']
		for kwargs in los_kwargs_list:
			self.assertTrue(all(elem in kwargs.keys() for
				elem in required_keys))

		self.assertGreater(np.min(los_z_list),self.los_parameters['z_min'])
		self.assertLess(np.max(los_z_list),self.source_parameters['z_source'])

		# Check that things still work if we set delta_los to negative
		# values
		self.ld.los_parameters['delta_los'] = -1
		los_model_list, los_kwargs_list, los_z_list = self.ld.draw_los()
		self.assertEqual(len(los_model_list),0)
		self.assertEqual(len(los_kwargs_list),0)
		self.assertEqual(len(los_z_list),0)

	def test_calculate_average_alpha(self):
		# Test that the interpolation class on average ensures
		# that each sightline has a deflection of 0.
		# Set the parameters of our alpha calculation
		num_pix = 256
		iml, imk, imz = self.ld.calculate_average_alpha(num_pix)
		n_draws = 1000
		num_pix = 64

		# Test for a few redshifts that the average deflection is 0.
		for zi in [len(imz)//2,-1]:
			ax_avg = np.zeros((num_pix,num_pix))
			ay_avg = np.zeros((num_pix,num_pix))
			z = imz[zi]
			pixel_scale = 0.08/(1+z)
			x_grid, y_grid = util.make_grid(numPix=num_pix,deltapix=pixel_scale)
			for _ in range(n_draws):
				z_masses = self.ld.draw_nfw_masses(z)
				# Don't add anything to the model if no masses were drawn
				if z_masses.size == 0:
					continue
				z_cart_pos = self.ld.sample_los_pos(z,len(z_masses))
				# Convert the mass and positions to lenstronomy models
				# and kwargs and append to our lists.
				model_list, kwargs_list = self.ld.convert_to_lenstronomy(
					z,z_masses,z_cart_pos)
				lm = LensModel(model_list,z,
					self.source_parameters['z_source'],
					cosmo=self.cosmo.toAstropy())
				ax, ay = lm.alpha(x_grid,y_grid,kwargs_list)
				ax = util.array2image(ax)
				ay = util.array2image(ay)
				ax_avg += ax
				ay_avg += ay
			ax_avg /= n_draws
			ay_avg /= n_draws
			# The quickest thing with the interpolation is just to query the
			# lens model.
			lm = LensModel([iml[zi]],imz[zi],
					self.source_parameters['z_source'],
					cosmo=self.cosmo.toAstropy())
			imax, imay = lm.alpha(x_grid,y_grid,[imk[zi]])
			imax = util.array2image(imax)
			imay = util.array2image(imay)
			self.assertLess(np.median(np.abs(np.sqrt(ax_avg**2+ay_avg**2)-
				np.sqrt(imax**2+imay**2))/np.sqrt(ax_avg**2+ay_avg**2)),0.6)
