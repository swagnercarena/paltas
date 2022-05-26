import numpy as np
import unittest
import os
from paltas.Sources.source_base import SourceBase
from paltas.Sources.sersic import SingleSersicSource, DoubleSersicData
from paltas.Sources.galaxy_catalog import GalaxyCatalog
from paltas.Sources.cosmos import COSMOSCatalog, COSMOSSersicCatalog, unfits
from paltas.Sources.cosmos import COSMOSExcludeCatalog, COSMOSIncludeCatalog
from paltas.Sources.cosmos import HUBBLE_ACS_PIXEL_WIDTH
from paltas.Sources.cosmos_sersic import COSMOSSersic
from paltas.Utils.cosmology_utils import get_cosmology, absolute_to_apparent
from paltas.Utils.cosmology_utils import get_k_correction
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util.simulation_util import data_configure_simple
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.Profiles import sersic_utils
from lenstronomy.Util.data_util import cps2magnitude
import scipy
from scipy.integrate import quad
import copy


class SourceBaseTests(unittest.TestCase):

	def setUp(self):
		self.c = SourceBase(
			cosmology_parameters='planck18',
			source_parameters=dict())
		self.cosmo = get_cosmology('planck18')

	def test_update_parameters(self):
		# Check that the update parameter call updates the cosmology
		h = self.c.cosmo.h
		self.c.update_parameters('WMAP9')
		self.assertNotEqual(h,self.c.cosmo.h)

	def test_draw_source(self):
		# Just test that the not implemented error is raised.
		with self.assertRaises(NotImplementedError):
			self.c.draw_source()


class SingleSersicSourceTests(SourceBaseTests):

	def setUp(self):
		self.source_parameters = dict(magnitude=20,output_ab_zeropoint=25,
			R_sersic=1.,n_sersic=2.,e1=0.,e2=0.,center_x=0.,center_y=0.,
			z_source=1.0)
		self.c = SingleSersicSource(
			cosmology_parameters='planck18',
			source_parameters=self.source_parameters)
		self.cosmo = get_cosmology('planck18')

	def test_draw_source(self):
		# Check that lenstronomy produces some non-zero image
		light_models, light_kwargs_list, source_redshift_list = (
			self.c.draw_source())

		self.assertListEqual(source_redshift_list,[1.0])

		lens_model = LensModel(['SPEP'])
		light_model = LightModel(light_models)

		n_pixels = 200
		pixel_width = 0.08
		image_model = ImageModel(
			data_class=ImageData(**data_configure_simple(numPix=n_pixels,
				deltaPix=pixel_width)),
			psf_class=PSF(psf_type='GAUSSIAN', fwhm=0.1 * pixel_width),
			lens_model_class=lens_model,source_model_class=light_model)
		# Create a lens that will do nothing
		lens_kwargs = [{'theta_E': 0.0, 'e1': 0., 'e2': 0., 'gamma': 0.,
			'center_x': 0, 'center_y': 0}]

		image = image_model.image(kwargs_lens=lens_kwargs,
			kwargs_source=light_kwargs_list)
		assert isinstance(image, np.ndarray)
		assert image.sum() > 0

		# Check that the input magnitude was treated as an absolute magnitude
		amp_abs = self.c.mag_to_amplitude(self.source_parameters['magnitude'],
			self.source_parameters['output_ab_zeropoint'],self.source_parameters)
		self.assertLess(light_kwargs_list[0]['amp'],amp_abs)

	def test_mag_to_amplitude(self):
		# Test that the magnitude to amplitude conversion follows our basic
		# intuition.
		# Start by passing in the same magnitude as the zero point and make
		# sure that the total flux is 1.
		mag = 10
		mag_zeropoint = 10
		kwargs_list = {'amp':1.0,'R_sersic':1.0,'n_sersic':2.0,
			'e1':0.0,'e2':0.0,'center_x':0.0,'center_y':0.0}
		amp_class = self.c.mag_to_amplitude(mag,mag_zeropoint,
			kwargs_list)

		# Generate a lenstronomy object with this amplitude and make sure
		# the total flux is 1.
		kwargs_list['amp'] = amp_class
		sersic_model = LightModel(['SERSIC_ELLIPSE'])
		self.assertEqual(sersic_model.total_flux([kwargs_list])[0],1)

		# Now just check that for a brighter magnitude it's greater than 1
		mag = 9.5
		amp_class = self.c.mag_to_amplitude(mag,mag_zeropoint,
			kwargs_list)
		kwargs_list['amp'] = amp_class
		sersic_model = LightModel(['SERSIC_ELLIPSE'])
		self.assertGreater(sersic_model.total_flux([kwargs_list])[0],1)

	def test_get_total_sersic_flux_r(self):
		# Calculates the sersic flux within a given radius
		R_sersic = 3
		n_sersic = 1
		amp_sersic = 1
		b_n = sersic_utils.SersicUtil.b_n(n_sersic)

		# Do the integration and compare to the analytic
		def integrated_flux(r,R_sersic,n_sersic,amp_sersic,b_n):
			return amp_sersic * quad(lambda x: 2*np.pi*x*np.exp(
				-b_n*((x/R_sersic)**(1/n_sersic)-1)),0,r)[0]

		# Try for a few radii
		r = R_sersic
		self.assertAlmostEqual(SingleSersicSource.get_total_sersic_flux_r(
			r,R_sersic,n_sersic,amp_sersic),integrated_flux(r,R_sersic,n_sersic,
			amp_sersic,b_n))
		r = R_sersic/3
		self.assertAlmostEqual(SingleSersicSource.get_total_sersic_flux_r(
			r,R_sersic,n_sersic,amp_sersic),integrated_flux(r,R_sersic,n_sersic,
			amp_sersic,b_n))
		r = R_sersic*3
		self.assertAlmostEqual(SingleSersicSource.get_total_sersic_flux_r(
			r,R_sersic,n_sersic,amp_sersic),integrated_flux(r,R_sersic,n_sersic,
			amp_sersic,b_n))

		# Switch the parameters and compare the results
		R_sersic = 2.4
		n_sersic = 2.6
		amp_sersic = 1.7
		b_n = sersic_utils.SersicUtil.b_n(n_sersic)

		# Try for a few radii
		r = R_sersic
		self.assertAlmostEqual(SingleSersicSource.get_total_sersic_flux_r(
			r,R_sersic,n_sersic,amp_sersic),integrated_flux(r,R_sersic,n_sersic,
			amp_sersic,b_n))
		r = R_sersic/5
		self.assertAlmostEqual(SingleSersicSource.get_total_sersic_flux_r(
			r,R_sersic,n_sersic,amp_sersic),integrated_flux(r,R_sersic,n_sersic,
			amp_sersic,b_n))
		r = R_sersic*5
		self.assertAlmostEqual(SingleSersicSource.get_total_sersic_flux_r(
			r,R_sersic,n_sersic,amp_sersic),integrated_flux(r,R_sersic,n_sersic,
			amp_sersic,b_n))

	def test_get_total_sersic_flux(self):
		# Test that the total integral agrees with the integral in the large
		# r limit.
		R_sersic = 3
		n_sersic = 1
		amp_sersic = 1
		r = 1e3*R_sersic
		self.assertAlmostEqual(SingleSersicSource.get_total_sersic_flux_r(
			r,R_sersic,n_sersic,amp_sersic),
			SingleSersicSource.get_total_sersic_flux(R_sersic,n_sersic,
				amp_sersic))

		R_sersic = 30
		n_sersic = 3.435
		amp_sersic = 1.56
		r = 1e3*R_sersic
		self.assertAlmostEqual(SingleSersicSource.get_total_sersic_flux_r(
			r,R_sersic,n_sersic,amp_sersic),
			SingleSersicSource.get_total_sersic_flux(R_sersic,n_sersic,
				amp_sersic))


class DoubleSersicDataTests(SourceBaseTests):
	def setUp(self):
		super().setUp()
		self.source_parameters = {'magnitude':-17, 'f_bulge':0.5,
		'output_ab_zeropoint':25,'n_bulge':1,'n_disk':4,
		'r_disk_bulge':2,'e1_bulge':0.0,'e2_bulge':0.0,'e1_disk':0.1,
		'e2_disk':0.1,'center_x':0.0,'center_y':0.0,'z_source':0.5}
		self.c = DoubleSersicData(cosmology_parameters='planck18',
			source_parameters=self.source_parameters)
		np.random.seed(4)

	def test_get_bulge_disk_mag(self):
		# Test that the magnitudes translate to a flux ratio sepcified by
		# the fraction.
		def get_flux_from_mag(mag):
			return 10**(-mag/2.5)

		# Get the three relevant magnitudes
		total_magnitude = absolute_to_apparent(
			self.source_parameters['magnitude'],
			self.source_parameters['z_source'],self.cosmo)
		bulge_mag, disk_mag = self.c.get_bulge_disk_mag()

		# Convert the magnitudes to fluxes
		total_flux = get_flux_from_mag(total_magnitude)
		bulge_flux = get_flux_from_mag(bulge_mag)
		disk_flux = get_flux_from_mag(disk_mag)

		self.assertAlmostEqual(total_flux,2*bulge_flux)
		self.assertAlmostEqual(total_flux,2*disk_flux)

		# Try again with a different fraction.
		self.c.update_parameters(source_parameters={'f_bulge':0.99})
		bulge_mag, disk_mag = self.c.get_bulge_disk_mag()
		bulge_flux = get_flux_from_mag(bulge_mag)
		disk_flux = get_flux_from_mag(disk_mag)

		self.assertAlmostEqual(total_flux*0.99,bulge_flux)
		self.assertAlmostEqual(total_flux*0.01,disk_flux)

	def test_get_total_half_light(self):
		# Test the it returns the correct mean value and that the scatter
		# behaves as expected for high / low magnitude galaxies.
		half_lights = []
		for _ in range(int(1e4)):
			half_lights.append(self.c.get_total_half_light())
		self.assertAlmostEqual(np.mean(np.log10(half_lights)),-0.98,places=1)
		self.assertAlmostEqual(np.std(np.log(half_lights)),0.45,places=1)

		# Shift to a higher magnitude and make sure the standard deviation and
		# mean shifts.
		self.c.update_parameters(source_parameters={'magnitude':-23})
		half_lights = []
		for _ in range(int(1e4)):
			half_lights.append(self.c.get_total_half_light())
		self.assertAlmostEqual(np.mean(np.log10(half_lights)),0.46,places=1)
		self.assertAlmostEqual(np.std(np.log(half_lights)),0.27,places=1)

	def test_get_bulge_disk_half_light(self):
		# Test that the model produced by the function respects the total
		# half light radius.
		amp_disk = 1
		amp_bulge = 1
		R_total = self.c.get_total_half_light()
		R_bulge, R_disk = self.c.get_bulge_disk_half_light(R_total)
		n_bulge = self.source_parameters['n_bulge']
		n_disk = self.source_parameters['n_disk']

		# Integrate the light profiles.
		flux_R = self.c.get_total_sersic_flux_r(R_total,R_bulge,n_bulge,
			amp_bulge)
		flux_R += self.c.get_total_sersic_flux_r(R_total,R_disk,n_disk,
			amp_disk)
		flux_total = self.c.get_total_sersic_flux(R_bulge,n_bulge,amp_bulge)
		flux_total += self.c.get_total_sersic_flux(R_disk,n_disk,amp_disk)
		self.assertAlmostEqual(flux_R/flux_total,0.5)

		# Test the two extremes.
		self.c.update_parameters(source_parameters={'f_bulge':1-1e-5})
		R_total = self.c.get_total_half_light()
		R_bulge, R_disk = self.c.get_bulge_disk_half_light(R_total)
		self.assertAlmostEqual(R_total,R_bulge,places=2)

		self.c.update_parameters(source_parameters={'f_bulge':1e-5})
		R_total = self.c.get_total_half_light()
		R_bulge, R_disk = self.c.get_bulge_disk_half_light(R_total)
		self.assertAlmostEqual(R_total,R_disk,places=2)

	def test_draw_source(self):
		# Test that the parameters returned by draw source create
		# lenstronomy light models that match our expectations
		light_models, light_kwargs_list, source_redshift_list = (
			self.c.draw_source())

		self.assertListEqual(source_redshift_list,[0.5,0.5])
		self.assertListEqual(light_models,['SERSIC_ELLIPSE',
			'SERSIC_ELLIPSE'])

		# Check that the light fractions match
		light_model_bulge = LightModel(light_models[:1])
		light_model_disk = LightModel(light_models[1:])

		self.assertAlmostEqual(light_model_bulge.total_flux(
			light_kwargs_list[:1])[0],light_model_disk.total_flux(
			light_kwargs_list[1:])[0])

		lens_model = LensModel(['SPEP'])
		light_model = LightModel(light_models)

		# Check that the ellipticites have been set correctly
		self.assertEqual(light_kwargs_list[0]['e1'],0.0)
		self.assertEqual(light_kwargs_list[0]['e2'],0.0)
		self.assertEqual(light_kwargs_list[1]['e1'],0.1)
		self.assertEqual(light_kwargs_list[1]['e2'],0.1)

		# Make sure the sources play well with lenstronomy
		n_pixels = 200
		pixel_width = 0.08
		image_model = ImageModel(
			data_class=ImageData(**data_configure_simple(numPix=n_pixels,
				deltaPix=pixel_width)),
			psf_class=PSF(psf_type='GAUSSIAN', fwhm=0.1 * pixel_width),
			lens_model_class=lens_model,source_model_class=light_model)
		# Create a lens that will do nothing
		lens_kwargs = [{'theta_E': 1e-10, 'e1': 0., 'e2': 0., 'gamma': 0.,
			'center_x': 0, 'center_y': 0}]

		image = image_model.image(kwargs_lens=lens_kwargs,
			kwargs_source=light_kwargs_list)
		assert isinstance(image, np.ndarray)
		assert image.sum() > 0


class GalaxyCatalogTests(SourceBaseTests):

	def setUp(self):
		self.c = GalaxyCatalog(cosmology_parameters='planck18',
			source_parameters={'random_rotation':False,
				'output_ab_zeropoint':None,'z_source':1.5,
				'center_x':0.0,'center_y':0.0})
		self.cosmo = get_cosmology('planck18')

	def test__len__(self):
		# Just test that the not implemented error is raised.
		with self.assertRaises(NotImplementedError):
			self.c.__len__()

	def test_image_and_metadata(self):
		# Just test that the not implemented error is raised.
		catalog_i = 2
		with self.assertRaises(NotImplementedError):
			self.c.image_and_metadata(catalog_i)

	def test_iter_lightmodel_kwargs_samples(self):
		# Just test that the not implemented error is raised.
		n_galaxies = 10
		z_new = 1.5
		with self.assertRaises(NotImplementedError):
			for _ in self.c.iter_lightmodel_kwargs_samples(n_galaxies,z_new):
				continue

	def test_iter_image_and_metadata(self):
		# Just test that the not implemented error is raised.
		with self.assertRaises(NotImplementedError):
			for _ in self.c.iter_image_and_metadata():
				continue

	def test_sample_indices(self):
		# Just test that the not implemented error is raised.
		n_galaxies = 10
		with self.assertRaises(NotImplementedError):
			self.c.sample_indices(n_galaxies)

	def test_draw_source(self):
		# Just test that the not implemented error is raised.
		catalog_i = 2
		with self.assertRaises(NotImplementedError):
			self.c.draw_source(catalog_i)

		# Now implement a fake draw_source function
		def fake_image_and_metadata(catalog_i):
			image = np.ones((64,64))
			metadata = {'pixel_width':1.0,'z':1.5}
			return image,metadata
		self.c.image_and_metadata=fake_image_and_metadata

		# Nothing changes if the two zeropoints are the same
		self.c.__class__.ab_zeropoint = 25.0
		self.c.source_parameters['output_ab_zeropoint'] = 25.0
		lens_model,lens_kwargs,source_z_list = self.c.draw_source(1)
		self.assertListEqual(source_z_list,[1.5])
		np.testing.assert_almost_equal(np.ones((64,64)),
			lens_kwargs[0]['image'])

		# The image gets brighter if the output telescope has a larger
		# zeropoint.
		self.c.source_parameters['output_ab_zeropoint'] = 26.0
		lens_model,lens_kwargs,source_z_list = self.c.draw_source(1)
		np.testing.assert_almost_equal(np.ones((64,64))*10**(1/2.5),
			lens_kwargs[0]['image'])

		# Test the absolute magnitude was used correctly
		self.c.source_parameters['source_absolute_magnitude'] = -21
		mag_apparent = absolute_to_apparent(
			self.c.source_parameters['source_absolute_magnitude'],
			self.c.source_parameters['z_source'],self.c.cosmo)
		light_model,light_kwargs,source_z_list = self.c.draw_source(1)
		lm = LightModel(light_model)
		flux_total = lm.total_flux(light_kwargs)
		self.assertAlmostEqual(mag_apparent,cps2magnitude(flux_total,
			self.c.source_parameters['output_ab_zeropoint']))

	def test_k_correct_image(self):
		# Test that the k-correction behaves as expected.
		image = np.random.randn(64,64)
		image_copy = copy.deepcopy(image)
		self.c.k_correct_image(image,0.5,0.5)
		np.testing.assert_almost_equal(image,image_copy)

		# Test that moving an object back in redshift space makes it dimmer
		image = np.random.randn(64,64)
		image_copy = copy.deepcopy(image)
		self.c.k_correct_image(image,0.5,0.8)
		np.testing.assert_array_less(np.abs(image),np.abs(image_copy))

		# Test that the magnitude change matches in the redshift 0 limit.
		image = np.abs(np.random.randn(64,64))
		image_copy = copy.deepcopy(image)
		self.c.k_correct_image(image,0,0.8)
		mag_orig = -2.5*np.log10(np.sum(image_copy))
		mag_k = -2.5*np.log10(np.sum(image))
		self.assertAlmostEqual(mag_orig+get_k_correction(0.8),mag_k)

		# Do the same for two different magnitudes
		image = np.abs(np.random.randn(64,64))
		image_copy = copy.deepcopy(image)
		self.c.k_correct_image(image,0.23,0.8)
		mag_orig = -2.5*np.log10(np.sum(image_copy))
		mag_k = -2.5*np.log10(np.sum(image))
		self.assertAlmostEqual(mag_orig+get_k_correction(0.8),
			mag_k+get_k_correction(0.23))

	def test_normalize_to_mag(self):
		# Test that the magnitude normalization agrees with the
		# lenstronomy calculations.

		# Generate an artificial image
		image = np.random.randn(64,64)
		mag_zeropoint = -16.565
		mag_apparent = -18
		pixel_width = 0.04
		self.c.normalize_to_mag(image,mag_apparent,mag_zeropoint,
			pixel_width)

		# Use the lens model for full consistency
		light_model = LightModel(['INTERPOL'])
		flux_lenstronomy = light_model.total_flux([{'image':image,
			'center_x':0.0,'center_y':0.0,'phi_G':0.0,
			'scale':pixel_width}])[0]
		self.assertAlmostEqual(cps2magnitude(flux_lenstronomy,mag_zeropoint),
			mag_apparent)

		# Try for one more configutation
		image = np.random.randn(64,64)
		mag_zeropoint = 2.565
		mag_apparent = 5
		pixel_width = 0.032
		self.c.normalize_to_mag(image,mag_apparent,mag_zeropoint,
			pixel_width)
		flux_lenstronomy = light_model.total_flux([{'image':image,
			'center_x':0.0,'center_y':0.0,'phi_G':0.0,
			'scale':pixel_width}])[0]
		self.assertAlmostEqual(cps2magnitude(flux_lenstronomy,mag_zeropoint),
			mag_apparent)

	def test_draw_phi(self):
		# Test that draw_phi returns a uniform distribution
		phis = []
		for _ in range(int(1e5)):
			phis.append(self.c.draw_phi())
		self.assertGreater(np.min(phis),0.0)
		self.assertLess(np.max(phis),2*np.pi)
		self.assertAlmostEqual(np.mean(phis), np.pi, places=1)

	def test_z_scale_factor(self):
		# Test that the scale factor is reasonable
		z_old = 0.2
		z_new = 0.2
		self.assertAlmostEqual(self.c.z_scale_factor(z_old,z_new),1.0)

		z_new = 1.5
		self.assertAlmostEqual(self.c.z_scale_factor(z_old,z_new),
			self.cosmo.angularDiameterDistance(z_old)/
			self.cosmo.angularDiameterDistance(z_new))


class COSMOSCatalogTests(SourceBaseTests):

	def setUp(self):
		# Use a trimmed version of cosmo data for testing.
		self.test_cosmo_folder = (os.path.dirname(
			os.path.abspath(__file__))+'/test_data/cosmos/')
		self.source_parameters = {
			'smoothing_sigma':0, 'max_z':None, 'minimum_size_in_pixels':None,
			'faintest_apparent_mag':None,'cosmos_folder':self.test_cosmo_folder,
			'random_rotation':False, 'min_flux_radius':None,
			'center_x':0.0,'center_y':0.0,
			'output_ab_zeropoint':25.95, 'z_source':1.5
		}
		self.c = COSMOSCatalog(cosmology_parameters='planck18',
			source_parameters=self.source_parameters)

		# Fix the seed so we don't have issues with randomness in tests
		np.random.seed(10)

		# Keys we'll want to check for
		self.rkeys = ['IDENT','RA','DEC','MAG','BAND','WEIGHT','GAL_FILENAME',
			'PSF_FILENAME','GAL_HDU','PSF_HDU','PIXEL_SCALE','NOISE_MEAN',
			'NOISE_VARIANCE','NOISE_FILENAME','stamp_flux']
		self.rfkeys = ['IDENT','mag_auto','flux_radius','zphot','sersicfit',
			'bulgefit','fit_status','fit_mad_s','fit_mad_b','fit_dvc_btt',
		'use_bulgefit','viable_sersic','hlr','flux']

	def tearDown(self):
		os.remove(self.test_cosmo_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(self.test_cosmo_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(self.test_cosmo_folder+'npy_files')

	def test_no_cosmos_folder(self):
		# Check that if the cosmos folder is not there, a helpful ValueError
		# is raised
		self.source_parameters['cosmos_folder'] = 'not_here'
		with self.assertRaises(ValueError):
			self.c = COSMOSCatalog(cosmology_parameters='planck18',
				source_parameters=self.source_parameters)

	def test_check_parameterization(self):
		# Check that trying to initialize a class without the correct
		# parameters raises a value error.
		with self.assertRaises(ValueError):
			COSMOSCatalog(cosmology_parameters='planck18',
				source_parameters={})

	def test_unfits(self):
		# Check that the returned arrays have the right elements and size.
		rfits = os.path.join(self.test_cosmo_folder,
			'real_galaxy_catalog_23.5.fits')
		rffits = os.path.join(self.test_cosmo_folder,
			'real_galaxy_catalog_23.5_fits.fits')

		# Use unfits on our data
		rarray = unfits(rfits)
		rfarray = unfits(rffits)

		self.assertEqual(len(rarray),10)
		self.assertEqual(len(rfarray),10)

		self.assertTrue(all(elem in rarray.dtype.names for elem in self.rkeys))
		self.assertTrue(all(elem in rfarray.dtype.names
			for elem in self.rfkeys))

	def test_file_number(self):
		# Test that the file number extraction works as desired
		test_fn = 'fake_fits_file_n20.fits'
		self.assertEqual(self.c._file_number(test_fn),20)

	def test__len__(self):
		# We've trimmed the length to 10, so make sure it returns that
		self.assertEqual(len(self.c),10)

	def test_update_parameters(self):
		# Check that the update parameter call works
		self.source_parameters['smoothing_sigma'] = 0.06
		self.c.update_parameters(source_parameters=self.source_parameters)
		self.assertEqual(self.c.source_parameters['smoothing_sigma'],0.06)

	def test_image_and_metadata(self):
		catalog_i = 0
		image, metadata = self.c.image_and_metadata(catalog_i)
		np.testing.assert_equal(image.shape,(234, 233))
		self.assertEqual(metadata['mag_auto'],21.04064178466797)
		self.assertEqual(metadata['IDENT'],141190)

		# Test that things still work with smoothing
		new_sp = copy.deepcopy(self.source_parameters)
		new_sp['smoothing_sigma'] = 0.06
		cs = COSMOSCatalog(cosmology_parameters='planck18',
			source_parameters=new_sp)

		# Use this opportunity to make sure the catalogs are identical
		np.testing.assert_equal(cs.catalog,self.c.catalog)

		image_s, metadata_s = cs.image_and_metadata(catalog_i)
		self.assertGreater(np.max(np.abs(image-image_s)),0.01)
		image_check = scipy.ndimage.gaussian_filter(image,
			sigma=new_sp['smoothing_sigma']/HUBBLE_ACS_PIXEL_WIDTH)
		np.testing.assert_almost_equal(image_check,image_s)

	def test_iter_lightmodel_kwargs_samples(self):
		# Just test that we get the expected kwargs
		n_galaxies = 10
		lm_keys_required = ['image','center_x','center_y','phi_G','scale']
		z_new = 1.5
		for lm_list, lm_kwargs in self.c.iter_lightmodel_kwargs_samples(
			n_galaxies,z_new):
			self.assertTrue(all(elem in lm_kwargs[0].keys()
				for elem in lm_keys_required))

	def test_iter_image_and_metadata_bulk(self):
		# Just test that image data is returned and that it agrees with
		# the shape of the images.
		for image, metadata in self.c.iter_image_and_metadata_bulk():
			im_shape = image.shape
			self.assertEqual(im_shape[0],metadata['size_x'])
			self.assertEqual(im_shape[1],metadata['size_y'])
			self.assertTrue(all(elem in metadata.dtype.names for
				elem in self.rkeys+self.rfkeys[1:]))

	def test_sample_indices(self):
		# Test the sampled indices respect the restriction we pass.
		# Sample alot to make sure we get the full range.
		n_galaxies = int(1e4)
		samples = self.c.sample_indices(n_galaxies)
		self.assertEqual(np.min(samples),0)
		self.assertEqual(np.max(samples),9)

		# Repeat the test with some cuts on apparent magnitude.
		# Only the first two entries meet this requirement
		new_sp = copy.deepcopy(self.source_parameters)
		new_sp['faintest_apparent_mag'] = 22
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		self.assertEqual(np.min(samples),0)
		self.assertEqual(np.max(samples),1)

		# Now do the same but with a size cut
		new_sp['faintest_apparent_mag'] = 22.5
		new_sp['minimum_size_in_pixels'] = 90
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0,1,3,7])

		# Test the redshift
		new_sp['max_z'] = 0.5
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0,7])

		# Test the minimum flux radius
		new_sp['min_flux_radius'] = 20
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0])

	def test_fill_catalog_i_phi_defaults(self):
		# Use the user-specified catalog_i & phi_G if provided
		self.assertEqual(
			self.c.fill_catalog_i_phi_defaults(catalog_i=42, phi=42.),
			(42, 42.))

		# Sample random catalog indices otherwise
		results = np.array([
			self.c.fill_catalog_i_phi_defaults()
			for _ in range(100)])

		self.assertGreater(
			len(np.unique(results[:, 0])),
			# Catalog may be << 100 items (indeed, just 10 for this test)
			min(len(self.c) * 0.5, 50))
		# phis should all be zero -- random_rotation is False
		self.assertTrue(np.all(results[:, 1] == 0))

		# Repeat with random_rotation True; now phis should all be distinct
		self.c.update_parameters(source_parameters={'random_rotation': True})
		results = np.array([
			self.c.fill_catalog_i_phi_defaults()
			for _ in range(100)])
		self.assertEqual(len(np.unique(results[:, 1])), 100)

	def test_draw_source(self):
		# Test that the lightmodel kwargs returned are what we would
		# expect to pass into lenstronomy.
		catalog_i = 0
		image, metadata = self.c.image_and_metadata(catalog_i)

		# First don't change the redshift
		self.source_parameters['z_source'] = metadata['z']
		self.c.update_parameters(source_parameters=self.source_parameters)
		lm_list, lm_kwargs, s_z_list = self.c.draw_source(catalog_i)
		lm_kwargs = lm_kwargs[0]
		self.assertEqual(s_z_list[0],metadata['z'])
		self.assertEqual(lm_list[0],'INTERPOL')
		np.testing.assert_equal(lm_kwargs['image'],
			image/lm_kwargs['scale']**2)
		low_z_scale = lm_kwargs['scale']

		# Now change the redshift
		z_new = 1.0
		self.source_parameters['z_source'] = z_new
		self.c.update_parameters(source_parameters=self.source_parameters)
		lm_list, lm_kwargs, s_z_list = self.c.draw_source(catalog_i)
		lm_kwargs = lm_kwargs[0]
		self.assertEqual(s_z_list[0],1.0)
		self.assertEqual(lm_list[0],'INTERPOL')
		# Calculate the k_correction effect
		image_k = copy.copy(image)
		self.c.k_correct_image(image_k,metadata['z'],z_new)
		np.testing.assert_almost_equal(lm_kwargs['image'],
			image_k/metadata['pixel_width']**2)
		high_z_scale = lm_kwargs['scale']

		self.assertLess(high_z_scale,low_z_scale)

		# Grab the cosmo to compare with
		cosmo = get_cosmology('planck18')
		self.assertAlmostEqual(lm_kwargs['scale'],metadata['pixel_width']*
			cosmo.angularDiameterDistance(metadata['z'])/
			cosmo.angularDiameterDistance(z_new))

		# Test that providing no catalog_i is not a problem
		lm_list, lm_kwargs, s_z_list = self.c.draw_source()

		# Test that we get rotations when we set that source parameter to
		# True
		self.source_parameters['random_rotation'] = True
		self.source_parameters['z_source'] = metadata['z']
		self.c.update_parameters(source_parameters=self.source_parameters)
		lm_list, lm_kwargs, s_z_list = self.c.draw_source()
		self.assertNotEqual(lm_kwargs[0]['phi_G'],0)
		self.source_parameters['random_rotation'] = False
		self.c.update_parameters(source_parameters=self.source_parameters)

		# Finally test that if we pass these kwargs into a lenstronomy
		# Interpolation class we get the image we expect.
		lens_model = LensModel(['SPEP'])
		light_model = LightModel(lm_list)

		# Deal with the fact that our catalog is not perfectly square
		image = image[17:-17,:]
		image = image[:,1:]/2 + image[:,:-1]/2
		image = image[:,16:-16]

		n_pixels = 200
		image_model = ImageModel(
			data_class=ImageData(**data_configure_simple(numPix=n_pixels,
				deltaPix=metadata['pixel_width'])),
			psf_class=PSF(psf_type='GAUSSIAN',
				fwhm=0.1 * metadata['pixel_width']),
			lens_model_class=lens_model,source_model_class=light_model)
		# Create a lens that will do nothing
		lens_kwargs = [{'theta_E': 0.0, 'e1': 0., 'e2': 0., 'gamma': 0.,
			'center_x': 0, 'center_y': 0}]
		source_kwargs = self.c.draw_source(catalog_i=catalog_i)[1]

		l_image = image_model.image(kwargs_lens=lens_kwargs,
			kwargs_source=source_kwargs)
		np.testing.assert_almost_equal(l_image,image)

		# Finally test that the images move as expected
		self.source_parameters['z_source'] = metadata['z']
		self.source_parameters['center_x'] = metadata['pixel_width']
		self.source_parameters['center_y'] = metadata['pixel_width']
		self.c.update_parameters(source_parameters=self.source_parameters)
		lm_list, lm_kwargs, s_z_list = self.c.draw_source(catalog_i)

		# Test that if we pass these kwargs into a lenstronomy
		# Interpolation class we get the shifted image.
		light_model = LightModel([lm_list[0]])
		image_model = ImageModel(
			data_class=ImageData(**data_configure_simple(numPix=n_pixels,
				deltaPix=metadata['pixel_width'])),
			psf_class=PSF(psf_type='GAUSSIAN',
				fwhm=0.1 * metadata['pixel_width']),
			lens_model_class=lens_model,source_model_class=light_model)
		source_kwargs = [self.c.draw_source(catalog_i=catalog_i)[1][0]]
		l_image = image_model.image(kwargs_lens=lens_kwargs,
			kwargs_source=source_kwargs)
		np.testing.assert_almost_equal(l_image[1:,1:],image[:-1,:-1])


class COSMOSSersicTests(COSMOSCatalogTests):
	
	def setUp(self):
		super().setUp()
		cosmo = get_cosmology('planck18')
		self.mag_apparent = 50
		mag_absolute = self.mag_apparent - 5 *np.log10(
			cosmo.luminosityDistance(1.5)*1e6/cosmo.h/10)
		self.source_parameters = {
			'smoothing_sigma':0, 'max_z':None, 'minimum_size_in_pixels':None,
			'faintest_apparent_mag':None,'cosmos_folder':self.test_cosmo_folder,
			'random_rotation':False, 'min_flux_radius':None,
			'center_x':0.0,'center_y':0.0,
			'output_ab_zeropoint':25.95, 'z_source':1.5,
			'mag_sersic':mag_absolute,'R_sersic':0.5,'n_sersic':2,
			'e1_sersic':0, 'e2_sersic':0, 'center_x_sersic':0,
			'center_y_sersic':0}

		self.c = COSMOSSersic(cosmology_parameters='planck18',
			source_parameters=self.source_parameters)

	def test_draw_source(self):
		super().test_draw_source()

		catalog_i = 0
		lm_list, lm_kwargs,s_z_list = self.c.draw_source(catalog_i)

		# draw source & make sure model list contains both INTERPOL & SERSIC
		self.assertTrue('INTERPOL' in lm_list)
		self.assertTrue('SERSIC_ELLIPSE' in lm_list)
		self.assertListEqual(s_z_list,[0.19499999284744263,0.19499999284744263])

		# make sure all parameters for sersic are there
		sersic_params = ('amp', 'R_sersic', 'n_sersic', 'e1', 'e2', 
			'center_x', 'center_y')
		for p in sersic_params:
			self.assertTrue(p in lm_kwargs[1].keys())

		# make sure that when you double the magnitude the amp is correct
		zeropoint = 25
		self.source_parameters['output_ab_zeropoint'] = zeropoint
		mag = 10
		self.source_parameters['mag_sersic'] = mag
		self.c.update_parameters(source_parameters=self.source_parameters)
		_, lm_kwargs_mag1,_ = self.c.draw_source(catalog_i)
		self.source_parameters['mag_sersic'] = 2*mag
		self.c.update_parameters(source_parameters=self.source_parameters)
		_, lm_kwargs_mag2,_ = self.c.draw_source(catalog_i)
		ratio_true = 10**(-(mag - zeropoint)/2.5) / 10 **(-(2*mag -
			zeropoint)/2.5)
		ratio_out = lm_kwargs_mag1[1]['amp'] / lm_kwargs_mag2[1]['amp']
		# checks out to 7 decimal places (default)
		self.assertAlmostEqual(ratio_true, ratio_out)

		# test difference betweeen COSMOS+Sersic & SingleSersic

		# define a generic lens model, psf model, data class
		lens_model = LensModel(['SPEP'])
		lens_kwargs = [{'theta_E': 0.5, 'e1': 0., 'e2': 0., 'gamma': 2.0,
			'center_x': 0, 'center_y': 0}]
		psf_class = PSF(psf_type='NONE')
		n_pixels = 128
		pixel_width = 0.08
		data_class = ImageData(**data_configure_simple(
			numPix=n_pixels,deltaPix=pixel_width))

		# Generate SingleSersicSource image
		single_sersic_kwargs = {'R_sersic':0.5, 'n_sersic':2, 'e1':0, 
			'e2':0, 'center_x':0, 'center_y':0}
		# This will test mag_to_amplitude conversion
		single_sersic_kwargs['amp'] = SingleSersicSource.mag_to_amplitude(
			self.mag_apparent,self.source_parameters['output_ab_zeropoint'],
			single_sersic_kwargs)
		light_model = LightModel(['SERSIC_ELLIPSE'])
		complete_image_model = ImageModel(data_class=data_class,
			psf_class=psf_class,lens_model_class=lens_model, 
			source_model_class=light_model)
		im_sersic = complete_image_model.image(kwargs_lens=lens_kwargs,
		kwargs_source=[single_sersic_kwargs])

		# generate COSMOSGalaxy image
		cosmos = COSMOSCatalog(cosmology_parameters='planck18', 
			source_parameters=self.source_parameters)
		light_model_list, cosmos_kwargs,_ = cosmos.draw_source(0)
		light_model = LightModel(light_model_list)
		complete_image_model = ImageModel(data_class=data_class,
			psf_class=psf_class,lens_model_class=lens_model,
			source_model_class=light_model)
		im_cosmos = complete_image_model.image(kwargs_lens=lens_kwargs, 
			kwargs_source=cosmos_kwargs)
		
		# generate COSMOSSersic image
		light_model_list, cosmossersic_kwargs, _ = self.c.draw_source(0)
		light_model = LightModel(light_model_list)
		complete_image_model = ImageModel(data_class=data_class,
			psf_class=psf_class,lens_model_class=lens_model,
			source_model_class=light_model)
		im_cosmossersic = complete_image_model.image(kwargs_lens=lens_kwargs,
			kwargs_source=cosmossersic_kwargs)

		# test image diff to ensure we get the same thing back
		np.testing.assert_almost_equal(im_sersic,im_cosmossersic-im_cosmos)
		

class COSMOSSercicCatalogTests(COSMOSCatalogTests):

	def setUp(self):
		super().setUp()
		self.c = COSMOSSersicCatalog(
			cosmology_parameters='planck18',
			source_parameters=self.source_parameters)

	def test_iter_image_and_metadata_bulk(self):
		with self.assertRaises(NotImplementedError):
			self.c.iter_image_and_metadata_bulk()

	def test_image_and_metadata(self):
		with self.assertRaises(NotImplementedError):
			self.c.image_and_metadata(catalog_i=2)

	def test_iter_lightmodel_kwargs_samples(self):
		# Just test that we get the expected kwargs
		n_galaxies = 10
		z_new = 1.5
		lm_keys_required = 'amp e1 e2 R_sersic n_sersic'.split()
		for lm_list, lm_kwargs in self.c.iter_lightmodel_kwargs_samples(
			n_galaxies,z_new):
			self.assertTrue(all(elem in lm_kwargs[0].keys()
				for elem in lm_keys_required))

	def test_draw_source(self):
		# Check lenstronomy eats what we are feeding it
		catalog_i = 2
		metadata = self.c.catalog[catalog_i]

		# Test providing catalog_i
		lm_list, lm_kwargs, s_z_list = self.c.draw_source(catalog_i)
		self.assertListEqual(s_z_list,[1.5])
		self.assertEqual(lm_list[0],'SERSIC_ELLIPSE')

		# Test providing no catalog_i
		self.c.draw_source()

		# Test that we get rotations when we set that source parameter to
		# True
		self.source_parameters['random_rotation'] = True
		self.c.update_parameters(source_parameters=self.source_parameters)
		_, lm_kwargs_rotated,_ = self.c.draw_source(
			catalog_i=catalog_i)
		self.assertNotEqual(lm_kwargs[0]['e1'], lm_kwargs_rotated[0]['e1'])
		self.assertNotEqual(lm_kwargs[0]['e2'], lm_kwargs_rotated[0]['e2'])
		self.source_parameters['random_rotation'] = False
		self.c.update_parameters(source_parameters=self.source_parameters)

		# Finally test that if we pass these kwargs into lenstronomy
		# we do not crash
		lens_model = LensModel(['SPEP'])
		light_model = LightModel(lm_list)
		n_pixels = 200
		image_model = ImageModel(
			data_class=ImageData(**data_configure_simple(numPix=n_pixels,
				deltaPix=metadata['pixel_width'])),
			psf_class=PSF(psf_type='GAUSSIAN',
				fwhm=0.1 * metadata['pixel_width']),
			lens_model_class=lens_model,source_model_class=light_model)
		lens_kwargs = [{'theta_E': 0.0, 'e1': 0., 'e2': 0., 'gamma': 0.,
			'center_x': 0, 'center_y': 0}]
		source_kwargs = [self.c.draw_source(catalog_i=catalog_i)[1][0]]

		image_model.image(kwargs_lens=lens_kwargs, kwargs_source=source_kwargs)


class COSMOSExcludeCatalogTests(COSMOSCatalogTests):

	def setUp(self):
		super().setUp()
		self.source_parameters['source_exclusion_list'] = [9,3]
		self.c = COSMOSExcludeCatalog(cosmology_parameters='planck18',
			source_parameters=self.source_parameters)

	def test_sample_indices(self):
		# Test the sampled indices respect the restriction we pass.
		# Sample alot to make sure we get the full range.
		n_galaxies = int(1e4)
		samples = self.c.sample_indices(n_galaxies)
		self.assertEqual(np.min(samples),0)
		self.assertEqual(np.max(samples),8)

		# Repeat the test with some cuts on apparent magnitude.
		# Only the first two entries meet this requirement
		new_sp = copy.deepcopy(self.source_parameters)
		new_sp['faintest_apparent_mag'] = 22
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		self.assertEqual(np.min(samples),0)
		self.assertEqual(np.max(samples),1)

		# Now do the same but with a size cut
		new_sp['faintest_apparent_mag'] = 22.5
		new_sp['minimum_size_in_pixels'] = 90
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0,1,7])

		# Test the redshift
		new_sp['max_z'] = 0.5
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0,7])

		# Test the minimum flux radius
		new_sp['min_flux_radius'] = 20
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0])


class COSMOSIncludeCatalogTests(COSMOSCatalogTests):

	def setUp(self):
		super().setUp()
		self.source_parameters['source_inclusion_list'] = [9,0,1,2,3,4]
		self.c = COSMOSIncludeCatalog(cosmology_parameters='planck18',
			source_parameters=self.source_parameters)

	def test_sample_indices(self):
		# Test the sampled indices respect the restriction we pass.
		# Sample alot to make sure we get the full range.
		n_galaxies = int(1e4)
		samples = self.c.sample_indices(n_galaxies)
		self.assertEqual(np.min(samples),0)
		self.assertEqual(np.max(samples),9)

		# Repeat the test with some cuts on apparent magnitude.
		# Only the first two entries meet this requirement
		new_sp = copy.deepcopy(self.source_parameters)
		new_sp['faintest_apparent_mag'] = 22
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		self.assertEqual(np.min(samples),0)
		self.assertEqual(np.max(samples),1)

		# Now do the same but with a size cut
		new_sp['faintest_apparent_mag'] = 22.5
		new_sp['minimum_size_in_pixels'] = 90
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0,1,3])

		# Test the redshift
		new_sp['max_z'] = 0.5
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0])

		# Test the minimum flux radius
		new_sp['min_flux_radius'] = 20
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0])
