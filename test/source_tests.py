import numpy as np
import unittest
import os
from manada.Sources.galaxy_catalog import GalaxyCatalog
from manada.Sources.cosmos import COSMOSCatalog, unfits
from manada.Sources.cosmos import HUBBLE_ACS_PIXEL_WIDTH
from manada.Utils.cosmology_utils import get_cosmology
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util.simulation_util import data_configure_simple
from lenstronomy.Data.psf import PSF
import scipy
import copy


class GalaxyCatalogTests(unittest.TestCase):

	def setUp(self):
		self.c = GalaxyCatalog(cosmology_parameters='planck18',
			source_parameters={})

	def test__len__(self):
		# Just test that the not implemented error is raised.
		with self.assertRaises(NotImplementedError):
			self.c.__len__()

	def test_update_parameters(self):
		# Check that the update parameter call updates the cosmology
		h = self.c.cosmo.h
		self.c.update_parameters('WMAP9')
		self.assertNotEqual(h,self.c.cosmo.h)

	def test_image_and_metadata(self):
		# Just test that the not implemented error is raised.
		catalog_i = 2
		with self.assertRaises(NotImplementedError):
			self.c.image_and_metadata(catalog_i)

	def test_iter_lightmodel_kwargs_samples(self):
		# Just test that the not implemented error is raised.
		n_galaxies = 10
		with self.assertRaises(NotImplementedError):
			for _ in self.c.iter_lightmodel_kwargs_samples(n_galaxies):
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

	def test_lightmodel_list_kwargs(self):
		# Just test that the not implemented error is raised.
		catalog_i = 2
		with self.assertRaises(NotImplementedError):
			self.c.lightmodel_list_kwargs(catalog_i)


class COSMOSCatalogTests(unittest.TestCase):

	def setUp(self):
		# Use a trimmed version of cosmo data for testing.
		self.test_cosmo_folder = (os.path.dirname(
			os.path.abspath(__file__))+'/test_data/cosmos/')
		self.source_parameters = {
			'smoothing_sigma':0, 'max_z':None, 'minimum_size_in_pixels':None,
			'min_apparent_mag':None
		}
		self.c = COSMOSCatalog(self.test_cosmo_folder,
			cosmology_parameters='planck18',
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
		os.remove(self.test_cosmo_folder+'manada_catalog.npy')
		for i in range(10):
			os.remove(self.test_cosmo_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(self.test_cosmo_folder+'npy_files')

	def test_check_parameterization(self):
		# Check that trying to initialize a class without the correct
		# parameters raises a value error.
		with self.assertRaises(ValueError):
			COSMOSCatalog(self.test_cosmo_folder,
				cosmology_parameters='planck18',
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
		# Check that the update parameter call updates the cosmology
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
		cs = COSMOSCatalog(self.test_cosmo_folder,
			cosmology_parameters='planck18',
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
		for lm_list, lm_kwargs in self.c.iter_lightmodel_kwargs_samples(
			n_galaxies):
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
		new_sp['min_apparent_mag'] = 22
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		self.assertEqual(np.min(samples),0)
		self.assertEqual(np.max(samples),1)

		# Now do the same but with a size cut
		new_sp['min_apparent_mag'] = 22.5
		new_sp['minimum_size_in_pixels'] = 90
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0,1,3,7])

		# Test the redshift
		new_sp['max_z'] = 0.5
		self.c.update_parameters(source_parameters=new_sp)
		samples = self.c.sample_indices(n_galaxies)
		np.testing.assert_equal(np.unique(samples),[0,7])

	def test_lightmodel_list_kwargs(self):
		# Test that the lightmodel kwargs returned are what we would
		# expect to pass into lenstronomy.
		catalog_i = 0
		image, metadata = self.c.image_and_metadata(catalog_i)

		# First don't change the redshift
		lm_list, lm_kwargs = self.c.lightmodel_list_kwargs(catalog_i,
			z_new=metadata['z'])
		lm_kwargs = lm_kwargs[0]
		self.assertEqual(lm_list[0],'INTERPOL')
		np.testing.assert_equal(lm_kwargs['image'],
			image/lm_kwargs['scale']**2)
		low_z_scale = lm_kwargs['scale']

		# Now change the redshift
		z_new = 1.0
		lm_list, lm_kwargs = self.c.lightmodel_list_kwargs(catalog_i,
			z_new=z_new)
		lm_kwargs = lm_kwargs[0]
		self.assertEqual(lm_list[0],'INTERPOL')
		np.testing.assert_equal(lm_kwargs['image'],
			image/metadata['pixel_width']**2)
		high_z_scale = lm_kwargs['scale']

		self.assertLess(high_z_scale,low_z_scale)

		# Grab the cosmo to compare with
		cosmo = get_cosmology('planck18')
		self.assertAlmostEqual(lm_kwargs['scale'],metadata['pixel_width']*
			cosmo.angularDiameterDistance(metadata['z'])/
			cosmo.angularDiameterDistance(z_new))

		# Finally test that if we pass these kwargs into a lenstronomy
		# Interpolation class we get the image we expect.
		lens_model = LensModel(['SPEP'])
		light_model = LightModel(['INTERPOL'])

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
		source_kwargs = [self.c.lightmodel_list_kwargs(catalog_i=catalog_i,
			z_new=metadata['z'])[1][0]]

		l_image = image_model.image(kwargs_lens=lens_kwargs,
			kwargs_source=source_kwargs)
		np.testing.assert_almost_equal(l_image,image)


def _check_lightmodel_kwargs(kwargs):
	assert isinstance(kwargs, dict)
	assert 'image' in kwargs
	img = kwargs['image']
	assert isinstance(img, np.ndarray)
	assert 0 < img[0, 0] < np.inf
