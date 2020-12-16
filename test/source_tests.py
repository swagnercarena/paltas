import numpy as np
import unittest
import os
from manada.Sources.galaxy_catalog import GalaxyCatalog
from manada.Sources.cosmos import COSMOSCatalog, unfits
from manada.Utils.cosmology_utils import get_cosmology
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util.simulation_util import data_configure_simple
from lenstronomy.Data.psf import PSF


class GalaxyCatalogTests(unittest.TestCase):

	def setUp(self):
		self.c = GalaxyCatalog(cosmology_parameters='planck18')

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

	def test_lightmodel_kwargs(self):
		# Just test that the not implemented error is raised.
		catalog_i = 2
		with self.assertRaises(NotImplementedError):
			self.c.lightmodel_kwargs(catalog_i)


class COSMOSCatalogTests(unittest.TestCase):

	def setUp(self):
		# Use a trimmed version of cosmo data for testing.
		self.test_cosmo_folder = (os.path.dirname(
			os.path.abspath(__file__))+'/test_data/cosmos/')
		self.c = COSMOSCatalog(self.test_cosmo_folder,
			cosmology_parameters='planck18')

		# Fix the seed so we don't have issues with randomness in tests
		np.random.seed(10)

		# Keys we'll want to check for
		self.rkeys = ['IDENT','RA','DEC','MAG','BAND','WEIGHT','GAL_FILENAME',
			'PSF_FILENAME','GAL_HDU','PSF_HDU','PIXEL_SCALE','NOISE_MEAN',
			'NOISE_VARIANCE','NOISE_FILENAME','stamp_flux']
		self.rfkeys = ['IDENT','mag_auto','flux_radius','zphot','sersicfit',
			'bulgefit','fit_status','fit_mad_s','fit_mad_b','fit_dvc_btt',
		'use_bulgefit','viable_sersic','hlr','flux']

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

	def test_image_and_metadata(self):
		catalog_i = 0
		image, metadata = self.c.image_and_metadata(catalog_i)
		np.testing.assert_equal(image.shape,(234, 233))
		self.assertEqual(metadata['mag_auto'],21.04064178466797)
		self.assertEqual(metadata['IDENT'],141190)

	def test_iter_lightmodel_kwargs_samples(self):
		# Just test that we get the expected kwargs
		n_galaxies = 10
		lm_keys_required = ['image','center_x','center_y','phi_G','scale']
		for lm_kwargs in self.c.iter_lightmodel_kwargs_samples(n_galaxies):
			self.assertTrue(all(elem in lm_kwargs.keys()
				for elem in lm_keys_required))

	def test_iter_image_and_metadata(self):
		# Just test that image data is returned and that it agrees with
		# the shape of the images.
		for image, metadata in self.c.iter_image_and_metadata():
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
		min_apparent_mag = 22
		samples = self.c.sample_indices(n_galaxies,
			min_apparent_mag=min_apparent_mag)
		self.assertEqual(np.min(samples),0)
		self.assertEqual(np.max(samples),1)

		# Now do the same but with a size cut
		minimum_size_in_pixels = 90
		min_apparent_mag = 22.5
		samples = self.c.sample_indices(n_galaxies,
			min_apparent_mag=min_apparent_mag,
			minimum_size_in_pixels=minimum_size_in_pixels)
		np.testing.assert_equal(np.unique(samples),[0,1,3,7])

	def test_lightmodel_kwargs(self):
		# Test that the lightmodel kwargs returned are what we would
		# expect to pass into lenstronomy.
		catalog_i = 0
		image, metadata = self.c.image_and_metadata(catalog_i)

		# First don't change the redshift
		lm_kwargs = self.c.lightmodel_kwargs(catalog_i,
			z_new=metadata['z'])
		np.testing.assert_equal(lm_kwargs['image'],
			image/lm_kwargs['scale']**2)

		# Now change the redshift
		z_new = 1.5
		lm_kwargs = self.c.lightmodel_kwargs(catalog_i,
			z_new=z_new)
		np.testing.assert_equal(lm_kwargs['image'],
			image/metadata['pixel_width']**2)

		# Grab the cosmo to compare with
		cosmo = get_cosmology('planck18')
		self.assertEqual(lm_kwargs['scale'],metadata['pixel_width']*
			cosmo.angularDiameterDistance(z_new)/
			cosmo.angularDiameterDistance(metadata['z']))

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
		source_kwargs = [self.c.lightmodel_kwargs(catalog_i=catalog_i,
			z_new=metadata['z'])]

		l_image = image_model.image(kwargs_lens=lens_kwargs,
			kwargs_source=source_kwargs)
		np.testing.assert_almost_equal(l_image,image)


def _check_lightmodel_kwargs(kwargs):
	assert isinstance(kwargs, dict)
	assert 'image' in kwargs
	img = kwargs['image']
	assert isinstance(img, np.ndarray)
	assert 0 < img[0, 0] < np.inf
