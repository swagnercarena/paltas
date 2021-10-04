import unittest
from manada.Utils import power_law, cosmology_utils, hubble_utils
from manada.Utils import lenstronomy_utils
from scipy.integrate import quad
import numpy as np
from colossus.cosmology import cosmology
from astropy import units as u
from astropy.wcs import wcs
from lenstronomy.Data.psf import PSF
from lenstronomy.SimulationAPI.data_api import DataAPI
from scipy.signal import fftconvolve


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


class HubbleUtilsTests(unittest.TestCase):

	def test_offset_wcs(self):
		# Check that providing a valid WCS and offset returns the expected
		# positions from all_pix2world.
		npix = 128
		pixel_width = 0.04/3600
		wcs_input_dict = {
			'CTYPE1': 'RA-TAN',
			'CTYPE2': 'DEC-TAN',

			'CUNIT1': 'deg',
			'CUNIT2': 'deg',

			'CDELT1': pixel_width,
			'CDELT2': pixel_width,

			'CRPIX1': npix/2,
			'CRPIX2': npix/2,

			# Just some standard reference location
			'CRVAL1': 337.5202808,
			'CRVAL2': -20.833333059999998,

			'NAXIS1': npix,
			'NAXIS2': npix
		}
		w = wcs.WCS(wcs_input_dict)

		# First start with no offset
		w_off = hubble_utils.offset_wcs(w,(0,0))
		x,y = np.meshgrid(np.arange(npix),np.arange(npix))
		np.testing.assert_almost_equal(w.all_pix2world(x,y,0),
			w_off.all_pix2world(x,y,0))

		# Now add an offset
		w_off = hubble_utils.offset_wcs(w,(0.5,0))
		x,y = np.meshgrid(np.arange(npix),np.arange(npix))
		np.testing.assert_almost_equal(w.all_pix2world(x-0.5,y,0),
			w_off.all_pix2world(x,y,0))

		# Now add another offset
		w_off = hubble_utils.offset_wcs(w,(0.5,0.7))
		x,y = np.meshgrid(np.arange(npix),np.arange(npix))
		np.testing.assert_almost_equal(w.all_pix2world(x-0.5,y-0.7,0),
			w_off.all_pix2world(x,y,0))

	def test_distort_image(self):
		# Check that the offset images returned match our expectations
		img_high_res = np.zeros((256,256))
		for i in range(len(img_high_res)):
			img_high_res[i] += i
		for j in range(img_high_res.shape[1]):
			img_high_res[:,j] += j
		pixel_width = 0.02/3600
		npix = 256
		wcs_hr_dict = {
			'CTYPE1': 'RA-TAN',
			'CTYPE2': 'DEC-TAN',
			'CUNIT1': 'deg',
			'CUNIT2': 'deg',
			'CDELT1': pixel_width,
			'CDELT2': pixel_width,
			'CRPIX1': npix/2,
			'CRPIX2': npix/2,
			'CRVAL1': 90,
			'CRVAL2': -20,
			'NAXIS1': npix,
			'NAXIS2': npix
		}
		w_hr = wcs.WCS(wcs_hr_dict)

		pixel_width = 0.04/3600
		npix = 128
		wcs_lr_dict = {
			'CTYPE1': 'RA-TAN',
			'CTYPE2': 'DEC-TAN',
			'CUNIT1': 'deg',
			'CUNIT2': 'deg',
			'CDELT1': pixel_width,
			'CDELT2': pixel_width,
			'CRPIX1': npix/2,
			'CRPIX2': npix/2,
			'CRVAL1': 90,
			'CRVAL2': -20,
			'NAXIS1': npix,
			'NAXIS2': npix
		}
		w_lr = wcs.WCS(wcs_lr_dict)

		offset_pattern = [(0,0),(-0.5,0.0),(0,-0.5)]
		psf_supersample_factor = 1
		img_dither_array = hubble_utils.distort_image(img_high_res,w_hr,w_lr,
			offset_pattern,psf_supersample_factor)

		# Test the no offset image.
		test_image = np.zeros((npix,npix))
		for i in range(len(test_image)):
			for j in range(test_image.shape[1]):
				test_image[i,j] = np.sum(img_high_res[2*i:2*i+2,2*j:2*j+2])
		np.testing.assert_almost_equal(test_image,img_dither_array[0])
		self.assertAlmostEqual(np.sum(img_high_res),
			np.sum(img_dither_array[0]))

		# Test the two images with offsets
		test_image = np.zeros((npix,npix))
		for i in range(len(test_image)):
			for j in range(test_image.shape[1]):
				test_image[i,j] = np.mean(img_high_res[2*i+1:2*i+3,
					2*j:2*j+2])*4
		np.testing.assert_almost_equal(test_image,img_dither_array[1])

		test_image = np.zeros((npix,npix))
		for i in range(len(test_image)):
			for j in range(test_image.shape[1]):
				test_image[i,j] = np.mean(img_high_res[2*i:2*i+2,2*j+1:
					2*j+3])*4
		np.testing.assert_almost_equal(test_image,img_dither_array[2])

		# Now test for a larger psf_supersample_factor
		psf_supersample_factor = 2
		offset_pattern = [(0,0),(-0.5,0.0),(0,-0.5)]
		img_dither_array = hubble_utils.distort_image(img_high_res,w_hr,w_lr,
			offset_pattern,psf_supersample_factor)
		np.testing.assert_almost_equal(img_dither_array[0],img_high_res)
		np.testing.assert_almost_equal(img_dither_array[1,:-1,:],
			img_high_res[1:,:])
		np.testing.assert_almost_equal(img_dither_array[2,:,:-1],
			img_high_res[:,1:])

	def test_generate_downsampled_wcs(self):
		# Check that the downsampled WCS maps as expected to the higher
		# res wcs.
		high_res_shape = (256,256)
		high_res_pixel_scale = 0.02
		low_res_pixel_scale = 0.04
		wcs_distortion = None

		w_lr = hubble_utils.generate_downsampled_wcs(high_res_shape,
			high_res_pixel_scale,low_res_pixel_scale,wcs_distortion)
		w_hr = hubble_utils.generate_downsampled_wcs(high_res_shape,
			high_res_pixel_scale,high_res_pixel_scale,wcs_distortion)

		x,y = np.meshgrid(np.arange(high_res_shape[0]),
			np.arange(high_res_shape[1]),indexing='ij')
		np.testing.assert_almost_equal(w_lr.all_pix2world(x/2,y/2,1),
			w_hr.all_pix2world(x,y,1))

		# Try for another resolution.
		low_res_pixel_scale = 0.03
		w_mr = hubble_utils.generate_downsampled_wcs(high_res_shape,
			high_res_pixel_scale,low_res_pixel_scale,wcs_distortion)
		np.testing.assert_almost_equal(w_mr.all_pix2world(2*x/3,2*y/3,1),
			w_hr.all_pix2world(x,y,1),decimal=4)
		np.testing.assert_almost_equal(w_lr.all_pix2world(3*x/4,3*y/4,1),
			w_mr.all_pix2world(x,y,1),decimal=4)

		# Check that the pixel shapes are integers
		for w in [w_lr,w_hr,w_mr]:
			self.assertTrue(type(w.pixel_shape[0]) is int)
			self.assertTrue(type(w.pixel_shape[1]) is int)

	def test_degrade_image(self):
		# Test that the degraded image gives the output we'd expect.
		fake_image = np.array([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]],
			dtype=np.float)

		# Start by degrading by a factor of 2
		degrade_factor = 2
		degrade_2 = hubble_utils.degrade_image(fake_image,degrade_factor)
		np.testing.assert_almost_equal(degrade_2,np.array(
			[[10,18],[42,50]]))

		# Now degrade by a factor of 4
		degrade_factor = 4
		degrade_2 = hubble_utils.degrade_image(fake_image,degrade_factor)
		np.testing.assert_almost_equal(degrade_2,np.array(
			[[120]]))

	def test_hubblify(self):
		# It's a little hard to test drizzle in detail here, but we can
		# do some surface level tests. We can make sure flux is convserved,
		# that it returns the image when all the resolutions are the same,
		# and that nothing goes crazy with the 0.02,0.04,0.03 configuration
		# we plan to use.

		# First generate an image with all the signal in the center
		numpix = 256
		img_high_res = np.ones((numpix,numpix))
		x,y = np.meshgrid(np.arange(img_high_res.shape[0]),
			np.arange(img_high_res.shape[1]),indexing='ij')
		r = np.sqrt((x-numpix/2)**2+((y-numpix/2)*2)**2)
		img_high_res *= (r<=35.5)

		# Now drizzle the image with no change in the pixel scale at any
		# step
		high_res_pixel_scale = 0.04
		detector_pixel_scale = 0.04
		drizzle_pixel_scale = 0.04

		# Create a noise model and psf model that do nothing
		def noise_model(image):
			return 0

		def psf_model(image):
			return image

		# Check that with integer offsets the input image is returned
		offset_pattern = [(0,0),(5.0,0),(10.0,0.0),(15.0,15.0)]
		img_drizz = hubble_utils.hubblify(img_high_res,high_res_pixel_scale,
			detector_pixel_scale,drizzle_pixel_scale,noise_model,psf_model,
			offset_pattern)
		# np.testing.assert_almost_equal(img_drizz,img_high_res*4)

		# Now repeat the test above with half-integer offsets.
		offset_pattern = [(0,0),(0.5,0),(0.0,0.5),(0.5,0.5)]
		img_drizz = hubble_utils.hubblify(img_high_res,high_res_pixel_scale,
			detector_pixel_scale,drizzle_pixel_scale,noise_model,psf_model,
			offset_pattern)
		# Check that flux is conserved and kept roughly within the correct
		# area.
		self.assertAlmostEqual(np.sum(img_drizz),np.sum(img_high_res*4))
		self.assertAlmostEqual(np.sum(img_drizz[r<=38]),np.sum(img_drizz))

		# Now let's change the resolutions a bit and make sure that doesn't
		# break everything
		high_res_pixel_scale = 0.02
		detector_pixel_scale = 0.04
		drizzle_pixel_scale = 0.03
		img_drizz = hubble_utils.hubblify(img_high_res,high_res_pixel_scale,
			detector_pixel_scale,drizzle_pixel_scale,noise_model,psf_model,
			offset_pattern)
		# Same basic checks
		self.assertAlmostEqual(np.sum(img_drizz),np.sum(img_high_res*4))
		x,y = np.meshgrid(np.arange(img_drizz.shape[0]),
			np.arange(img_drizz.shape[1]),indexing='ij')
		r = np.sqrt((x-85)**2+((y-85)*2)**2)
		self.assertAlmostEqual(np.sum(img_drizz[r<=27]),np.sum(img_drizz))

		# Okay now let's test that we get correlated noise.
		img_high_res *= 0

		class NM():
			def __init__(self):
				self.pos = 30

			def noise_model(self,image):
				noise = np.zeros(image.shape)
				noise[self.pos,self.pos] = 20
				self.pos += 30
				return noise

		offset_pattern = [(0,0),(0.5,0),(0.0,0.5),(-0.5,-0.5)]
		img_drizz = hubble_utils.hubblify(img_high_res,high_res_pixel_scale,
			detector_pixel_scale,drizzle_pixel_scale,NM().noise_model,
			psf_model,offset_pattern)
		# Check that at each of the four positions only a part of the
		# noise is present, but that all the noise is present in the
		# image
		for i in range(1,5):
			self.assertTrue(img_drizz[40*i,40*i]>0 and img_drizz[40*i,40*i]<20)
		self.assertAlmostEqual(np.sum(img_drizz),80)

		# Now let's check that the psf blurs the image as we want. Simulate
		# the use of the lenstronomy psf functions
		# Image is a point source
		img_high_res[128,128] = 20

		# Our PSF is just an exponentially decaying line. This way we
		# can make sure that the psf orientation is preserved here.
		psf_pixel = np.zeros((129,129))
		psf_pixel[64,:] = np.exp(-(np.arange(129)-64)**2/100)
		psf_parameters = {'psf_type':'PIXEL',
			'kernel_point_source': psf_pixel}
		kwargs_detector = {'pixel_scale':detector_pixel_scale,
			'ccd_gain':2.5,'read_noise':4.0,'magnitude_zero_point':25.0,
			'exposure_time':5400.0,'sky_brightness':22,'num_exposures':1,
			'background_noise':None}
		kwargs_numerics = {'supersampling_factor':1,
			'supersampling_convolution':True,
			'point_source_supersampling_factor':1}

		# Make the objects we need to interact with the lenstronomy api.
		psf_model = PSF(**psf_parameters)
		data_class = DataAPI(numpix=numpix//2,**kwargs_detector).data_class

		# Use the lenstronomy helper class.
		psf_helper = lenstronomy_utils.PSFHelper(data_class,psf_model,
			kwargs_numerics)

		# Use the psf model in the image generation.
		img_drizz = hubble_utils.hubblify(img_high_res,high_res_pixel_scale,
			detector_pixel_scale,drizzle_pixel_scale,noise_model,
			psf_helper.psf_model,offset_pattern)

		# Check that all the signal is still there
		self.assertAlmostEqual(np.sum(img_drizz),np.sum(img_high_res*4))

		# Check that the signal is contained within the correct strip
		self.assertAlmostEqual(np.sum(img_drizz[83:88]),np.sum(img_drizz))
		self.assertGreater(np.sum(img_drizz),np.sum(img_drizz[:,83:88]))
		self.assertLess(np.sum(img_drizz[:,85]),
			80/np.sum(np.exp(-(np.arange(129)-64)**2/100)))

		# Repeat the psf test, but now use the psf_supersample_factor
		# specification.
		psf_supersample_factor = 2
		psf_pixel[64,:] = np.exp(-(np.arange(129)-64)**2/100/
			psf_supersample_factor**2)
		psf_parameters = {'psf_type':'PIXEL',
			'kernel_point_source': psf_pixel}
		kwargs_detector['pixel_scale'] = (
			detector_pixel_scale/psf_supersample_factor)
		psf_model = PSF(**psf_parameters)
		data_class = DataAPI(numpix=numpix,**kwargs_detector).data_class
		psf_helper = lenstronomy_utils.PSFHelper(data_class,psf_model,
			kwargs_numerics)
		img_drizz_ss = hubble_utils.hubblify(img_high_res,high_res_pixel_scale,
			detector_pixel_scale,drizzle_pixel_scale,noise_model,
			psf_helper.psf_model,offset_pattern,
			psf_supersample_factor=psf_supersample_factor)

		# Check that this supersampling does not change the image much but
		# it does change the image some.
		np.testing.assert_array_less(np.zeros(img_drizz.shape),
			np.abs(img_drizz-img_drizz_ss))
		np.testing.assert_almost_equal(img_drizz_ss,img_drizz,
			decimal=2)


class LenstronomyUtilsTests(unittest.TestCase):

	def test_psf_model(self):
		# Test that the psf model class behaves well with a wide set of
		# psfs.
		# Create a Gaussian pixel map
		psf_pixel = np.zeros((63,63))
		x,y = np.meshgrid(np.arange(63),np.arange(63),indexing='ij')
		psf_pixel[x,y] = np.exp(-((x-31)**2+(y-31)**2))
		psf_pixel /= np.sum(psf_pixel)

		# Generate a ring image.
		image = np.zeros((64,64))
		x,y = np.meshgrid(np.arange(64),np.arange(64),indexing='ij')
		image[x,y] = np.exp(-(np.sqrt((x-31)**2+(y-31)**2)-15)**2/4)

		# Setup the lenstronomy objects we want to compare to.
		detector_pixel_scale = 0.04
		numpix = 64
		psf_parameters = {'psf_type':'PIXEL',
			'kernel_point_source': psf_pixel}
		kwargs_detector = {'pixel_scale':detector_pixel_scale,
			'ccd_gain':2.5,'read_noise':4.0,'magnitude_zero_point':25.0,
			'exposure_time':5400.0,'sky_brightness':22,'num_exposures':1,
			'background_noise':None}
		kwargs_numerics = {'supersampling_factor':1,
			'supersampling_convolution':False,
			'point_source_supersampling_factor':1}
		psf_model = PSF(**psf_parameters)
		data_class = DataAPI(numpix=numpix,**kwargs_detector).data_class
		psf_helper = lenstronomy_utils.PSFHelper(data_class,psf_model,
			kwargs_numerics)

		# Convolve with lenstronomy and with scipy
		helper_image = psf_helper.psf_model(image)
		scipy_image = fftconvolve(image,psf_pixel,mode='same')

		# Compare the outputs
		np.testing.assert_almost_equal(helper_image,scipy_image)

		# Check that specify no psf works as well
		psf_parameters = {'psf_type':'NONE'}
		psf_model = PSF(**psf_parameters)
		psf_helper = lenstronomy_utils.PSFHelper(data_class,psf_model,
			kwargs_numerics)

		# Make sure the helper image is not convolved
		helper_image = psf_helper.psf_model(image)
		np.testing.assert_almost_equal(helper_image,image)

		# # Now do the same but with supersampling
		# numpix = 32
		# psf_parameters['point_source_supersampling_factor'] = 2
		# kwargs_numerics = {'supersampling_factor':2,
		# 	'supersampling_convolution':True,
		# 	'point_source_supersampling_factor':2}
		# psf_model = PSF(**psf_parameters)
		# data_class = DataAPI(numpix=numpix,**kwargs_detector).data_class
		# psf_helper = lenstronomy_utils.PSFHelper(data_class,psf_model,
		# 	kwargs_numerics)
		# helper_image = psf_helper.psf_model(image)
		# scipy_image = fftconvolve(image,psf_pixel,mode='same')

		# # Compare the outputs
		# np.testing.assert_almost_equal(helper_image,scipy_image)
