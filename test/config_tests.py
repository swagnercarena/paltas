import unittest
from paltas.Utils import cosmology_utils, hubble_utils
from paltas.Configs import config_handler
import paltas
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from scipy.signal import fftconvolve
from paltas.Sources.cosmos import COSMOSIncludeCatalog
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource

# Define the cosmos path
cosmos_folder = 'test_data/cosmos/'


class ConfigUtilsTests(unittest.TestCase):

	def setUp(self):
		# Fix the random seed to be able to have reliable tests
		np.random.seed(10)
		self.c = config_handler.ConfigHandler('test_data/config_dict.py')
		self.c_all = config_handler.ConfigHandler('test_data/config_dict_all.py')

	def test_init(self):
		# Test that the class was initialized correctly

		# Start by checking some of the global parameters
		self.assertEqual(self.c.kwargs_numerics['supersampling_factor'],1)
		self.assertEqual(self.c.numpix,64)
		self.assertFalse(self.c.do_drizzle)
		self.assertEqual(self.c.mag_cut,1.0)
		self.assertEqual(self.c.add_noise,True)

		# Now check some of the classes and make sure they match
		# our expectations from the config file.
		self.assertEqual(self.c.point_source_class,None)
		self.assertEqual(self.c.lens_light_class,None)
		self.assertTrue(isinstance(self.c.los_class,
			paltas.Substructure.los_dg19.LOSDG19))
		self.assertTrue(isinstance(self.c.main_deflector_class,
			paltas.MainDeflector.simple_deflectors.PEMDShear))
		self.assertTrue(isinstance(self.c.subhalo_class,
			paltas.Substructure.subhalos_dg19.SubhalosDG19))
		self.assertTrue(isinstance(self.c.source_class,
			paltas.Sources.cosmos.COSMOSCatalog))

		# Check the values that should be different for config_dict_all
		self.assertEqual(self.c_all.mag_cut,None)
		self.assertEqual(self.c_all.add_noise,False)
		self.assertTrue(isinstance(self.c_all.lens_light_class,
			paltas.Sources.sersic.SingleSersicSource))
		self.assertTrue(isinstance(self.c_all.point_source_class,
			paltas.PointSource.single_point_source.SinglePointSource))

	def test_get_lenstronomy_models_kwargs(self):
		# Test that the kwargs returned are consistent with what was provided
		# in the configuration file.
		kwargs_model, kwargs_params = self.c.get_lenstronomy_models_kwargs(
			new_sample=False)

		# First check the lens lists.
		self.assertGreater(kwargs_model['lens_model_list'].count('TNFW'),0)
		self.assertGreater(kwargs_model['lens_model_list'].count('NFW'),0)
		self.assertEqual(kwargs_model['lens_model_list'].count('INTERPOL'),7)
		self.assertEqual(kwargs_model['lens_model_list'].count('EPL_NUMBA'),1)
		self.assertEqual(kwargs_model['lens_model_list'].count('SHEAR'),1)
		self.assertEqual(len(kwargs_model['lens_model_list']),
			len(kwargs_params['kwargs_lens']))
		self.assertEqual(len(kwargs_model['lens_model_list']),
			len(kwargs_model['lens_redshift_list']))
		self.assertTrue(kwargs_model['multi_plane'])

		# Now check the source lists
		self.assertListEqual(kwargs_model['source_light_model_list'],['INTERPOL'])
		self.assertTrue(isinstance(
			kwargs_params['kwargs_source'][0]['image'],np.ndarray))
		self.assertListEqual(kwargs_model['source_redshift_list'],[1.5])
		self.assertEqual(kwargs_model['z_source_convention'],1.5)
		self.assertEqual(kwargs_model['z_source'],1.5)

		# Now check that there is no lens light or point source for this
		# config.
		self.assertEqual(len(kwargs_model['lens_light_model_list']),0)
		self.assertEqual(len(kwargs_params['kwargs_lens_light']),0)
		self.assertEqual(len(kwargs_model['point_source_model_list']),0)
		self.assertEqual(len(kwargs_params['kwargs_ps']),0)

		# Check that if new sample is specified, that the underlying sample
		# changes.
		kwargs_model_new, kwargs_params_new = (
			self.c.get_lenstronomy_models_kwargs(new_sample=True))
		self.assertFalse(kwargs_params['kwargs_lens'][-2]['theta_E'] ==
			kwargs_params_new['kwargs_lens'][-2]['theta_E'])

		# Finally check that the config with a point source fills in those
		# lists.
		kwargs_model, kwargs_params = self.c_all.get_lenstronomy_models_kwargs(
			new_sample=False)
		self.assertListEqual(kwargs_model['lens_light_model_list'],
			['SERSIC_ELLIPSE'])
		self.assertEqual(len(kwargs_params['kwargs_lens_light']),1)
		self.assertListEqual(kwargs_model['point_source_model_list'],
			['SOURCE_POSITION'])
		self.assertEqual(len(kwargs_params['kwargs_ps']),1)

		# Check that the multiplane triggers when you have los halos
		kwargs_model, kwargs_params = self.c_all.get_lenstronomy_models_kwargs()
		self.assertTrue(kwargs_model['multi_plane'])

	def test_get_metadata(self):
		# Test that the metadata matches the sample
		kwargs_model, kwargs_params = self.c.get_lenstronomy_models_kwargs(
			new_sample=False)
		metadata = self.c.get_metadata()

		# Check that the draw caused the phi and catalog_i to be written into
		# the metadata.
		self.assertTrue('source_parameters_phi' in metadata)
		self.assertTrue('source_parameters_catalog_i' in metadata)

		# Check the value of a few parameters
		self.assertEqual(metadata['main_deflector_parameters_theta_E'],
			kwargs_params['kwargs_lens'][-2]['theta_E'])
		self.assertEqual(metadata['main_deflector_parameters_gamma1'],
			kwargs_params['kwargs_lens'][-1]['gamma1'])
		self.assertEqual(metadata['cosmology_parameters_cosmology_name'],
			'planck18')
		self.assertEqual(metadata['subhalo_parameters_c_0'],18)
		self.assertEqual(metadata['los_parameters_c_0'],18)

	def test_get_sample_cosmology(self):
		# Just test that this gives the correct cosmology
		cosmo = self.c.get_sample_cosmology()
		cosmo_comp = cosmology_utils.get_cosmology('planck18')
		self.assertEqual(cosmo.H0,cosmo_comp.H0)

		# Check that the astropy version works
		cosmo = self.c.get_sample_cosmology(as_astropy=True)
		self.assertEqual(cosmo.H0,cosmo_comp.toAstropy().H0)

	def test__calculate_ps_metadata(self):
		# Check that the metadata is added as expected.
		# Get all of the lenstronomy parameters and models that we need
		kwargs_model, kwargs_params = self.c_all.get_lenstronomy_models_kwargs(
			new_sample=False)
		sample = self.c_all.get_current_sample()
		z_source = kwargs_model['source_redshift_list'][0]
		cosmo = cosmology_utils.get_cosmology(sample['cosmology_parameters'])
		lens_equation_params = sample['lens_equation_solver_parameters']
		lens_model = LensModel(kwargs_model['lens_model_list'],
			z_source=z_source,
			lens_redshift_list=kwargs_model['lens_redshift_list'],
			cosmo=cosmo.toAstropy(),multi_plane=kwargs_model['multi_plane'])
		point_source_model = PointSource(
			kwargs_model['point_source_model_list'],lensModel=lens_model,
			save_cache=True,kwargs_lens_eqn_solver=lens_equation_params)

		# Initialize empty metadata and populate it.
		metadata = {}
		self.c_all._calculate_ps_metadata(metadata,kwargs_params,
			point_source_model,lens_model)

		# Check that all the new metadata is there.
		pfix = 'point_source_parameters_'
		self.assertTrue(pfix+'num_images' in metadata.keys())
		self.assertTrue(pfix+'x_image_0' in metadata.keys())
		self.assertTrue(pfix+'y_image_1' in metadata.keys())
		self.assertTrue(pfix+'x_image_3' in metadata.keys())
		self.assertTrue(pfix+'y_image_3' in metadata.keys())

		# Check that image magnifications are written to metadata
		self.assertTrue(pfix+'magnification_0' in metadata.keys())
		self.assertTrue(pfix+'magnification_3' in metadata.keys())

		# Check that if num_images < 3, we get Nan for image 2 & image 3
		if(metadata[pfix+'num_images'] < 3):
			self.assertTrue(np.isnan(metadata[pfix+'x_image_3']))
			self.assertTrue(np.isnan(metadata[pfix+'y_image_2']))

		# Check that the time delay metadata is written
		self.assertTrue(pfix+'time_delay_0' in metadata.keys())
		self.assertTrue(pfix+'time_delay_3' in metadata.keys())
		self.assertTrue(pfix+'ddt' in metadata.keys())

		# Check that if kappa_ext is not defined, we get a ValueError
		del sample['point_source_parameters']['kappa_ext']
		with self.assertRaises(ValueError):
			self.c_all._calculate_ps_metadata(metadata,kwargs_params,
				point_source_model,lens_model)

	def test__draw_image_standard(self):
		# Test that drawing the standard image behaves as expected.

		# Grab the image we want to compare to
		orig_image,orig_meta = self.c.source_class.image_and_metadata(0)
		orig_image = orig_image[17:-17,:]
		orig_image = orig_image[:,1:]/2 + orig_image[:,:-1]/2
		orig_image = orig_image[:,16:-16]

		# Start with a simple configuration, a source with no lens.
		self.c.lens_light_class = None
		self.c.main_deflector_class = None
		self.c.los_class = None
		self.c.subhalo_class = None
		self.c.sample['source_parameters'] = {'z_source':0.19499999,
			'cosmos_folder':cosmos_folder,'max_z':None,
			'minimum_size_in_pixels':None,'faintest_apparent_mag':None,
			'smoothing_sigma':0.0,'random_rotation':False,
			'min_flux_radius':None,'output_ab_zeropoint':25.95,
			'center_x':0.0,'center_y':0.0,
			'source_inclusion_list':np.array([0])}
		self.c.source_class = COSMOSIncludeCatalog(
			cosmology_parameters='planck18',
			source_parameters=self.c.sample['source_parameters'])
		self.c.add_noise = False
		self.c.numpix = 200
		self.c.mag_cut = None
		self.c.sample['main_deflector_parameters']['z_lens'] = 0.0
		self.c.sample['detector_parameters'] = {
			'pixel_scale':orig_meta['pixel_width'],'ccd_gain':2.5,
			'read_noise':4.0,'magnitude_zero_point':25.0,'exposure_time':5400.0,
			'sky_brightness':22,'num_exposures':1,'background_noise':None}
		self.c.sample['psf_parameters'] = {'psf_type':'GAUSSIAN',
				'fwhm': 0.1*orig_meta['pixel_width']}

		# Draw our image. This should just be the source itself
		image, metadata = self.c._draw_image_standard(self.c.add_noise)

		# Check that the image is just the source
		np.testing.assert_almost_equal(image,orig_image)

		# Repeat the same test, but now with a really big psf and demanding
		# that no psf be added via the boolean input.
		self.c.sample['psf_parameters']['fwhm'] = 10
		apply_psf = False
		image, metadata = self.c._draw_image_standard(self.c.add_noise,apply_psf)
		np.testing.assert_almost_equal(image,orig_image)
		self.c.sample['psf_parameters']['fwhm'] =  0.1*orig_meta['pixel_width']

		# Now introduce rotations to the source and make sure that goes through
		self.c.sample['source_parameters']['random_rotation'] = True
		image, metadata = self.c._draw_image_standard(self.c.add_noise)
		np.testing.assert_array_less(np.ones(image.shape)*1e-10,
			np.abs(image-orig_image))

		# Check nothing weird happened to the metadata.
		self.assertListEqual(list(metadata.keys()),
			list(self.c.get_metadata().keys()))

		# Add noise
		self.c.sample['source_parameters']['random_rotation'] = False
		self.c.add_noise = True
		image, metadata = self.c._draw_image_standard(self.c.add_noise)
		np.testing.assert_array_less(np.ones(image.shape)*1e-10,
			np.abs(image-orig_image))

		# Check that the mag_cut works
		self.c.add_noise =False
		self.c.mag_cut = 1.2
		image, metadata = self.c._draw_image_standard(self.c.add_noise)
		self.assertTrue(image is None)
		self.assertTrue(metadata is None)

		# Now add a deflector and see if we get a ring
		self.c.add_noise = False
		self.c.sample['source_parameters']['z_source'] = 1.0
		self.c.sample['main_deflector_parameters'] =  {'M200':1e13,'z_lens': 0.5,
			'gamma': 2.0,'theta_E': 0.0,'e1':0.1,'e2':0.1,'center_x':0.02,
			'center_y':-0.03,'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0,
			'dec_0':0.0}
		self.c.main_deflector_class = PEMDShear(
			cosmology_parameters='planck18',
			main_deflector_parameters=self.c.sample['main_deflector_parameters'])
		# Update the main deflector after the fact to ensure that the values
		# are actually being updated in the draw_image call.
		self.c.sample['main_deflector_parameters']['theta_E'] = 1.0
		image, metadata = self.c._draw_image_standard(self.c.add_noise)

		# Check for magnification and check most light is not in center of image
		self.c.source_class.k_correct_image(orig_image,orig_meta['z'],
			self.c.sample['source_parameters']['z_source'])
		self.assertGreater(np.sum(image),np.sum(orig_image))
		self.assertGreater(np.mean(image[0:90,0:90]),
			np.mean(image[90:110,90:110]))

		# Now we'll turn off our main deflector but create a fake LOS
		# and subhalo class that gives the same profile.
		class FakeLOSClass():

			def update_parameters(self,*args,**kwargs):
				return

			def draw_los(self,*args,**kwargs):
				model_list = ['EPL_NUMBA','SHEAR']
				kwargs_list = [{'gamma': 2.0,'theta_E': 1.0,'e1':0.1,'e2':0.1,
					'center_x':0.02,'center_y':-0.03},
					{'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0, 'dec_0':0.0}]
				z_list = [0.5]*2
				return model_list,kwargs_list,z_list

			def calculate_average_alpha(self,*args,**kwargs):
				return ([],[],[])

		self.c.los_class = FakeLOSClass()
		self.c.main_deflector_class = None
		self.c.sample['los_parameters'] = {}
		los_image, metadata = self.c._draw_image_standard(self.c.add_noise)
		np.testing.assert_almost_equal(image,los_image)

		# Repeat the same excercise but for the subhalos
		class FakeSuhaloClass():

			def update_parameters(self,*args,**kwargs):
				return

			def draw_subhalos(self,*args,**kwargs):
				model_list = ['EPL_NUMBA','SHEAR']
				kwargs_list = [{'gamma': 2.0,'theta_E': 1.0,'e1':0.1,'e2':0.1,
					'center_x':0.02,'center_y':-0.03},
					{'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0, 'dec_0':0.0}]
				z_list = [0.5]*2
				return model_list,kwargs_list,z_list

		self.c.subhalo_class = FakeSuhaloClass()
		self.c.los_class = None
		self.c.sample['subhalo_parameters'] = {}
		sub_image, metadata = self.c._draw_image_standard(self.c.add_noise)
		np.testing.assert_almost_equal(image,sub_image)

		# Generate image with deflector & lens light
		self.c.sample['lens_light_parameters'] = {'z_source':0.5,'magnitude':20,
			'output_ab_zeropoint':25.95,'R_sersic':1.,'n_sersic':1.2,'e1':0.,
			'e2':0.,'center_x':0.0,'center_y':0.0}
		self.c.lens_light_class = SingleSersicSource(
			cosmology_parameters='planck18',
			source_parameters=self.c.sample['lens_light_parameters'])
		lens_light_image, metadata = self.c._draw_image_standard(self.c.add_noise)

		# Assert sum of center with lens light > sum of center orig_image
		self.assertTrue(np.sum(lens_light_image[90:110,90:110]) >
			np.sum(image[90:110,90:110]))

		# Add point source and validate output
		self.c.sample['point_source_parameters'] = {'x_point_source':0.001,
			'y_point_source':0.001,'magnitude':22,'output_ab_zeropoint':25.95,
			'compute_time_delays':False}
		self.c.point_source_class = SinglePointSource(
			self.c.sample['point_source_parameters'])

		image_ps, metadata = self.c._draw_image_standard(self.c.add_noise)

		# Check that more light is added to the image
		self.assertTrue(np.sum(image_ps) > np.sum(image))

	def test__draw_image_drizzle(self):
		# Test that drawing drizzled images works as expected.
		c_drizz = config_handler.ConfigHandler('test_data/config_dict_drizz.py')

		# Start with the simplest configuration, a source with nothing lensing
		# the source
		c_drizz.sample['source_parameters'] = {'z_source':0.19499999,
			'cosmos_folder':cosmos_folder,'max_z':None,
			'minimum_size_in_pixels':None,'faintest_apparent_mag':None,
			'smoothing_sigma':0.0,'random_rotation':False,
			'min_flux_radius':None,'output_ab_zeropoint':25.95,
			'center_x':0.0,'center_y':0.0,
			'source_inclusion_list':np.array([0])}
		c_drizz.los_class = None
		c_drizz.subhalo_class = None
		c_drizz.main_deflector_class = None
		c_drizz.source_class = COSMOSIncludeCatalog(
			cosmology_parameters='planck18',
			source_parameters=c_drizz.sample['source_parameters'])
		c_drizz.numpix = 200
		c_drizz.kwargs_numerics = {'supersampling_factor':1}
		c_drizz.mag_cut = None
		c_drizz.add_noise = False

		# Grab the image we want to compare to
		orig_image,orig_meta = c_drizz.source_class.image_and_metadata(0)
		orig_image = orig_image[17:-17,:]
		orig_image = orig_image[:,1:]/2 + orig_image[:,:-1]/2
		orig_image = orig_image[:,16:-16]

		# Create a fake sample from our sampler
		sim_pixel_width = orig_meta['pixel_width']
		c_drizz.sample['main_deflector_parameters'] = {'z_lens':0.0}
		c_drizz.sample['cosmology_parameters'] = {'cosmology_name': 'planck18'}
		c_drizz.sample['psf_parameters'] = {'psf_type':'GAUSSIAN',
			'fwhm': 0.1*orig_meta['pixel_width']}
		c_drizz.sample['detector_parameters'] = {'pixel_scale':sim_pixel_width,
			'ccd_gain':1.58,'read_noise':3.0,'magnitude_zero_point':25.127,
			'exposure_time':1380.0,'sky_brightness':15.83,'num_exposures':1,
			'background_noise':None}
		c_drizz.sample['drizzle_parameters']  = {
			'supersample_pixel_scale':sim_pixel_width,
			'output_pixel_scale':sim_pixel_width,'wcs_distortion':None,
			'offset_pattern':[(0,0),(0.0,0),(0.0,0.0),(-0.0,-0.0)]}

		# Draw our image. This should just be the source itself
		image, metadata = c_drizz._draw_image_drizzle()

		# Check that the image is just the source
		np.testing.assert_almost_equal(image,orig_image)

		# Check that the metadata is correct
		self.assertEqual(metadata['detector_parameters_pixel_scale'],
			sim_pixel_width)

		# Make the offset pattern more realistic and change the pixel widths
		c_drizz.sample['drizzle_parameters']['offset_pattern'] = [(0,0),(0.5,0),
			(0.0,0.5),(0.5,0.5)]
		c_drizz.sample['detector_parameters']['pixel_scale'] = 0.04
		c_drizz.sample['drizzle_parameters']['supersample_pixel_scale'] = 0.02
		c_drizz.sample['drizzle_parameters']['output_pixel_scale'] = 0.03
		c_drizz.numpix = 128

		# Check that the mag_cut works
		c_drizz.add_noise=False
		c_drizz.mag_cut = 1.2
		image, metadata = c_drizz._draw_image_drizzle()
		self.assertTrue(image is None)
		self.assertTrue(metadata is None)

		# Now add a deflector and see if we get a ring
		c_drizz.sample['source_parameters']['z_source'] = 1.0
		c_drizz.sample['main_deflector_parameters'] =  {'M200':1e13,
			'z_lens': 0.5,'gamma': 2.0,'theta_E': 1.0,'e1':0.1,'e2':0.1,
			'center_x':0.02,'center_y':-0.03,'gamma1':0.01,'gamma2':-0.02,
			'ra_0':0.0, 'dec_0':0.0}
		c_drizz.main_deflector_class = PEMDShear(cosmology_parameters='planck18',
			main_deflector_parameters=c_drizz.sample['main_deflector_parameters'])
		image, metadata = c_drizz._draw_image_drizzle()

		# Check for magnification and check most light is not in
		# center of image
		self.assertTupleEqual((170,170),image.shape)
		c_drizz.source_class.k_correct_image(orig_image,orig_meta['z'],
			c_drizz.sample['source_parameters']['z_source'])
		self.assertGreater(np.sum(image),np.sum(orig_image))
		self.assertGreater(np.mean(image[0:80,0:80]),
			np.mean(image[80:90,80:90]))

		# Check that setting the noise flag returns a noisy image
		c_drizz.add_noise = True
		los_image_noise, metadata = c_drizz._draw_image_drizzle()

		np.testing.assert_array_less(np.ones(image.shape)*1e-10,
			np.abs(image-los_image_noise))

	def test__draw_image_drizzle_psf(self):
		# Test the pixel psf behaves identically to using fftconvolve
		# Setup a fairly basic situation with a source at redshift 1.0 an a
		# massive main deflector at redshift 0.5.
		# Test that drawing drizzled images works as expected.
		c_drizz = config_handler.ConfigHandler('test_data/config_dict_drizz.py')

		# Start with the simplest configuration, a source with nothing lensing
		# the source
		c_drizz.sample['source_parameters'] = {'z_source':1.0,
			'cosmos_folder':cosmos_folder,'max_z':None,
			'minimum_size_in_pixels':None,'faintest_apparent_mag':None,
			'smoothing_sigma':0.0,'random_rotation':False,
			'min_flux_radius':None,'output_ab_zeropoint':25.95,'center_x':0.0,
			'center_y':0.0,'source_inclusion_list':np.array([0])}
		c_drizz.los_class = None
		c_drizz.subhalo_class = None
		c_drizz.sample['main_deflector_parameters'] = {'M200':1e13,
			'z_lens': 0.5,'gamma': 2.0,'theta_E': 1.0,'e1':0.1,'e2':0.1,
			'center_x':0.02,'center_y':-0.03,'gamma1':0.01,'gamma2':-0.02,
			'ra_0':0.0, 'dec_0':0.0}
		c_drizz.main_deflector_class = PEMDShear(
			cosmology_parameters='planck18',
			main_deflector_parameters=c_drizz.sample['main_deflector_parameters'])
		c_drizz.source_class = COSMOSIncludeCatalog(
			cosmology_parameters='planck18',
			source_parameters=c_drizz.sample['source_parameters'])
		c_drizz.numpix = 128
		c_drizz.kwargs_numerics = {'supersampling_factor':1}
		c_drizz.mag_cut = None
		c_drizz.add_noise = False

		# Create a fake sample from our sampler
		sim_pixel_width = 0.04
		c_drizz.sample['cosmology_parameters'] = {'cosmology_name': 'planck18'}
		c_drizz.sample['psf_parameters'] = {'psf_type':'NONE'}
		c_drizz.sample['detector_parameters'] = {'pixel_scale':sim_pixel_width,
			'ccd_gain':1.58,'read_noise':3.0,'magnitude_zero_point':25.127,
			'exposure_time':1380.0,'sky_brightness':15.83,'num_exposures':1,
			'background_noise':None}
		c_drizz.sample['drizzle_parameters']  = {
			'supersample_pixel_scale':sim_pixel_width,
			'output_pixel_scale':sim_pixel_width,'wcs_distortion':None,
			'offset_pattern':[(0,0),(0.0,0),(0.0,0.0),(-0.0,-0.0)],
			'psf_supersample_factor':1}

		# Draw our image. This should just be the lensed source without
		# noise and without a psf. This will be our supersamled image.
		image, metadata = c_drizz._draw_image_drizzle()
		image_degrade = hubble_utils.degrade_image(image,2)

		# Now generate a pixel level psf that isn't supersampled.
		psf_pixel = np.zeros((63,63))
		x,y = np.meshgrid(np.arange(63),np.arange(63),indexing='ij')
		psf_pixel[x,y] = np.exp(-((x-31)**2+(y-31)**2))
		psf_pixel /= np.sum(psf_pixel)
		c_drizz.sample['psf_parameters'] = {'psf_type':'PIXEL',
			'kernel_point_source': psf_pixel,
			'point_source_supersampling_factor':1}

		# Now generate the image again in the degraded resolution
		c_drizz.kwargs_numerics = {'supersampling_factor':2}
		c_drizz.numpix = 64
		c_drizz.sample['detector_parameters']['pixel_scale'] = sim_pixel_width*2
		c_drizz.sample['drizzle_parameters']['output_pixel_scale'] = (
			sim_pixel_width*2)
		image_degrade_psf, metadata = c_drizz._draw_image_drizzle()

		# Compare to the scipy image
		scipy_image = fftconvolve(image_degrade,psf_pixel,mode='same')

		np.testing.assert_almost_equal(scipy_image,image_degrade_psf)

		# Now repeat this process but doing the psf convolution at the
		# supersampling scale.
		c_drizz.sample['psf_parameters']['point_source_supersampling_factor'] = 2
		c_drizz.sample['drizzle_parameters']['psf_supersample_factor'] = 2
		image_degrade_psf, metadata = c_drizz._draw_image_drizzle()
		scipy_image = hubble_utils.degrade_image(
			fftconvolve(image,psf_pixel,mode='same'),2)
		np.testing.assert_almost_equal(scipy_image,image_degrade_psf,
			decimal=6)

		# Make sure the sample detector_parameters weren't changed in place.
		self.assertEqual(c_drizz.sample['detector_parameters']['pixel_scale'],
			sim_pixel_width*2)
		self.assertEqual(metadata['detector_parameters_pixel_scale'],
			sim_pixel_width*2)

		# Now just make sure we can raise some errors. First an error
		# if no point_source_supersampling_factor was specified.
		with self.assertRaises(ValueError):
			c_drizz.sample['psf_parameters'] = {'psf_type':'PIXEL',
				'kernel_point_source': psf_pixel}
			image_degrade_psf, meta_values = c_drizz._draw_image_drizzle()

		# Next an error if it doesn't equal the psf_supersample_factor
		with self.assertRaises(ValueError):
			c_drizz.sample['psf_parameters'] = {'psf_type':'PIXEL',
				'kernel_point_source': psf_pixel,
				'point_source_supersampling_factor':1}
			image_degrade_psf, meta_values = c_drizz._draw_image_drizzle()

		# Next an error if the psf_supersample_factor is larger than the scaling
		# provided by the drizzle parameters.
		with self.assertRaises(ValueError):
			c_drizz.sample['psf_parameters'][
				'point_source_supersampling_factor'] = 4
			c_drizz.sample['psf_parameters'] = {'psf_type':'PIXEL',
				'kernel_point_source': psf_pixel,
				'point_source_supersampling_factor':4}
			image_degrade_psf, meta_values = c_drizz._draw_image_drizzle()

	def test_draw_image(self):
		# Just test that nothing crashes.
		c_drizz = config_handler.ConfigHandler('test_data/config_dict_drizz.py')
		_,_ = c_drizz.draw_image(new_sample=True)
		image,_ = self.c.draw_image(new_sample=True)

		# Also make sure the mask radius was applied
		self.assertEqual(np.sum(image[len(image)//2-2:len(image)//2+2,
			len(image)//2-2:len(image)//2+2]),0.0)

		# Make sure that if the sample has crazy parameters, those carry
		# thourgh.
		self.c.sample['main_deflector_parameters']['theta_E'] = 0.1
		self.c.mag_cut = None
		image_small,metadata = self.c.draw_image(new_sample=False)
		self.assertEqual(metadata['main_deflector_parameters_theta_E'],0.1)
		self.assertLess(np.sum(image_small),np.sum(image))
