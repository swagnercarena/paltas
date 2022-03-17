import numpy as np
import pandas as pd
import unittest
import sys, glob, copy, os
from paltas import generate
from scipy.signal import fftconvolve
from paltas.Sources.cosmos import COSMOSIncludeCatalog
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Utils import hubble_utils

# Define the cosmos path
cosmos_folder = 'test_data/cosmos/'


class GenerateTests(unittest.TestCase):

	def setUp(self):
		# Set the random seed so we don't run into trouble
		np.random.seed(20)

	def test_parse_args(self):
		# Check that the argument parser works as intended
		# We have to modify the sys.argv input which is bad practice
		# outside of a test
		old_sys = copy.deepcopy(sys.argv)
		sys.argv = ['test','config_dict.py','save_folder',
			'--n','1000']
		args = generate.parse_args()
		self.assertEqual(args.config_dict,'config_dict.py')
		self.assertEqual(args.save_folder,'save_folder')
		self.assertEqual(args.n,1000)
		sys.argv = old_sys

	def test_draw_image(self):

		# Start with the simplest configuration, a source with nothing lensing
		# the source
		source_parameters = {'z_source':0.19499999,
			'cosmos_folder':cosmos_folder,'max_z':None,
			'minimum_size_in_pixels':None,'faintest_apparent_mag':None,
			'smoothing_sigma':0.0,'random_rotation':False,
			'min_flux_radius':None,'output_ab_zeropoint':25.95,
			'center_x':0.0,'center_y':0.0,
			'source_inclusion_list':np.array([0])}
		los_class = None
		subhalo_class = None
		main_deflector_class = None
		source_class = COSMOSIncludeCatalog(cosmology_parameters='planck18',
			source_parameters=source_parameters)
		# Set the source redshift to the redshift of the catalog image to avoid
		# any rescaling
		source_parameters['z_source'] = source_class.catalog['z'][0]
		numpix = 200
		multi_plane = False
		kwargs_numerics = {'supersampling_factor':1}
		mag_cut = None
		add_noise = False

		# Grab the image we want to compare to
		orig_image,orig_meta = source_class.image_and_metadata(0)
		orig_image = orig_image[17:-17,:]
		orig_image = orig_image[:,1:]/2 + orig_image[:,:-1]/2
		orig_image = orig_image[:,16:-16]

		# Create a fake sample from our sampler
		sample = {
			'main_deflector_parameters':{'z_lens':0.0},
			'source_parameters':source_parameters,
			'cosmology_parameters':'planck18',
			'psf_parameters':{'psf_type':'GAUSSIAN',
				'fwhm': 0.1*orig_meta['pixel_width']},
			'detector_parameters':{'pixel_scale':orig_meta['pixel_width'],
				'ccd_gain':2.5,'read_noise':4.0,'magnitude_zero_point':25.0,
				'exposure_time':5400.0,'sky_brightness':22,'num_exposures':1,
				'background_noise':None}
		}

		# Draw our image. This should just be the source itself
		image, meta_values = generate.draw_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None, None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)

		# Check that the image is just the source
		np.testing.assert_almost_equal(image,orig_image)

		# Repeat the same test, but now with a really big psf and demanding
		# that no psf be added via the boolean input.
		sample['psf_parameters']['fwhm'] = 10
		apply_psf = False
		image, meta_values = generate.draw_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise,
			apply_psf=apply_psf)
		np.testing.assert_almost_equal(image,orig_image)
		sample['psf_parameters']['fwhm'] =  0.1*orig_meta['pixel_width']

		# Now introduce rotations to the source and make sure that
		# goes through
		source_parameters['random_rotation'] = True
		image, meta_values = generate.draw_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		np.testing.assert_array_less(np.ones(image.shape)*1e-10,
			np.abs(image-orig_image))

		# Same for noise
		source_parameters['random_rotation'] = False
		add_noise = True
		image, meta_values = generate.draw_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		np.testing.assert_array_less(np.ones(image.shape)*1e-10,
			np.abs(image-orig_image))

		# Check that the mag_cut works
		add_noise=False
		mag_cut = 1.2
		image, meta_values = generate.draw_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		self.assertTrue(image is None)
		self.assertTrue(meta_values is None)

		# Now add a deflector and see if we get a ring
		add_noise = False
		source_parameters['z_source'] = 1.0
		main_deflector_parameters =  {'M200':1e13,'z_lens': 0.5,'gamma': 2.0,
			'theta_E': 0.0,'e1':0.1,'e2':0.1,'center_x':0.02,'center_y':-0.03,
			'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0, 'dec_0':0.0}
		sample['main_deflector_parameters'] = main_deflector_parameters
		main_deflector_class = PEMDShear(cosmology_parameters='planck18',
			main_deflector_parameters=main_deflector_parameters)
		# Update the main deflector after the fact to ensure that the values
		# are actually being updated in the draw_image call.
		main_deflector_parameters['theta_E'] = 1.0
		image, meta_values = generate.draw_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)

		# Check for magnification and check most light is not in
		# center of image
		self.assertGreater(np.sum(image),np.sum(orig_image))
		self.assertGreater(np.mean(image[0:90,0:90]),
			np.mean(image[90:110,90:110]))

		# Now we'll turn off our main deflector but create a fake LOS
		# and subhalo class that gives the same profile.
		class FakeLOSClass():

			def update_parameters(self,*args,**kwargs):
				return

			def draw_los(self,*args,**kwargs):
				model_list = ['EPL','SHEAR']
				kwargs_list = [{'gamma': 2.0,'theta_E': 1.0,'e1':0.1,'e2':0.1,
					'center_x':0.02,'center_y':-0.03},
					{'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0, 'dec_0':0.0}]
				z_list = [main_deflector_parameters['z_lens']]*2
				return model_list,kwargs_list,z_list

			def calculate_average_alpha(self,*args,**kwargs):
				return ([],[],[])

		los_class = FakeLOSClass()
		sample['los_parameters'] = None
		multi_plane = True
		los_image, meta_values = generate.draw_image(sample,los_class,
			subhalo_class,None,source_class,None,None,numpix,multi_plane,
			kwargs_numerics,mag_cut,add_noise)
		np.testing.assert_almost_equal(image,los_image)

		# Repeat the same excercise but for the subhalos
		class FakeSuhaloClass():

			def update_parameters(self,*args,**kwargs):
				return

			def draw_subhalos(self,*args,**kwargs):
				model_list = ['EPL','SHEAR']
				kwargs_list = [{'gamma': 2.0,'theta_E': 1.0,'e1':0.1,'e2':0.1,
					'center_x':0.02,'center_y':-0.03},
					{'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0, 'dec_0':0.0}]
				z_list = [main_deflector_parameters['z_lens']]*2
				return model_list,kwargs_list,z_list

		subhalo_class = FakeSuhaloClass()
		los_class = None
		sample['subhalo_parameters'] = None
		multi_plane = True
		sub_image, meta_values = generate.draw_image(sample,los_class,
			subhalo_class,None,source_class,None,None,numpix,multi_plane,
			kwargs_numerics,mag_cut,add_noise)
		np.testing.assert_almost_equal(image,sub_image)

		# Add lens light & validate output
		# Generate image w/ deflector & w/out lens light
		sample['psf_parameters'] = {'psf_type':'GAUSSIAN',
				'fwhm': 0.1*orig_meta['pixel_width']}
		mag_cut=None
		image, meta_values = generate.draw_image(sample,None,None,
			main_deflector_class,source_class,None,None,numpix,multi_plane,
			kwargs_numerics,mag_cut,add_noise)
		# Generate image w/ deflector & lens light
		sample['lens_light_parameters'] = {'z_source':0.5,
			'magnitude':20,
			'output_ab_zeropoint':25.95,
			'R_sersic':1.,
			'n_sersic':1.2,
			'e1':0.,
			'e2':0.,
			'center_x':0.0,
			'center_y':0.0}
		lens_light_class = SingleSersicSource(cosmology_parameters='planck18',
			source_parameters=sample['lens_light_parameters'])
		lens_light_image, meta_values = generate.draw_image(sample,None,None,
			main_deflector_class,source_class,lens_light_class,
			None,numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		# Assert sum of center w/ lens light > sum of center orig_image
		self.assertTrue(np.sum(lens_light_image[90:110,90:110]) >
			np.sum(image[90:110,90:110]))

		# Add point source & validate output
		sample['point_source_parameters'] = {
			'x_point_source':0.001,
			'y_point_source':0.001,
			'magnitude':22,
			'output_ab_zeropoint':25.95,
			'compute_time_delays':False}
		point_source_class = SinglePointSource(
			sample['point_source_parameters'])
		image_ps, meta_values = generate.draw_image(sample,None,None,
				main_deflector_class,source_class,None,point_source_class,
				numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		
		# Check that more light is added to the image
		self.assertTrue(np.sum(image_ps) > np.sum(image))

		# Check that image positions are written to metadata
		pfix = 'point_source_parameters_'
		self.assertTrue(pfix+'num_images' in meta_values.keys())
		self.assertTrue(pfix+'x_image_0' in meta_values.keys())
		self.assertTrue(pfix+'y_image_1' in meta_values.keys())
		self.assertTrue(pfix+'x_image_3' in meta_values.keys())
		self.assertTrue(pfix+'y_image_3' in meta_values.keys())

		# Check that image magnifications are written to metadata
		self.assertTrue(pfix+'magnification_0' in meta_values.keys()) 
		self.assertTrue(pfix+'magnification_3' in meta_values.keys())

		# Check that if num_images < 3, we get Nan for image 2 & image 3
		if(meta_values[pfix+'num_images'] < 3):
			self.assertTrue(meta_values[pfix+'x_image_3'] == np.nan)
			self.assertTrue(meta_values[pfix+'y_image_2'] == np.nan)

		# Test using lens_equation_solver parameters in sample:
		sample['lens_equation_solver_parameters'] = {
			'min_distance':0.05
		}
		image_ps, meta_values = generate.draw_image(sample,None,None,
				main_deflector_class,source_class,None,point_source_class,
				numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
				
		# Check that more light is added to the image
		self.assertTrue(np.sum(image_ps) > np.sum(image))

		# Test time delay computation
		sample['point_source_parameters']['compute_time_delays'] = True

		# Check that if kappa_ext is not defined, we get a ValueError
		with self.assertRaises(ValueError):
			image, meta_values = generate.draw_image(sample,None,None,
				main_deflector_class,source_class,None,point_source_class,
				numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)

		# Check that correct metadata is written
		sample['point_source_parameters']['kappa_ext'] = 0.01
		image, meta_values = generate.draw_image(sample,None,None,
			main_deflector_class,source_class,None,point_source_class,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		self.assertTrue(pfix+'x_image_0' in meta_values.keys())
		self.assertTrue(pfix+'time_delay_0' in meta_values.keys())
		self.assertTrue(pfix+'time_delay_3' in meta_values.keys())
		self.assertTrue(pfix+'ddt' in meta_values.keys())

		# Check that if num_images < 3, we get Nan for time delay 3
		if(meta_values[pfix+'num_images'] < 4):
			self.assertTrue(meta_values[pfix+'time_delay_3'] == np.nan)

		# Cleanup
		os.remove(cosmos_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(cosmos_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(cosmos_folder+'npy_files')

	def test_draw_drizzled_image(self):
		# Check that the pipeline works as expected by running through the
		# same excercises as the draw_image pipeline and comparing
		# directly to the outputs of that pipeline.
		# Start with the simplest configuration, a source with nothing lensing
		# the source
		source_parameters = {'z_source':0.19499999,
			'cosmos_folder':cosmos_folder,'max_z':None,
			'minimum_size_in_pixels':None,'faintest_apparent_mag':None,
			'smoothing_sigma':0.0,'random_rotation':False,
			'min_flux_radius':None,'output_ab_zeropoint':25.95,
			'center_x':0.0,'center_y':0.0,
			'source_inclusion_list':np.array([0])}
		los_class = None
		subhalo_class = None
		main_deflector_class = None
		source_class = COSMOSIncludeCatalog(cosmology_parameters='planck18',
			source_parameters=source_parameters)
		# Set the source redshift to the redshift of the catalog image to avoid
		# any rescaling
		source_parameters['z_source'] = source_class.catalog['z'][0]
		numpix = 200
		multi_plane = False
		kwargs_numerics = {'supersampling_factor':1}
		mag_cut = None
		add_noise = False

		# Grab the image we want to compare to
		orig_image,orig_meta = source_class.image_and_metadata(0)
		orig_image = orig_image[17:-17,:]
		orig_image = orig_image[:,1:]/2 + orig_image[:,:-1]/2
		orig_image = orig_image[:,16:-16]

		# Create a fake sample from our sampler
		sim_pixel_width = orig_meta['pixel_width']
		sample = {
			'main_deflector_parameters':{'z_lens':0.0},
			'source_parameters':source_parameters,
			'cosmology_parameters':'planck18',
			'psf_parameters':{'psf_type':'GAUSSIAN',
				'fwhm': 0.1*orig_meta['pixel_width']},
			'detector_parameters':{'pixel_scale':sim_pixel_width,
				'ccd_gain':1.58,'read_noise':3.0,'magnitude_zero_point':25.127,
				'exposure_time':1380.0,'sky_brightness':15.83,'num_exposures':1,
				'background_noise':None},
			'drizzle_parameters':{'supersample_pixel_scale':sim_pixel_width,
				'output_pixel_scale':sim_pixel_width,'wcs_distortion':None,
				'offset_pattern':[(0,0),(0.0,0),(0.0,0.0),(-0.0,-0.0)]}
		}

		# Draw our image. This should just be the source itself
		image, meta_values = generate.draw_drizzled_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)

		# Check that the image is just the source
		np.testing.assert_almost_equal(image,orig_image)

		# Make the offset pattern more realistic and change the pixel widths
		sample['drizzle_parameters']['offset_pattern'] = [(0,0),(0.5,0),
			(0.0,0.5),(0.5,0.5)]
		sample['detector_parameters']['pixel_scale'] = 0.04
		sample['drizzle_parameters']['supersample_pixel_scale'] = 0.02
		sample['drizzle_parameters']['output_pixel_scale'] = 0.03
		numpix = 128

		# Check that the mag_cut works
		add_noise=False
		mag_cut = 1.2
		image, meta_values = generate.draw_drizzled_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		self.assertTrue(image is None)
		self.assertTrue(meta_values is None)

		# Now add a deflector and see if we get a ring
		add_noise = False
		source_parameters['z_source'] = 1.0
		main_deflector_parameters =  {'M200':1e13,'z_lens': 0.5,'gamma': 2.0,
			'theta_E': 1.0,'e1':0.1,'e2':0.1,'center_x':0.02,'center_y':-0.03,
			'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0, 'dec_0':0.0}
		sample['main_deflector_parameters'] = main_deflector_parameters
		main_deflector_class = PEMDShear(cosmology_parameters='planck18',
			main_deflector_parameters=main_deflector_parameters)
		image, meta_values = generate.draw_drizzled_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)

		# Check for magnification and check most light is not in
		# center of image
		self.assertTupleEqual((170,170),image.shape)
		self.assertGreater(np.sum(image),np.sum(orig_image))
		self.assertGreater(np.mean(image[0:80,0:80]),
			np.mean(image[80:90,80:90]))

		# Now we'll turn off our main deflector but create a fake LOS
		# and subhalo class that gives the same profile.
		class FakeLOSClass():

			def update_parameters(self,*args,**kwargs):
				return

			def draw_los(self,*args,**kwargs):
				model_list = ['EPL','SHEAR']
				kwargs_list = [{'gamma': 2.0,'theta_E': 1.0,'e1':0.1,'e2':0.1,
					'center_x':0.02,'center_y':-0.03},
					{'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0, 'dec_0':0.0}]
				z_list = [main_deflector_parameters['z_lens']]*2
				return model_list,kwargs_list,z_list

			def calculate_average_alpha(self,*args,**kwargs):
				return ([],[],[])

		los_class = FakeLOSClass()
		sample['los_parameters'] = None
		multi_plane = True
		los_image, meta_values = generate.draw_drizzled_image(sample,
			los_class,subhalo_class,None,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		np.testing.assert_almost_equal(image,los_image)

		# Check that setting the noise flag returns a noisy image
		add_noise = True
		sample['psf_parameters']['point_source_supersampling_factor'] = 1
		kwargs_numerics = {'supersampling_factor':1,
			'point_source_supersampling_factor':1}
		los_image_noise, meta_values = generate.draw_drizzled_image(sample,
			los_class,subhalo_class,None,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)

		self.assertGreater(np.std(los_image_noise-image),1e-3)

		# Cleanup
		os.remove(cosmos_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(cosmos_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(cosmos_folder+'npy_files')

	def test_draw_drizzled_image_psf(self):
		# Test the pixel psf behaves identically to using fftconvolve
		# Setup a fairly basic situation with a source at redshift 1.0 an a
		# massive main deflector at redshift 0.5.
		source_parameters = {'z_source':1.0,'cosmos_folder':cosmos_folder,
			'max_z':None,'minimum_size_in_pixels':None,
			'faintest_apparent_mag':None,'smoothing_sigma':0.0,
			'random_rotation':False,'min_flux_radius':None,
			'output_ab_zeropoint':25.95,'center_x':0.0,'center_y':0.0,
			'source_inclusion_list':np.array([0])}
		los_class = None
		subhalo_class = None
		main_deflector_parameters =  {'M200':1e13,'z_lens': 0.5,'gamma': 2.0,
			'theta_E': 1.0,'e1':0.1,'e2':0.1,'center_x':0.02,'center_y':-0.03,
			'gamma1':0.01,'gamma2':-0.02,'ra_0':0.0, 'dec_0':0.0}
		main_deflector_class = PEMDShear(cosmology_parameters='planck18',
			main_deflector_parameters=main_deflector_parameters)
		source_class = COSMOSIncludeCatalog(cosmology_parameters='planck18',
			source_parameters=source_parameters)
		numpix = 128
		multi_plane = False
		kwargs_numerics = {'supersampling_factor':1}
		mag_cut = None
		add_noise = False
		sim_pixel_width = 0.04

		# Create a fake sample from our sampler
		sim_pixel_width = 0.04
		sample = {
			'main_deflector_parameters':main_deflector_parameters,
			'source_parameters':source_parameters,
			'cosmology_parameters':'planck18',
			'psf_parameters':{'psf_type':'NONE'},
			'detector_parameters':{'pixel_scale':sim_pixel_width,
				'ccd_gain':1.58,'read_noise':3.0,'magnitude_zero_point':25.127,
				'exposure_time':1380.0,'sky_brightness':15.83,'num_exposures':1,
				'background_noise':None},
			'drizzle_parameters':{'supersample_pixel_scale':sim_pixel_width,
				'output_pixel_scale':sim_pixel_width,'wcs_distortion':None,
				'offset_pattern':[(0,0),(0.0,0),(0.0,0.0),(-0.0,-0.0)],
				'psf_supersample_factor':1}
		}

		# Draw our image. This should just be the lensed source without
		# noise and without a psf. This will be our supersamled image.
		image, meta_values = generate.draw_drizzled_image(sample,los_class,
			subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		image_degrade = hubble_utils.degrade_image(image,2)

		# Now generate a pixel level psf that isn't supersampled.
		psf_pixel = np.zeros((63,63))
		x,y = np.meshgrid(np.arange(63),np.arange(63),indexing='ij')
		psf_pixel[x,y] = np.exp(-((x-31)**2+(y-31)**2))
		psf_pixel /= np.sum(psf_pixel)
		sample['psf_parameters'] = {'psf_type':'PIXEL',
			'kernel_point_source': psf_pixel,
			'point_source_supersampling_factor':1}

		# Now generate the image again in the degraded resolution
		kwargs_numerics = {'supersampling_factor':2}
		numpix = 64
		sample['detector_parameters']['pixel_scale'] = sim_pixel_width*2
		sample['drizzle_parameters']['output_pixel_scale'] = sim_pixel_width*2
		image_degrade_psf, meta_values = generate.draw_drizzled_image(sample,
			los_class,subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)

		# Compare to the scipy image
		scipy_image = fftconvolve(image_degrade,psf_pixel,mode='same')

		np.testing.assert_almost_equal(scipy_image,image_degrade_psf)

		# Now repeat this process but doing the psf convolution at the
		# supersampling scale.
		sample['psf_parameters']['point_source_supersampling_factor'] = 2
		sample['drizzle_parameters']['psf_supersample_factor'] = 2
		image_degrade_psf, meta_values = generate.draw_drizzled_image(sample,
			los_class,subhalo_class,main_deflector_class,source_class,None,None,
			numpix,multi_plane,kwargs_numerics,mag_cut,add_noise)
		scipy_image = hubble_utils.degrade_image(
			fftconvolve(image,psf_pixel,mode='same'),2)
		np.testing.assert_almost_equal(scipy_image,image_degrade_psf,
			decimal=6)

		# Make sure the sample detector_parameters weren't changed in place.
		self.assertEqual(sample['detector_parameters']['pixel_scale'],
			sim_pixel_width*2)

		# Now just make sure we can raise some errors. First an error
		# if no point_source_supersampling_factor was specified.
		with self.assertRaises(ValueError):
			sample['psf_parameters'] = {'psf_type':'PIXEL',
				'kernel_point_source': psf_pixel}
			image_degrade_psf, meta_values = generate.draw_drizzled_image(
				sample,los_class,subhalo_class,main_deflector_class,
				source_class,None,None,numpix,multi_plane,kwargs_numerics,
				mag_cut,add_noise)

		# Next an error if it doesn't equal the psf_supersample_factor
		with self.assertRaises(ValueError):
			sample['psf_parameters'] = {'psf_type':'PIXEL',
				'kernel_point_source': psf_pixel,
				'point_source_supersampling_factor':1}
			image_degrade_psf, meta_values = generate.draw_drizzled_image(
				sample,los_class,subhalo_class,main_deflector_class,
				source_class,None,None,numpix,multi_plane,kwargs_numerics,
				mag_cut,add_noise)

		# Next an error if the psf_supersample_factor is larger than the scaling
		# provided by the drizzle parameters.
		with self.assertRaises(ValueError):
			sample['psf_parameters']['point_source_supersampling_factor'] = 4
			sample['psf_parameters'] = {'psf_type':'PIXEL',
				'kernel_point_source': psf_pixel,
				'point_source_supersampling_factor':4}
			image_degrade_psf, meta_values = generate.draw_drizzled_image(
				sample,los_class,subhalo_class,main_deflector_class,
				source_class,None,None,numpix,multi_plane,kwargs_numerics,
				mag_cut,add_noise)

		# Cleanup
		os.remove(cosmos_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(cosmos_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(cosmos_folder+'npy_files')

	def test_main(self):
		# Test that the main function makes some images
		old_sys = copy.deepcopy(sys.argv)
		output_folder = 'test_data/test_dataset'
		sys.argv = ['test','test_data/config_dict.py',output_folder,'--n','10']
		generate.main()

		image_file_list = glob.glob(os.path.join(output_folder,'image_*.npy'))

		self.assertEqual(len(image_file_list),10)

		# Make sure all of the files are readable and have the correct size
		# for the config
		for image_file in image_file_list:
			img = np.load(image_file)
			self.assertTupleEqual(img.shape,(64,64))
			os.remove(image_file)

		# Make sure the metadata makes sense
		metadata = pd.read_csv(os.path.join(output_folder,'metadata.csv'))
		self.assertEqual(len(metadata),10)
		self.assertListEqual(list(
			metadata['cosmology_parameters_cosmology_name']),['planck18']*10)
		self.assertListEqual(list(
			metadata['detector_parameters_pixel_scale']),[0.08]*10)
		self.assertListEqual(list(metadata['subhalo_parameters_c_0']),
			[18.0]*10)
		self.assertListEqual(list(metadata['main_deflector_parameters_z_lens']),
			[0.5]*10)
		self.assertListEqual(list(metadata['source_parameters_z_source']),
			[1.5]*10)
		self.assertGreater(np.std(metadata['source_parameters_catalog_i']),0)
		self.assertListEqual(list(metadata['psf_parameters_fwhm']),
			[0.1]*10)
		# Check that the subhalo_parameters_sigma_sub are being drawn
		self.assertGreater(np.std(metadata['los_parameters_delta_los']),
			0.0)
		# Check that nothing is getting written under cross_object
		for key in metadata.keys():
			self.assertFalse('cross_object' in key)
			self.assertFalse('source_exclusion_list' in key)

		# Remove the metadata file
		os.remove(os.path.join(output_folder,'metadata.csv'))
		os.remove(os.path.join(output_folder,'config_dict.py'))

		sys.argv = old_sys

		# Also clean up the test cosmos cache
		test_cosmo_folder = 'test_data/cosmos/'
		os.remove(test_cosmo_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(test_cosmo_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(test_cosmo_folder+'npy_files')

	def test_main_drizzle(self):
		# Test that the main function makes some images
		old_sys = copy.deepcopy(sys.argv)
		output_folder = 'test_data/test_dataset'
		sys.argv = ['test','test_data/config_dict_drizz.py',output_folder,
			'--n','10']
		generate.main()

		image_file_list = glob.glob(os.path.join(output_folder,'image_*.npy'))

		self.assertEqual(len(image_file_list),10)

		# Make sure all of the files are readable and have the correct size
		# for the config
		for image_file in image_file_list:
			img = np.load(image_file)
			self.assertTupleEqual(img.shape,(85,85))
			os.remove(image_file)

		# Make sure the metadata makes sense
		metadata = pd.read_csv(os.path.join(output_folder,'metadata.csv'))
		self.assertEqual(len(metadata),10)
		self.assertListEqual(list(
			metadata['cosmology_parameters_cosmology_name']),['planck18']*10)
		self.assertListEqual(list(
			metadata['detector_parameters_pixel_scale']),[0.08]*10)
		self.assertListEqual(list(metadata['subhalo_parameters_c_0']),
			[18.0]*10)
		self.assertListEqual(list(metadata['main_deflector_parameters_z_lens']),
			[0.5]*10)
		self.assertListEqual(list(metadata['source_parameters_z_source']),
			[1.5]*10)
		self.assertGreater(np.std(metadata['source_parameters_catalog_i']),0)
		self.assertListEqual(list(metadata['psf_parameters_fwhm']),
			[0.1]*10)
		# Check that the subhalo_parameters_sigma_sub are being drawn
		self.assertGreater(np.std(metadata['los_parameters_delta_los']),
			0.0)
		# Check that nothing is getting written under cross_object
		for key in metadata.keys():
			self.assertFalse('cross_object' in key)
			self.assertFalse('source_exclusion_list' in key)

		# Remove the metadata file
		os.remove(os.path.join(output_folder,'metadata.csv'))
		os.remove(os.path.join(output_folder,'config_dict_drizz.py'))

		sys.argv = old_sys

		# Also clean up the test cosmos cache
		test_cosmo_folder = 'test_data/cosmos/'
		os.remove(test_cosmo_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(test_cosmo_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(test_cosmo_folder+'npy_files')
