import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from paltas import Analysis
from lenstronomy.SimulationAPI.observation_api import SingleBand
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.Data.psf import PSF
import os
from shutil import copyfile
from matplotlib import pyplot as plt
import numba, copy, sys
from scipy import special
from scipy.stats import truncnorm, lognorm, multivariate_normal
from scipy.integrate import dblquad

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DatasetGenerationTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
		tf.random.set_seed(2)
		self.fake_test_folder = (os.path.dirname(
			os.path.abspath(__file__))+'/test_data/fake_train/')

	def dataset_comparison(self,metadata,learning_params,dataset,batch_size,
		num_npy):
		# Test that the dataset matches what is saved in the test directory
		# Run the same test as above
		index_array = []
		npy_counts = 0
		for batch in iter(dataset):
			# Read the image out
			height = batch['height'].numpy()[0]
			width = batch['width'].numpy()[0]
			batch_images = tf.io.decode_raw(batch['image'],
				out_type=np.float32).numpy().reshape(-1,
					height,width)
			npy_indexs = batch['index'].numpy()
			lens_params_batch = []
			for param in learning_params:
				lens_params_batch.append(batch[param].numpy())
			# Load the original image and lens parameters and make sure that they
			# match
			for batch_index in range(batch_size):
				npy_index = npy_indexs[batch_index]
				index_array.append(npy_index)
				image = batch_images[batch_index]
				original_image = np.load(self.fake_test_folder+
					'image_%07d.npy'%(npy_index)).astype(np.float32)
				np.testing.assert_almost_equal(image,original_image)
				lpi = 0
				for param in learning_params:
					lens_param_value = lens_params_batch[lpi][batch_index]
					self.assertAlmostEqual(lens_param_value,metadata[param][
						npy_index],places=4)
					lpi += 1
				npy_counts += 1
		# Ensure the total number of files is correct
		self.assertEqual(npy_counts,num_npy)

	def test_normalize_outputs(self):
		# Test that normalizing inputs works as expected. First normalize
		# the metadata and make sure it agrees with hand computed values.
		metadata = pd.read_csv(self.fake_test_folder + 'metadata.csv')
		learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E',
			'subhalo_parameters_conc_beta']
		input_norm_path = self.fake_test_folder + 'norms.csv'
		norm_dict = Analysis.dataset_generation.normalize_outputs(metadata,
			learning_params,input_norm_path)

		# Check that the norms agree with what we would expect
		for lp in learning_params:
			self.assertAlmostEqual(np.mean(metadata[lp]),norm_dict['mean'][lp])
			self.assertAlmostEqual(np.std(metadata[lp]),norm_dict['std'][lp])

		# Change the metadata, but make sure the previous value is returned
		metadata['subhalo_parameters_conc_beta'] *= 2
		norm_dict = Analysis.dataset_generation.normalize_outputs(metadata,
			learning_params,input_norm_path)
		for lp in ['subhalo_parameters_sigma_sub','los_parameters_delta_los',
			'main_deflector_parameters_theta_E']:
			self.assertAlmostEqual(np.mean(metadata[lp]),norm_dict['mean'][lp])
			self.assertAlmostEqual(np.std(metadata[lp]),norm_dict['std'][lp])
		self.assertAlmostEqual(np.mean(metadata['subhalo_parameters_conc_beta']),
			norm_dict['mean']['subhalo_parameters_conc_beta']*2)

		# Finally check that the code complains if there are some parameters
		# missing in the norm
		with self.assertRaises(ValueError):
			learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E',
			'subhalo_parameters_conc_beta','not_there']
			norm_dict = Analysis.dataset_generation.normalize_outputs(metadata,
				learning_params,input_norm_path)

		# Check that setting some parameters to the log works as well
		os.remove(input_norm_path)
		learning_params = ['los_parameters_delta_los',
			'main_deflector_parameters_theta_E','subhalo_parameters_conc_beta']
		log_learning_params = ['subhalo_parameters_sigma_sub']
		norm_dict = Analysis.dataset_generation.normalize_outputs(metadata,
				learning_params,input_norm_path,
				log_learning_params=log_learning_params)
		# Check that the norms agree with what we would expect
		for lp in learning_params:
			self.assertAlmostEqual(np.mean(metadata[lp]),norm_dict['mean'][lp])
			self.assertAlmostEqual(np.std(metadata[lp]),norm_dict['std'][lp])
		# Check that the log norms agree with what we would expect
		for lp in log_learning_params:
			print(lp)
			self.assertAlmostEqual(np.mean(np.log(metadata[lp])),
				norm_dict['mean'][lp])
			self.assertAlmostEqual(np.std(np.log(metadata[lp])),
				norm_dict['std'][lp])

		# Get rid of the file we made
		os.remove(input_norm_path)

	def test_unormalize_outputs(self):
		# Test that unormalizing the inputs works correctly
		# Create the normalization file
		learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los']
		metadata = pd.read_csv(self.fake_test_folder + 'metadata.csv')
		input_norm_path = self.fake_test_folder + 'norms.csv'
		norm_dict = Analysis.dataset_generation.normalize_outputs(metadata,
			learning_params,input_norm_path)

		mean = np.array([[1,2]]*2,dtype=np.float64)
		cov_mat = np.array([[[1,0.9],[0.9,1]]]*2)

		Analysis.dataset_generation.unnormalize_outputs(input_norm_path,
			learning_params,mean,cov_mat=cov_mat)

		mean_corrected = np.array([[1*norm_dict['std'][learning_params[0]]+
			norm_dict['mean'][learning_params[0]],
			2*norm_dict['std'][learning_params[1]]+
			norm_dict['mean'][learning_params[1]]]]*2)

		cov_corrected = np.array([[[1*norm_dict['std'][learning_params[0]]**2,
			0.9*norm_dict['std'][learning_params[0]]
			*norm_dict['std'][learning_params[1]]],
			[0.9*norm_dict['std'][learning_params[0]]
			*norm_dict['std'][learning_params[1]],
			1*norm_dict['std'][learning_params[1]]**2]]]*2)

		np.testing.assert_almost_equal(mean,mean_corrected)
		np.testing.assert_almost_equal(cov_mat,cov_corrected)

		# Get rid of the file we made
		os.remove(input_norm_path)

	def test_kwargs_detector_to_tf(self):
		# Test that pushing numpy images through lenstronomy returns the
		# same results as the tensorflow version.
		kwargs_detector = {'pixel_scale':0.08,'ccd_gain':2.5,'read_noise':4.0,
			'magnitude_zero_point':25.9463,'exposure_time':5400.0,
			'sky_brightness':22,'num_exposures':1, 'background_noise':None}
		# image = np.random.rand(64,64)
		image = np.zeros((8,8))
		tf_image = tf.constant(image,dtype=tf.float32)

		# Make our own single band for comparison
		single_band = SingleBand(**kwargs_detector)

		# Draw a bunch of realizations to make sure that something similar
		# is returned
		np_total = []
		tf_total = []
		for _ in range(1000):
			np_total.append(single_band.noise_for_model(image))
			tf_total.append(
				Analysis.dataset_generation.kwargs_detector_to_tf_noise(
					tf_image,kwargs_detector).numpy())
		np.testing.assert_allclose(np.std(np_total,axis=0),
			np.std(tf_total,axis=0),rtol=0.2)

		# Repeat with an image of roughly mean 100
		image = np.ones((8,8))*100 + np.random.randn(8,8)*2
		tf_image = tf.constant(image,dtype=tf.float32)
		np_total = []
		tf_total = []
		for _ in range(1000):
			np_total.append(single_band.noise_for_model(image))
			tf_total.append(
				Analysis.dataset_generation.kwargs_detector_to_tf_noise(
					tf_image,kwargs_detector).numpy())
		np.testing.assert_allclose(np.std(np_total,axis=0),
			np.std(tf_total,axis=0),rtol=0.2)

	def test_generate_tf_record(self):
		# Test that a reasonable tf record is generated.
		metadata = pd.read_csv(self.fake_test_folder + 'metadata.csv')
		learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E']
		metadata_path = self.fake_test_folder + 'metadata.csv'
		tf_record_path = self.fake_test_folder + 'tf_record_test'
		Analysis.dataset_generation.generate_tf_record(self.fake_test_folder,
			learning_params,metadata_path,tf_record_path)
		self.assertTrue(os.path.exists(tf_record_path))

		# Probe the number of npy files to make sure the total number of files
		# each epoch matches what is expected
		num_npy = len(glob.glob(self.fake_test_folder+'image_*.npy'))

		# Open up this TFRecord file and take a look inside
		raw_dataset = tf.data.TFRecordDataset(tf_record_path)

		# Define a mapping function to parse the image
		def parse_image(example):
			data_features = {
				'image': tf.io.FixedLenFeature([],tf.string),
				'height': tf.io.FixedLenFeature([],tf.int64),
				'width': tf.io.FixedLenFeature([],tf.int64),
				'index': tf.io.FixedLenFeature([],tf.int64),
			}
			for param in learning_params:
					data_features[param] = tf.io.FixedLenFeature(
						[],tf.float32)
			return tf.io.parse_single_example(example,data_features)
		batch_size = 10
		dataset = raw_dataset.map(parse_image).batch(batch_size)

		self.dataset_comparison(metadata,learning_params,dataset,batch_size,
			num_npy)

		# Clean up the file now that we're done
		os.remove(tf_record_path)

		# Use a learning parameter that isn't present in the metadata
		learning_params = ['subhalo_parameters_sigma_sub_not']
		Analysis.dataset_generation.generate_tf_record(self.fake_test_folder,
			learning_params,metadata_path,tf_record_path)
		raw_dataset = tf.data.TFRecordDataset(tf_record_path)
		dataset = raw_dataset.map(parse_image).batch(batch_size)
		self.assertTrue(os.path.exists(tf_record_path))
		fake_metadata = {
			'subhalo_parameters_sigma_sub_not':np.zeros(num_npy)}
		self.dataset_comparison(fake_metadata,learning_params,dataset,
			batch_size,num_npy)

		# Clean up the file now that we're done
		os.remove(tf_record_path)

	def test_rotate_image_batch(self):
		# Test that rotating an image and resimulating it with the new
		# parameters both give good values
		# Put together all the lenstronomy tools we'll need.
		lens_model_list = ['PEMD','SHEAR']
		kwargs_spemd = {'gamma': 1.96,'theta_E': 1.0, 'e1': -0.34, 'e2': 0.02,
			'center_x': -0.05, 'center_y': 0.12}
		kwargs_shear = {'gamma1': 0.05, 'gamma2': 0.02}
		lens_model_kwargs = [kwargs_spemd,kwargs_shear]
		source_model_list = ['SERSIC']
		kwargs_source = [{'amp':1.0, 'R_sersic':0.3, 'n_sersic':1.0,
			'center_x':0.0, 'center_y':0.0}]
		kwargs_numerics = {'supersampling_factor':1}
		lens_model = LensModel(lens_model_list)
		source_model = LightModel(source_model_list)
		kwargs_psf = {'psf_type': 'GAUSSIAN', 'fwhm': 0.1}
		psf_model = PSF(**kwargs_psf)
		kwargs_detector = {'pixel_scale':0.08, 'ccd_gain':2.5, 'read_noise':0.0,
			'magnitude_zero_point':25.9463, 'exposure_time':54000.0,
			'sky_brightness':50, 'num_exposures':1, 'background_noise':None}
		numpix = 64
		data_api = DataAPI(numpix=numpix,**kwargs_detector)
		image_model = ImageModel(data_api.data_class, psf_model, lens_model,
			source_model, None, None, kwargs_numerics=kwargs_numerics)

		image = image_model.image(lens_model_kwargs, kwargs_source, None,
			None).astype(np.float32)
		image = image.reshape((1,64,64,1))
		learning_params = ['main_deflector_parameters_center_x',
			'main_deflector_parameters_center_y','main_deflector_parameters_e1',
			'main_deflector_parameters_e2','main_deflector_parameters_gamma1',
			'main_deflector_parameters_gamma2']
		output = np.array([kwargs_spemd['center_x'],
			kwargs_spemd['center_y'],kwargs_spemd['e1'],kwargs_spemd['e2'],
			kwargs_shear['gamma1'],kwargs_shear['gamma2']]).reshape((1,6))

		rot_angle = np.random.uniform()*2*np.pi
		image = Analysis.dataset_generation.rotate_image_batch(image,
			learning_params,output,rot_angle)

		# Now confirm that updating the parameters to these values returns what
		# we want
		kwargs_spemd['center_x'] = output[0,0]
		kwargs_spemd['center_y'] = output[0,1]
		kwargs_spemd['e1'] = output[0,2]
		kwargs_spemd['e2'] = output[0,3]
		kwargs_shear['gamma1'] = output[0,4]
		kwargs_shear['gamma2'] = output[0,5]
		image_new = image_model.image(lens_model_kwargs, kwargs_source, None,
			None).astype(np.float32)

		self.assertLess(np.max((image_new-image[0,:,:,0])/(image_new+1e-2)),
			0.1)

	def test_rotate_covariance_batch(self):
		# Test that rotating the parameters of a covariance matrix works
		# as expected.
		learning_params = ['main_deflector_parameters_center_x',
			'main_deflector_parameters_center_y']

		# Generate a bunch of draws from this base covariance matrix
		cov_mat_gen = np.array([[2.0,0.3,0.0],[0.3,1.1,0.0],[0.0,0.0,1.0]])
		n_samps = int(5e5)
		y_pred = np.random.multivariate_normal(mean=np.zeros(3),
			cov=cov_mat_gen,size=n_samps)
		cov_mats = np.tile(cov_mat_gen,n_samps).T.reshape(n_samps,3,3)

		# Rotate the covariance matrix and make sure it squares with the
		# rotated covariance matrices
		rot_angle = np.pi/4
		Analysis.dataset_generation.rotate_covariance_batch(learning_params,
			cov_mats,rot_angle)
		Analysis.dataset_generation.rotate_params_batch(learning_params,
			y_pred,rot_angle)
		np.testing.assert_almost_equal(np.cov(y_pred.T),cov_mats[0],decimal=1)

		rot_angle = -np.pi/3
		learning_params = ['main_deflector_parameters_e1',
			'main_deflector_parameters_e2']
		Analysis.dataset_generation.rotate_covariance_batch(learning_params,
			cov_mats,rot_angle)
		Analysis.dataset_generation.rotate_params_batch(learning_params,
			y_pred,rot_angle)
		np.testing.assert_almost_equal(np.cov(y_pred.T),cov_mats[0],decimal=1)

		# Test the shear components
		learning_params = ['main_deflector_parameters_gamma1',
			'main_deflector_parameters_gamma2']
		Analysis.dataset_generation.rotate_covariance_batch(learning_params,
			cov_mats,rot_angle)
		Analysis.dataset_generation.rotate_params_batch(learning_params,
			y_pred,rot_angle)
		np.testing.assert_almost_equal(np.cov(y_pred.T),cov_mats[0],decimal=1)

	def test_generate_tf_dataset(self):
		# Test that build_tf_dataset has the correct batching behaviour and
		# returns the same data contained in the npy files and csv.
		num_npy = len(glob.glob(self.fake_test_folder+'image_*.npy'))

		learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E',
			'main_deflector_parameters_center_x',
			'main_deflector_parameters_center_y']
		metadata_path = self.fake_test_folder + 'metadata.csv'
		tf_record_path = self.fake_test_folder + 'tf_record_test'
		input_norm_path = self.fake_test_folder + 'norms.csv'
		Analysis.dataset_generation.generate_tf_record(self.fake_test_folder,
			learning_params,metadata_path,tf_record_path)
		metadata = pd.read_csv(metadata_path)
		_ = Analysis.dataset_generation.normalize_outputs(metadata,
			learning_params,input_norm_path)

		# Try batch size 10
		batch_size = 10
		n_epochs = 1
		norm_images = False
		dataset = Analysis.dataset_generation.generate_tf_dataset(
			tf_record_path,learning_params,batch_size,n_epochs,
			norm_images=norm_images,kwargs_detector=None)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,64,64,1])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,5])
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		# Try batch size 5 and n_epochs 2
		batch_size = 5
		n_epochs = 2
		dataset = Analysis.dataset_generation.generate_tf_dataset(
			tf_record_path,learning_params,batch_size,n_epochs,
			norm_images=norm_images,kwargs_detector=None)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,64,64,1])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,5])
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		# Try normalizing the data
		batch_size = 5
		n_epochs = 2
		norm_images=True
		dataset = Analysis.dataset_generation.generate_tf_dataset(
			tf_record_path,learning_params,batch_size,n_epochs,
			norm_images=norm_images,kwargs_detector=None)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,64,64,1])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,5])
			for image in batch[0].numpy():
				self.assertAlmostEqual(np.std(image),1,places=4)
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		# Finally, just check that the noise statistics follow what we've
		# specified in the kwargs.
		kwargs_detector = {'pixel_scale':0.08,'ccd_gain':2.5,'read_noise':4.0,
			'magnitude_zero_point':25.9463,'exposure_time':540.0,
			'sky_brightness':14,'num_exposures':1, 'background_noise':None}
		dataset = Analysis.dataset_generation.generate_tf_dataset(
			tf_record_path,learning_params,batch_size,n_epochs,
			norm_images=norm_images,kwargs_detector=kwargs_detector)
		npy_counts = 0
		for batch in dataset:
			for image_i in range(len(batch[0].numpy())):
				image = batch[0].numpy()[image_i]
				self.assertGreater(np.std(image[:1,:,0]),5e-1)
				self.assertGreater(np.std(image[-1:,:,0]),5e-1)
				self.assertGreater(np.std(image[:,:1,0]),5e-1)
				self.assertGreater(np.std(image[:,-1:,0]),5e-1)

		# Test that passing in multiple tf_records works
		second_tf_record = tf_record_path + '2'
		copyfile(tf_record_path, second_tf_record)
		batch_size = 10
		n_epochs = 1
		norm_images = False
		dataset = Analysis.dataset_generation.generate_tf_dataset(
			[tf_record_path,second_tf_record],learning_params,batch_size,
			n_epochs,norm_images=norm_images,kwargs_detector=None)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,64,64,1])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,5])
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs*2)

		# Clean up the file now that we're done
		os.remove(input_norm_path)
		os.remove(tf_record_path)
		os.remove(second_tf_record)

	def test_generate_tf_dataset_log(self):
		# Test that build_tf_dataset has the correct batching behaviour and
		# returns the same data contained in the npy files and csv.
		num_npy = len(glob.glob(self.fake_test_folder+'image_*.npy'))

		learning_params = ['los_parameters_delta_los',
			'main_deflector_parameters_theta_E',
			'main_deflector_parameters_center_x',
			'main_deflector_parameters_center_y']
		log_learning_params = ['subhalo_parameters_sigma_sub']
		metadata_path = self.fake_test_folder + 'metadata.csv'
		tf_record_path = self.fake_test_folder + 'tf_record_test'
		input_norm_path = self.fake_test_folder + 'norms.csv'
		Analysis.dataset_generation.generate_tf_record(self.fake_test_folder,
			learning_params+log_learning_params,metadata_path,tf_record_path)
		metadata = pd.read_csv(metadata_path)
		_ = Analysis.dataset_generation.normalize_outputs(metadata,
			learning_params,input_norm_path,
			log_learning_params=log_learning_params)

		# Try batch size 10
		batch_size = 10
		n_epochs = 1
		norm_images = True
		dataset = Analysis.dataset_generation.generate_tf_dataset(
			tf_record_path,learning_params,batch_size,n_epochs,
			norm_images=norm_images,kwargs_detector=None,
			log_learning_params=log_learning_params)
		npy_counts = 0
		for batch in dataset:
			self.assertListEqual(batch[0].get_shape().as_list(),
				[batch_size,64,64,1])
			self.assertListEqual(batch[1].get_shape().as_list(),
				[batch_size,5])
			self.assertEqual(np.sum(batch[1].numpy()[:,-1]<0),10)
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		# Clean up the file now that we're done
		os.remove(input_norm_path)
		os.remove(tf_record_path)

	def test_generate_rotations_dataset(self):
		# Test that the rotations dataset works, and that it returns
		# images that are rotated when compared to the raw dataset
		num_npy = len(glob.glob(self.fake_test_folder+'image_*.npy'))
		learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E',
			'main_deflector_parameters_center_x',
			'main_deflector_parameters_center_y']
		metadata_path = self.fake_test_folder + 'metadata.csv'
		tf_record_path = self.fake_test_folder + 'tf_record_test'
		input_norm_path = self.fake_test_folder + 'norms.csv'
		Analysis.dataset_generation.generate_tf_record(self.fake_test_folder,
			learning_params,metadata_path,tf_record_path)
		metadata = pd.read_csv(metadata_path)
		_ = Analysis.dataset_generation.normalize_outputs(metadata,
			learning_params,input_norm_path)
		batch_size = 10
		n_epochs = 1
		norm_images = True

		dataset = Analysis.dataset_generation.generate_tf_dataset(
			tf_record_path,learning_params,batch_size,n_epochs,
			norm_images=norm_images,kwargs_detector=None)
		rotated_dataset = (
			Analysis.dataset_generation.generate_rotations_dataset(
				tf_record_path,learning_params,batch_size,n_epochs,
				norm_images=norm_images,kwargs_detector=None))
		npy_counts = 0
		for rotated_batch in rotated_dataset:
			self.assertListEqual(list(rotated_batch[0].shape),
				[batch_size,64,64,1])
			self.assertListEqual(list(rotated_batch[1].shape),
				[batch_size,5])
			npy_counts += batch_size
		self.assertEqual(npy_counts,num_npy*n_epochs)

		rotated_dataset = (
			Analysis.dataset_generation.generate_rotations_dataset(
				tf_record_path,learning_params,batch_size,n_epochs,
				norm_images=norm_images,kwargs_detector=None))
		for batch in dataset:
			rotated_batch = next(rotated_dataset)
			rotated_images = rotated_batch[0]
			rotated_outputs = rotated_batch[1]
			images = batch[0].numpy()
			outputs = batch[1].numpy()

			# Sort both outputs to be able to compare them
			rotated_images = rotated_images[np.argsort(
				rotated_outputs[:,0])]
			rotated_outputs = rotated_outputs[np.argsort(
				rotated_outputs[:,0])]
			images = images[np.argsort(outputs[:,0])]
			outputs = outputs[np.argsort(outputs[:,0])]

			# Assert parameters that should have changed haven't
			np.testing.assert_almost_equal(outputs[:,0],
				rotated_outputs[:,0])
			np.testing.assert_almost_equal(outputs[:,3]**2+outputs[:,4]**2,
				rotated_outputs[:,3]**2+rotated_outputs[:,4]**2)

			# Assert that those that should have changed have
			np.testing.assert_array_less(np.zeros(len(outputs)),
				np.abs(outputs[:,3]-rotated_outputs[:,3]))
			np.testing.assert_array_less(np.zeros(images.shape),
				np.abs(images-rotated_images))

		# Just make sure that when we pass the log of the parameters in
		# things behave as expected
		learning_params = ['los_parameters_delta_los',
			'main_deflector_parameters_theta_E',
			'main_deflector_parameters_center_x',
			'main_deflector_parameters_center_y']
		log_learning_params = ['subhalo_parameters_sigma_sub']
		log_dataset = (
			Analysis.dataset_generation.generate_rotations_dataset(
				tf_record_path,learning_params,batch_size,n_epochs,
				norm_images=norm_images,kwargs_detector=None,
				log_learning_params=log_learning_params))
		for batch in dataset:
			log_batch = next(log_dataset)
			log_images = log_batch[0]
			log_outputs = log_batch[1]
			images = batch[0].numpy()
			outputs = batch[1].numpy()

			# Sort both outputs to be able to compare them
			log_images = log_images[np.argsort(
				log_outputs[:,-1])]
			log_outputs = log_outputs[np.argsort(
				log_outputs[:,-1])]
			images = images[np.argsort(outputs[:,0])]
			outputs = outputs[np.argsort(outputs[:,0])]

			# Assert parameters that should have changed haven't
			np.testing.assert_almost_equal(np.log(outputs[:,0]),
				log_outputs[:,-1])
			np.testing.assert_almost_equal(outputs[:,3]**2+outputs[:,4]**2,
				log_outputs[:,2]**2+log_outputs[:,3]**2)

		# Clean up the file now that we're done
		os.remove(input_norm_path)
		os.remove(tf_record_path)

	def test_generate_params_as_input_dataset(self):
		# Test with an artificial dataset as well as the outputs of a call
		# to the other dataset generation functions.

		# Create our simplistic base dataset
		def artificial_dataset():
			for _ in range(2):
				images = np.ones((10,64,64,1))
				values = np.repeat(np.arange(5).reshape((1,5)),5,axis=0)
				yield images, values

		# Set some arbitrary parameters
		base_dataset = artificial_dataset()
		all_params = ['a','b','c','d','e']
		params_as_inputs = ['b','e']

		# Generate our new dataset
		dataset_params_inputs = (
			Analysis.dataset_generation.generate_params_as_input_dataset(
				base_dataset,params_as_inputs,all_params))

		for inputs,output in dataset_params_inputs:
			np.testing.assert_almost_equal(inputs[0],np.ones((10,64,64,1)))
			np.testing.assert_almost_equal(inputs[1],np.repeat(
				np.array([[1,4]]),5,axis=0))
			np.testing.assert_almost_equal(output,np.repeat(
				np.array([[0,2,3]]),5,axis=0))

		# Now we can pass in a dataset coming from generate_tf_dataset.
		all_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E',
			'main_deflector_parameters_center_x',
			'main_deflector_parameters_center_y']
		params_as_inputs = ['subhalo_parameters_sigma_sub']
		metadata_path = self.fake_test_folder + 'metadata.csv'
		tf_record_path = self.fake_test_folder + 'tf_record_test'
		input_norm_path = self.fake_test_folder + 'norms.csv'
		Analysis.dataset_generation.generate_tf_record(self.fake_test_folder,
			all_params,metadata_path,tf_record_path)
		metadata = pd.read_csv(metadata_path)
		_ = Analysis.dataset_generation.normalize_outputs(metadata,
			all_params,input_norm_path)
		batch_size = 5
		n_epochs = 1
		norm_images = False
		base_dataset = Analysis.dataset_generation.generate_tf_dataset(
			tf_record_path,all_params,batch_size,n_epochs,
			norm_images=norm_images,kwargs_detector=None)
		dataset_params_inputs = (
			Analysis.dataset_generation.generate_params_as_input_dataset(
				base_dataset,params_as_inputs,all_params))
		for batch in dataset_params_inputs:
			self.assertListEqual(list(batch[0][0].shape),
				[batch_size,64,64,1])
			self.assertListEqual(list(batch[0][1].shape),
				[batch_size,1])
			self.assertListEqual(list(batch[1].shape),
				[batch_size,4])

		# Repeat the same but for the rotation dataset
		rotated_dataset = (
			Analysis.dataset_generation.generate_rotations_dataset(
				tf_record_path,all_params,batch_size,n_epochs,
				norm_images=norm_images,kwargs_detector=None))
		dataset_params_inputs = (
			Analysis.dataset_generation.generate_params_as_input_dataset(
				rotated_dataset,params_as_inputs,all_params))
		for batch in dataset_params_inputs:
			self.assertListEqual(list(batch[0][0].shape),
				[batch_size,64,64,1])
			self.assertListEqual(list(batch[0][1].shape),
				[batch_size,1])
			self.assertListEqual(list(batch[1].shape),
				[batch_size,4])

		# Clean up the file now that we're done
		os.remove(input_norm_path)
		os.remove(tf_record_path)


class MSELossTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency.
		np.random.seed(2)

	def test_convert_output(self):
		# Test that the output is converted correctly
		num_params = 10
		loss_class = Analysis.loss_functions.MSELoss(num_params,None)
		output = tf.ones((1024,10))
		y_pred = loss_class.convert_output(output)
		np.testing.assert_array_equal(y_pred.numpy(),output.numpy())

	def test_draw_samples(self):
		# Test that the prediction is always the same
		num_params = 10
		loss_class = Analysis.loss_functions.MSELoss(num_params,None)
		output = tf.ones((1024,10))*2
		n_samps = 20
		predict_samps = loss_class.draw_samples(output,n_samps)
		for i in range(len(predict_samps)):
			np.testing.assert_array_equal(predict_samps[i],output.numpy())

	def test_mse_loss(self):
		# Test that the loss treats flip pairs correctly.
		flip_pairs = None
		for num_params in range(1,10):
			loss_class = Analysis.loss_functions.MSELoss(num_params,
				flip_pairs)
			y_true = np.random.randn(num_params).reshape((1,-1))
			y_pred = np.random.randn(num_params*2).reshape((1,-1))
			loss_tensor = loss_class.loss(tf.constant(y_true,dtype=tf.float32),
				tf.constant(y_pred,dtype=tf.float32))
			self.assertAlmostEqual(loss_tensor.numpy()[0],np.mean(np.square(
				y_true-y_pred[:,:num_params])),places=5)

		# Test that including some flip matrices works
		flip_pairs = [[1,2],[3,4],[5,6]]
		num_params = 8
		loss_class = Analysis.loss_functions.MSELoss(num_params,flip_pairs)

		# Test a few combinations
		y_true = np.ones((4,num_params))
		y_pred = np.ones((4,num_params))
		for fp in flip_pairs:
			y_pred[:,fp] = -1
			loss_tensor = loss_class.loss(tf.constant(y_true,dtype=tf.float32),
				tf.constant(y_pred,dtype=tf.float32))
			self.assertEqual(np.sum(loss_tensor.numpy()),0)

		# Check that flipping something outside the flip pair returns error
		y_pred[:] = -1
		loss_tensor = loss_class.loss(tf.constant(y_true,dtype=tf.float32),
			tf.constant(y_pred,dtype=tf.float32))
		self.assertEqual(np.sum(loss_tensor.numpy()),4)

		# Check the weighting the loss works
		weight_terms = [[0,10]]
		flip_pairs = None
		num_params = 10
		loss_class = Analysis.loss_functions.MSELoss(num_params,None,
			weight_terms)
		y_true = np.random.rand(num_params).reshape((1,-1))
		y_pred = np.random.rand(num_params).reshape((1,-1))
		loss_tensor = loss_class.loss(tf.constant(y_true,dtype=tf.float32),
			tf.constant(y_pred,dtype=tf.float32))
		weights = np.ones(num_params)
		weights[0] = 10
		manual_loss = np.mean(np.square(y_true-y_pred)*weights)
		self.assertAlmostEqual(loss_tensor.numpy()[0],manual_loss,places=5)


class DiagonalCovarianceTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)

	def test_convert_output(self):
		# Test that the output is converted correctly
		num_params = 10
		loss_class = Analysis.loss_functions.DiagonalCovarianceLoss(num_params,
			None)
		output = tf.random.normal((1024,20))
		y_pred, std_pred = loss_class.convert_output(output)
		np.testing.assert_array_equal(y_pred.numpy(),output.numpy()[:,:10])
		np.testing.assert_array_equal(std_pred.numpy(),output.numpy()[:,10:])

	def test_draw_samples(self):
		# Test that the prediction is always the same
		num_params = 10
		loss_class = Analysis.loss_functions.DiagonalCovarianceLoss(num_params,
			None)
		output = tf.ones((1024,20))*2
		n_samps = 10000
		predict_samps = loss_class.draw_samples(output,n_samps)
		np.testing.assert_almost_equal(np.mean(predict_samps,axis=0),
			output.numpy()[:,:10],decimal=1)
		np.testing.assert_almost_equal(np.std(predict_samps,axis=0),
			np.exp(output.numpy()[:,:10]/2),decimal=1)

	def test_log_gauss_diag(self):
		# Will not be used for this test, but must be passed in.
		flip_pairs = None
		for num_params in range(1,20):
			# Pick a random true, pred, and std and make sure it agrees with the
			# scipy calculation
			loss_class = Analysis.loss_functions.DiagonalCovarianceLoss(
				num_params,flip_pairs)
			y_true = np.random.randn(num_params)
			y_pred = np.random.randn(num_params)
			std_pred = np.random.randn(num_params)
			nlp_tensor = loss_class.log_gauss_diag(tf.constant(y_true),
				tf.constant(y_pred),tf.constant(std_pred))

			# Compare to scipy function to be exact. Add 2 pi offset.
			scipy_nlp = -multivariate_normal.logpdf(y_true,y_pred,
				np.diag(np.exp(std_pred))) - np.log(2 * np.pi) * num_params/2
			self.assertAlmostEqual(nlp_tensor.numpy(),scipy_nlp)

	def test_loss(self):
		# Test that the diagonal covariance loss gives the correct values
		flip_pairs = [[1,2],[3,4]]
		num_params = 6
		loss_class = Analysis.loss_functions.DiagonalCovarianceLoss(num_params,
			flip_pairs)

		# Set up a couple of test function to make sure that the minimum loss
		# is taken
		y_true = np.ones((4,num_params))
		y_pred = np.ones((4,num_params))
		y_pred[1,[1,2]] = -1
		y_pred[2,[3,4]] = -1
		y_pred[3,[1,2,3,4]] = -1
		std_pred = np.ones((4,num_params))

		yptf = tf.constant(np.concatenate([y_pred,std_pred],axis=-1),
			dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		diag_loss = loss_class.loss(yttf,yptf)
		self.assertAlmostEqual(np.sum(diag_loss.numpy()),12)

		# Repeat this excercise, but introducing error in prediction
		y_pred[:,0] = 10
		# The correct value of the nlp
		scipy_nlp = 0
		for i in range(len(y_pred)):
			scipy_nlp += -multivariate_normal.logpdf(y_true[i],y_pred[0],
				np.diag(np.exp(std_pred[i]))) -np.log(2 * np.pi)*num_params/2

		yptf = tf.constant(np.concatenate([y_pred,std_pred],axis=-1),
			dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		diag_loss = loss_class.loss(yttf,yptf)
		self.assertAlmostEqual(np.sum(diag_loss.numpy()),scipy_nlp,places=4)

		# Confirm that when the wrong pair is flipped, it does not
		# return the same answer.
		y_pred[:,0] = -1
		yptf = tf.constant(np.concatenate([y_pred,std_pred],axis=-1),
			dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		diag_loss = loss_class.loss(yttf,yptf)
		self.assertGreater(np.sum(diag_loss.numpy()),12)


class FullCovarianceLossTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)

	def test_convert_output(self):
		# Test that the output is converted correctly
		num_params = 10
		loss_class = Analysis.loss_functions.FullCovarianceLoss(num_params,None)
		output = tf.ones((1024,65))
		y_pred, prec_mat, L_diag = loss_class.convert_output(output)
		np.testing.assert_array_equal(y_pred.numpy(),output.numpy()[:,:10])
		# Just check the shape for the rest, the hard work here is done by
		# test_construct_precision_matrix.
		self.assertTupleEqual(prec_mat.numpy().shape,(1024,10,10))
		self.assertTupleEqual(L_diag.numpy().shape,(1024,10))

	def test_draw_samples(self):
		# Test that the prediction is always the same
		num_params = 10
		loss_class = Analysis.loss_functions.FullCovarianceLoss(num_params,None)
		output = tf.ones((1024,65))
		y_pred, prec_mat, L_diag = loss_class.convert_output(output)
		prec_mat = prec_mat.numpy()
		cov_mat = np.linalg.inv(prec_mat)
		n_samps = 10000
		predict_samps = loss_class.draw_samples(output,n_samps)
		np.testing.assert_almost_equal(np.mean(predict_samps,axis=0),
			output.numpy()[:,:10],decimal=1)
		for i in range(len(output)):
			np.testing.assert_almost_equal(np.cov(predict_samps[:,i,:].T),
				cov_mat[i],decimal=1)

	def test_construct_precision_matrix(self):
		# A couple of test cases to make sure that the generalized precision
		# matrix code works as expected.
		num_params = 4
		flip_pairs = None
		loss_class = Analysis.loss_functions.FullCovarianceLoss(num_params,
			flip_pairs)

		# Set up a fake l matrix with elements
		l_mat_elements = np.array([[1,2,3,4,5,6,7,8,9,10]],dtype=float)
		l_mat = np.array([[np.exp(1),0,0,0],[2,np.exp(3),0,0],[4,5,np.exp(6),0],
			[7,8,9,np.exp(10)]])
		prec_mat = np.matmul(l_mat,l_mat.T)

		# Get the tf representation of the prec matrix
		l_mat_elements_tf = tf.constant(l_mat_elements)
		p_mat_tf, diag_tf = loss_class.construct_precision_matrix(
			l_mat_elements_tf)

		# Make sure everything matches
		np.testing.assert_almost_equal(p_mat_tf.numpy()[0],prec_mat,decimal=5)
		diag_elements = np.array([1,3,6,10])
		np.testing.assert_almost_equal(diag_tf.numpy()[0],diag_elements)

		# Rinse and repeat for a different number of elements with batching
		num_params = 3
		flip_pairs = None
		loss_class = Analysis.loss_functions.FullCovarianceLoss(num_params,
			flip_pairs)

		# Set up a fake l matrix with elements
		l_mat_elements = np.array([[1,2,3,4,5,6],[1,2,3,4,5,6]],dtype=float)
		l_mat = np.array([[np.exp(1),0,0],[2,np.exp(3),0],[4,5,np.exp(6)]])
		prec_mat = np.matmul(l_mat,l_mat.T)

		# Get the tf representation of the prec matrix
		l_mat_elements_tf = tf.constant(l_mat_elements)
		p_mat_tf, diag_tf = loss_class.construct_precision_matrix(
			l_mat_elements_tf)

		# Make sure everything matches
		for p_mat in p_mat_tf.numpy():
			np.testing.assert_almost_equal(p_mat,prec_mat)
		diag_elements = np.array([1,3,6])
		for diag in diag_tf.numpy():
			np.testing.assert_almost_equal(diag,diag_elements)

	def test_log_gauss_full(self):
		# Will not be used for this test, but must be passed in.
		flip_pairs = None
		for num_params in range(1,10):
			# Pick a random true, pred, and std and make sure it agrees with the
			# scipy calculation
			loss_class = Analysis.loss_functions.FullCovarianceLoss(num_params,
				flip_pairs)
			y_true = np.random.randn(num_params)
			y_pred = np.random.randn(num_params)

			l_mat_elements_tf = tf.constant(
				np.expand_dims(np.random.randn(
					int(num_params*(num_params+1)/2)),axis=0),dtype=tf.float32)

			p_mat_tf, L_diag = loss_class.construct_precision_matrix(
				l_mat_elements_tf)

			p_mat = p_mat_tf.numpy()[0]

			nlp_tensor = loss_class.log_gauss_full(tf.constant(np.expand_dims(
				y_true,axis=0),dtype=float),tf.constant(np.expand_dims(
					y_pred,axis=0),dtype=float),p_mat_tf,L_diag)

			# Compare to scipy function to be exact. Add 2 pi offset.
			scipy_nlp = (-multivariate_normal.logpdf(y_true,y_pred,
				np.linalg.inv(p_mat)) - np.log(2 * np.pi) * num_params/2)
			# The decimal error can be significant due to inverting the precision
			# matrix
			self.assertAlmostEqual(np.sum(nlp_tensor.numpy())/scipy_nlp,1,
				places=3)

	def test_loss(self):
		# Test that the diagonal covariance loss gives the correct values
		flip_pairs = [[1,2],[3,4]]
		num_params = 6
		loss_class = Analysis.loss_functions.FullCovarianceLoss(num_params,
				flip_pairs)

		# Set up a couple of test function to make sure that the minimum loss
		# is taken
		y_true = np.ones((4,num_params))
		y_pred = np.ones((4,num_params))
		y_pred[1,[1,2]] = -1
		y_pred[2,[3,4]] = -1
		y_pred[3,[1,2,3,4]] = -1
		L_elements_len = int(num_params*(num_params+1)/2)
		# Have to keep this matrix simple so that we still get a reasonable
		# answer when we invert it for scipy check
		L_elements = np.zeros((4,L_elements_len))+1e-2

		# Get out the covariance matrix in numpy
		l_mat_elements_tf = tf.constant(L_elements,dtype=tf.float32)
		p_mat_tf, L_diag = loss_class.construct_precision_matrix(
			l_mat_elements_tf)
		cov_mats = []
		for p_mat in p_mat_tf:
			cov_mats.append(np.linalg.inv(p_mat.numpy()))

		# The correct value of the nlp
		scipy_nlp = -multivariate_normal.logpdf(y_true[0],y_pred[0],
			cov_mats[0]) -np.log(2 * np.pi)*num_params/2
		scipy_nlp *= 4

		yptf = tf.constant(np.concatenate([y_pred,L_elements],axis=-1),
			dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		loss = loss_class.loss(yttf,yptf)

		self.assertAlmostEqual(np.sum(loss.numpy()),scipy_nlp,places=4)

		# Repeat this excercise, but introducing error in prediction
		y_pred[:,0] = 10
		scipy_nlp = -multivariate_normal.logpdf(y_true[0],y_pred[0],
			cov_mats[0]) -np.log(2 * np.pi)*num_params/2
		scipy_nlp *= 4

		yptf = tf.constant(np.concatenate([y_pred,L_elements],axis=-1),
			dtype=tf.float32)
		yttf = tf.constant(y_true,dtype=tf.float32)
		loss = loss_class.loss(yttf,yptf)

		self.assertAlmostEqual(np.sum(loss.numpy()),scipy_nlp,places=4)


class ConvModelsTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
		tf.random.set_seed(2)

	def test__xresnet_stack(self):
		# Test that building the xresnet stack works with different input
		# shapes.
		# First try stides of 2 with a dimension size that is divisible by 2
		x = tf.ones((8,64,64,64))
		kernel_size = 3
		strides = 2
		conv_shortcut = False
		name = 'test'
		blocks = 4
		filters = 64
		out = Analysis.conv_models._xresnet_stack(x,filters,kernel_size,strides,
			conv_shortcut,name,blocks)
		self.assertTupleEqual((8,32,32,64),tuple(out.shape))

		# Now repeat the same but with an odd dimension
		x = tf.ones((8,63,63,64))
		out = Analysis.conv_models._xresnet_stack(x,filters,kernel_size,strides,
			conv_shortcut,name,blocks)
		self.assertTupleEqual((8,32,32,64),tuple(out.shape))

	def test_build_xresnet34(self):
		# Testing every aspect of this will be tricky, so we'll just
		# do a few sanity check
		image_size = (64,64,1)
		num_outputs = 8

		model = Analysis.conv_models.build_xresnet34(image_size,num_outputs)

		# Check that the xresnet34 tricks have been applied
		n_bn2 = 0
		for weights in model.trainable_weights:
			if 'stack' in weights.name and 'bn2/gamma' in weights.name:
				n_bn2 += 1
				wnp = weights.numpy()
				np.testing.assert_array_equal(wnp,np.zeros(wnp.shape))
		self.assertEqual(n_bn2,16)

		# A few checks for the different layers
		for layer in model.layers:
			if 'conv' in layer.name:
				self.assertTrue(layer.bias is None)
			if 'dense' in layer.name:
				self.assertFalse(layer.bias is None)
			if 'stack4_block3_out' in layer.name:
				self.assertListEqual(layer.output.shape.as_list(),
					[None,2,2,512])
		self.assertTupleEqual((None,num_outputs),model.output_shape)

		# Check that adding the custom head used in fastai doesn't cause
		# any major issues.
		model_fa = Analysis.conv_models.build_xresnet34(image_size,num_outputs,
			custom_head=True)

		# Check some of the individual layers again.
		for layer in model_fa.layers:
			if 'conv' in layer.name:
				self.assertTrue(layer.bias is None)
			if 'dense' in layer.name:
				self.assertTrue(layer.bias is None)
		self.assertEqual(len(model_fa.layers)-len(model.layers),5)
		self.assertTupleEqual((None,num_outputs),model_fa.output_shape)
		del model

		# Build a resnet with an image shape that requires padding before the
		# stacks and make sure the output is sane.
		image_size = (170,170,1)
		num_outputs = 8
		model = Analysis.conv_models.build_xresnet34(image_size,num_outputs)
		image = np.ones((1,170,170,1))
		self.assertEqual(np.sum(np.isnan(model.predict(image))),0)
		del model

		# Check that freezing the head works
		model = Analysis.conv_models.build_xresnet34(image_size,num_outputs,
			train_only_head=False)
		self.assertGreater(len(model.trainable_weights),2)
		del model
		model = Analysis.conv_models.build_xresnet34(image_size,num_outputs,
			train_only_head=True)
		self.assertEqual(len(model.trainable_weights),2)

	def test_build_xresnet34_fc_inputs(self):
		# Confirm that the model behaves correctly with the new inputs.
		img_size = (64,64,1)
		num_outputs = 8
		num_fc_inputs = 2
		fc_model = Analysis.conv_models.build_xresnet34_fc_inputs(img_size,
			num_outputs,num_fc_inputs)

		# Check that the expected layers are present.
		self.assertEqual(fc_model.get_layer('fc_dense1').input.shape[1],514)
		self.assertEqual(fc_model.get_layer('fc_dense1').output.shape[1],256)

		# Now check that the model takes in the expected input and is dependent
		# on the fc inputs.
		input_image = np.zeros((1,)+img_size)
		input_fc = np.zeros((1,)+(num_fc_inputs,))
		zero_out = fc_model([input_image,input_fc])
		np.testing.assert_array_equal(zero_out,np.zeros((1,num_outputs)))

		ones_out = fc_model([input_image,input_fc+1])
		np.testing.assert_array_less(zero_out,np.abs(ones_out))


class TransformerModelsTests(unittest.TestCase):

	def test_build_population_transformer(self):
		# Test that the build transformer behaves as expected.
		num_outputs = 10
		img_size = ((128,128,1))
		max_n_images = 16
		num_layers = 2
		embedding_dim = 128
		num_heads = 2
		dff = 128
		droput_rate = 0.1
		batch_size = 4
		conv_trainable = False

		# Test that you can build the transformer and pass some inputs
		# through it.
		transformer = Analysis.transformer_models.build_population_transformer(
			num_outputs,img_size,max_n_images,num_layers,embedding_dim,
			num_heads,dff,droput_rate,conv_trainable=conv_trainable)
		image_mask = np.ones((batch_size,max_n_images),dtype=bool)
		image_mask[1,5:] = 0
		image_mask[2,10:] = 0
		image_mask = tf.convert_to_tensor(image_mask)
		fake_input = tf.random.uniform((batch_size,max_n_images) + img_size)

		out = transformer.predict([fake_input,image_mask])
		self.assertTupleEqual(out.shape,(batch_size,num_outputs))
		np.testing.assert_array_less(np.zeros(out.shape),np.abs(out))

		# Check that masking all the inputs returns a zero output.
		image_mask = tf.cast(tf.zeros((batch_size,max_n_images)),dtype=tf.bool)
		out = transformer.predict([fake_input,image_mask])
		np.testing.assert_array_equal(out,np.zeros(out.shape))


class PosteriorFunctionsTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
		tf.random.set_seed(2)

	def test_plot_coverage(self):
		# Just make sure that the plot is generated without issue
		batch_size = 1024
		parameter_names = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E',
			'subhalo_parameters_conc_beta']
		num_params = len(parameter_names)

		y_pred = np.random.normal(size=(batch_size,num_params))
		y_true = np.random.normal(size=(batch_size,num_params))
		std_pred = np.abs(np.random.normal(size=(batch_size,num_params)))

		Analysis.posterior_functions.plot_coverage(y_pred,y_true,std_pred,
			parameter_names,block=False)
		plt.close('all')

		Analysis.posterior_functions.plot_coverage(y_pred,y_true,std_pred,
			parameter_names,block=False,show_error_bars=False)
		plt.close('all')

	def test_calc_p_dlt(self):
		# Test the the calc_p_dlt returns the right percentages
		size = int(1e6)
		predict_samps = np.random.normal(size=(size,10,2))
		predict_samps[:,:,1] = 0
		y_test = np.array([[1,2,3,4,5,6,7,8,9,10],[0,0,0,0,0,0,0,0,0,0]]).T

		p_dlt = Analysis.posterior_functions.calc_p_dlt(predict_samps,y_test,
			cov_dist_mat=np.diag(np.ones(2)))
		t_p_dlt = np.array([0.682689,0.954499,0.997300,0.999936,0.999999]+
			[1.0]*5)
		np.testing.assert_almost_equal(p_dlt,t_p_dlt,decimal=3)

	def test_plot_calibration(self):
		# Test that the calibration plot is generated without any issue.
		size = int(1e6)
		predict_samps = np.random.normal(size=(size,10,2))
		predict_samps[:,:,1] = 0
		y_test = np.array([[1,2,3,4,5,6,7,8,9,10],[0,0,0,0,0,0,0,0,0,0]]).T
		Analysis.posterior_functions.plot_calibration(predict_samps,y_test,
			block=False)
		plt.close('all')


class HierarchicalInferenceTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
		tf.random.set_seed(2)

	def test_log_p_xi_omega(self):
		# Check that the calculation agrees with the evaluation function.
		predict_samps_hier = np.random.randn(10,1000,1024)
		hyperparameters = np.array([1,1])

		# Make sure it works with numba
		@numba.njit()
		def eval_func_xi_omega(predict_samps_hier,hyperparameters):
			logpdf = np.sum(predict_samps_hier,axis=0) + hyperparameters[0]
			logpdf *= hyperparameters[1]
			return logpdf

		np.testing.assert_array_equal(
			Analysis.hierarchical_inference.log_p_xi_omega(
				predict_samps_hier,hyperparameters,eval_func_xi_omega),
			np.sum(predict_samps_hier,axis=0)+1)

		# Change the hyperparameters to make sure the calculation matches.
		hyperparameters[1] = 20
		np.testing.assert_array_equal(
			Analysis.hierarchical_inference.log_p_xi_omega(
				predict_samps_hier,hyperparameters,eval_func_xi_omega),
			(np.sum(predict_samps_hier,axis=0)+1)*20)

	def test_log_p_omega(self):
		# Test that the evaluation funciton is used
		hyperparameters = np.array([1,0.2])

		# Test that it works with numba
		@numba.njit()
		def eval_func_omega(hyperparameters):
			if hyperparameters[0] < 0:
				return np.nan
			else:
				return hyperparameters[0]*hyperparameters[1]

		self.assertEqual(Analysis.hierarchical_inference.log_p_omega(
			hyperparameters,eval_func_omega),0.2)
		hyperparameters = np.array([-1,0.2])
		self.assertEqual(Analysis.hierarchical_inference.log_p_omega(
			hyperparameters,eval_func_omega),-np.inf)

	def test_gaussian_product_analytical(self):
		# Test for a few combinations of covariance matrices that the results
		# agree with a numerical integration.
		# Start simple, covariance matrix is the identity and means are all
		# the same
		mu_pred = np.ones(2)
		prec_pred = np.identity(2)
		mu_omega_i = np.ones(2)
		prec_omega_i = np.identity(2)
		mu_omega = np.ones(2)
		prec_omega = np.identity(2)

		# Calculate the integral using scipy.
		mn = multivariate_normal(mean=mu_pred,cov=np.linalg.inv(prec_pred))
		mn_pred = mn
		mn_om = mn
		mn_omi = mn

		def scipy_call(y,x):
			arr = np.array([x,y])
			return mn_pred.pdf(arr)*mn_om.pdf(arr)/mn_omi.pdf(arr)

		s_est = dblquad(scipy_call, -20, 20, -20, 20)
		self.assertAlmostEqual(np.log(s_est[0]),
			Analysis.hierarchical_inference.gaussian_product_analytical(
				mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,prec_omega))

		# Now make it so that omega_i has a larger covariance.
		prec_omega_i *= 0.5
		mn_omi = multivariate_normal(mean=mu_omega_i,
			cov=np.linalg.inv(prec_omega_i))
		s_est = dblquad(scipy_call, -20, 20, -20, 20)
		self.assertAlmostEqual(np.log(s_est[0]),
			Analysis.hierarchical_inference.gaussian_product_analytical(
				mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,prec_omega))

		# Now shift the means
		mu_omega_i *= 2
		mu_pred *= 0.5
		mn_pred = multivariate_normal(mean=mu_pred,
			cov=np.linalg.inv(prec_pred))
		mn_om = multivariate_normal(mean=mu_omega,
			cov=np.linalg.inv(prec_omega))
		mn_omi = multivariate_normal(mean=mu_omega_i,
			cov=np.linalg.inv(prec_omega_i))
		s_est = dblquad(scipy_call, -20, 20, -20, 20)
		self.assertAlmostEqual(np.log(s_est[0]),
			Analysis.hierarchical_inference.gaussian_product_analytical(
				mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,prec_omega))

		# Finally complicate the covariances a bit
		prec_pred = np.array([[1,0.3],[0.3,1]])
		prec_omega_i = np.array([[1,-0.3],[-0.3,1]])
		prec_omega = np.array([[10,0.05],[0.05,10]])
		mn_pred = multivariate_normal(mean=mu_pred,
			cov=np.linalg.inv(prec_pred))
		mn_om = multivariate_normal(mean=mu_omega,
			cov=np.linalg.inv(prec_omega))
		mn_omi = multivariate_normal(mean=mu_omega_i,
			cov=np.linalg.inv(prec_omega_i))
		s_est = dblquad(scipy_call, -20, 20, -20, 20)
		self.assertAlmostEqual(np.log(s_est[0]),
			Analysis.hierarchical_inference.gaussian_product_analytical(
				mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,prec_omega))

		# Make sure the providing an invalid set of precision matrices gives
		# -np.inf
		prec_pred = np.array([[1,0.8],[0.8,1]])
		prec_omega_i = np.array([[1,0.0],[0.0,1]])
		prec_omega = np.array([[0.5,0.05],[0.05,0.5]])
		self.assertEqual(-np.inf,
			Analysis.hierarchical_inference.gaussian_product_analytical(
				mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,prec_omega))


class ProbabilityClassTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
		tf.random.set_seed(2)

	def test_set_samples(self):
		# Make sure that setting the samples with both input types works
		predict_samps_input = np.random.randn(1000,1024,10)
		predict_samps_hier_input = np.transpose(predict_samps_input,[2,0,1])

		# Establish the eval_func_xi_omega_i that we actually call
		eval_func_xi_omega = None
		eval_func_omega = None

		@numba.njit()
		def eval_func_xi_omega_i(predict_samps_hier):
			return np.sum(predict_samps_hier,axis=0)

		# Establish our ProbabilityClass
		prob_class = Analysis.hierarchical_inference.ProbabilityClass(
			eval_func_xi_omega_i,eval_func_xi_omega,eval_func_omega)

		# Try setting the samples with predict_samps_hier_input
		prob_class.set_samples(predict_samps_hier_input=predict_samps_hier_input)
		self.assertFalse(
			Analysis.hierarchical_inference.predict_samps_hier is None)
		np.testing.assert_almost_equal(prob_class.p_samps_omega_i,
			np.sum(predict_samps_hier_input,axis=0))

		# Try setting the samples with predict_samps_input
		prob_class.set_samples(predict_samps_input=predict_samps_input)
		self.assertFalse(
			Analysis.hierarchical_inference.predict_samps_hier is None)
		np.testing.assert_array_equal(
			Analysis.hierarchical_inference.predict_samps_hier,
			predict_samps_hier_input)
		np.testing.assert_almost_equal(prob_class.p_samps_omega_i,
			np.sum(predict_samps_hier_input,axis=0))

	def test_log_post_omega(self):
		# Make sure that the prediction class combines all of the evaluation
		# functions.

		predict_samps_hier_input = np.random.rand(10,1000,1024)

		@numba.njit()
		def eval_func_xi_omega(predict_samps_hier,hyperparameters):
			logpdf = np.sum(predict_samps_hier,axis=0) + hyperparameters[0]
			logpdf *= hyperparameters[1]
			return logpdf

		@numba.njit()
		def eval_func_omega(hyperparameters):
			if hyperparameters[0] < 0:
				return -np.inf
			return hyperparameters[0]*hyperparameters[1]

		@numba.njit()
		def eval_func_xi_omega_i(predict_samps_hier):
			return np.sum(predict_samps_hier,axis=0)

		# Establish the class at the predict samples
		prob_class = Analysis.hierarchical_inference.ProbabilityClass(
			eval_func_xi_omega_i,eval_func_xi_omega,eval_func_omega)
		prob_class.set_samples(
			predict_samps_hier_input=predict_samps_hier_input)

		# Evaluate a few hyperparameter values to make sure it all adds up.
		hyperparameters = np.array([1,1])

		# Calculate things manually as a confirmation
		def calculate_manual_log_post(hyperparameters):
			lprior = eval_func_omega(hyperparameters)

			omega_i_term = eval_func_xi_omega_i(predict_samps_hier_input)
			omega_term = eval_func_xi_omega(predict_samps_hier_input,
				hyperparameters)

			like_ratio = special.logsumexp(omega_term - omega_i_term,axis=0)

			return lprior + np.sum(like_ratio)

		self.assertAlmostEqual(prob_class.log_post_omega(hyperparameters),
			calculate_manual_log_post(hyperparameters))

		hyperparameters = hyperparameters*0.02
		self.assertAlmostEqual(prob_class.log_post_omega(hyperparameters),
			calculate_manual_log_post(hyperparameters))


class ProbabilityClassAnalyticalTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
		tf.random.set_seed(2)

	def test_set_predictions(self):
		# Make sure that setting the samples with both input types works
		n_lenses = 1000
		mu_pred_array_input = np.random.randn(n_lenses,10)
		prec_pred_array_input = np.tile(np.expand_dims(np.identity(10),axis=0),
			(n_lenses,1,1))
		mu_omega_i = np.ones(10)
		cov_omega_i = np.identity(10)
		eval_func_omega = None

		# Establish our ProbabilityClassAnalytical
		prob_class = Analysis.hierarchical_inference.ProbabilityClassAnalytical(
			mu_omega_i,cov_omega_i,eval_func_omega)

		# Try setting the predictions
		prob_class.set_predictions(mu_pred_array_input,prec_pred_array_input)
		self.assertFalse(Analysis.hierarchical_inference.mu_pred_array is None)
		self.assertFalse(Analysis.hierarchical_inference.prec_pred_array is None)

	def test_log_integral_product(self):
		# Make sure that the log integral product just sums the log of each
		# integral
		n_lenses = 1000
		mu_pred_array = np.random.randn(n_lenses,10)
		prec_pred_array = np.tile(np.expand_dims(np.identity(10),axis=0),
			(n_lenses,1,1))
		mu_omega_i = np.ones(10)
		prec_omega_i = np.identity(10)
		mu_omega = np.ones(10)
		prec_omega = np.identity(10)

		# First calculate the value by hand.
		hand_integral = 0
		for i in range(len(mu_pred_array)):
			mu_pred = mu_pred_array[i]
			prec_pred = prec_pred_array[i]
			hand_integral += (
				Analysis.hierarchical_inference.gaussian_product_analytical(
					mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,
					prec_omega))

		# Now use the class.
		prob_class = Analysis.hierarchical_inference.ProbabilityClassAnalytical
		integral = prob_class.log_integral_product(mu_pred_array,
			prec_pred_array,mu_omega_i,prec_omega_i,mu_omega,prec_omega)

		self.assertAlmostEqual(integral,hand_integral,places=20)

	def test_log_post_omega(self):
		# Test that the log_post_omega calculation includes both the integral
		# and the prior.
		# Initialize the values we'll need for the probability class.
		n_lenses = 1000
		mu_pred_array_input = np.random.randn(n_lenses,10)
		prec_pred_array_input = np.tile(np.expand_dims(np.identity(10),axis=0),
			(n_lenses,1,1))
		mu_omega_i = np.ones(10)
		cov_omega_i = np.identity(10)
		prec_omega_i = cov_omega_i

		@numba.njit()
		def eval_func_omega(hyperparameters):
			if np.any(hyperparameters[len(hyperparameters)//2:] < 0):
				return -np.inf
			return 0

		# Establish our ProbabilityClassAnalytical
		prob_class = Analysis.hierarchical_inference.ProbabilityClassAnalytical(
			mu_omega_i,cov_omega_i,eval_func_omega)
		prob_class.set_predictions(mu_pred_array_input,prec_pred_array_input)

		# Test a simple array of zeros
		hyperparameters = np.zeros(20)
		mu_omega = np.zeros(10)
		prec_omega = np.identity(10)
		hand_calc = prob_class.log_integral_product(mu_pred_array_input,
			prec_pred_array_input,mu_omega_i,prec_omega_i,mu_omega,prec_omega)
		self.assertAlmostEqual(hand_calc,
			prob_class.log_post_omega(hyperparameters))

		# Check that violating the prior returns -np.inf
		hyperparameters = -np.ones(20)
		self.assertEqual(-np.inf,prob_class.log_post_omega(hyperparameters))

		# Check a more complicated variance matrix
		hyperparameters = np.random.rand(20)
		mu_omega = hyperparameters[:10]
		prec_omega = np.linalg.inv(np.diag(np.exp(hyperparameters[10:])**2))
		hand_calc = prob_class.log_integral_product(mu_pred_array_input,
			prec_pred_array_input,mu_omega_i,prec_omega_i,mu_omega,prec_omega)
		self.assertAlmostEqual(hand_calc,
			prob_class.log_post_omega(hyperparameters))


class ProbabilityClassEnsembleTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
		tf.random.set_seed(2)

	def test_set_predictions(self):
		# Make sure that setting the samples with both input types works
		n_lenses = 1000
		n_ensemble = 5
		mu_pred_array_input = np.random.randn(n_ensemble,n_lenses,10)
		prec_pred_array_input = np.tile(np.expand_dims(np.expand_dims(
			np.identity(10),axis=0),axis=0),(n_ensemble,n_lenses,1,1))
		mu_omega_i = np.ones(10)
		cov_omega_i = np.identity(10)
		eval_func_omega = None

		# Establish our ProbabilityClassAnalytical
		prob_class = Analysis.hierarchical_inference.ProbabilityClassEnsemble(
			mu_omega_i,cov_omega_i,eval_func_omega)

		# Try setting the predictions
		prob_class.set_predictions(mu_pred_array_input,prec_pred_array_input)
		self.assertFalse(
			Analysis.hierarchical_inference.mu_pred_array_ensemble is None)
		self.assertFalse(
			Analysis.hierarchical_inference.prec_pred_array_ensemble is None)

	def test_log_integral_product(self):
		# Make sure that the log integral product just sums the log of each
		# integral
		n_lenses = 1000
		n_ensemble = 5

		prec_pred_array = np.tile(np.expand_dims(np.expand_dims(
			np.identity(10),axis=0),axis=0),(n_ensemble,n_lenses,1,1))
		mu_omega_i = np.ones(10)
		prec_omega_i = np.identity(10)
		mu_omega = np.ones(10)
		prec_omega = np.identity(10)*5

		# First start with the case where each ensemble prediction is the same
		mu_pred_array = np.tile(np.expand_dims(np.random.randn(n_lenses,10),
			axis=0),(n_ensemble,1,1))

		# In this case we can just compare to the non ensemble class.
		prob_class = Analysis.hierarchical_inference.ProbabilityClassAnalytical
		integral = prob_class.log_integral_product(mu_pred_array[0],
			prec_pred_array[0],mu_omega_i,prec_omega_i,mu_omega,prec_omega)
		prob_class_ens = Analysis.hierarchical_inference.ProbabilityClassEnsemble
		ens_integral = prob_class_ens.log_integral_product(mu_pred_array,
			prec_pred_array,mu_omega_i,prec_omega_i,mu_omega,prec_omega)
		self.assertAlmostEqual(integral,ens_integral)

		# Now do a hand calculation for the more complicated case without
		# logsumexp
		mu_pred_array = np.random.randn(n_ensemble,n_lenses,10)*0.05
		hand_integral = 0
		for j in range(mu_pred_array.shape[1]):
			ensemble_integral = 0
			for i in range(len(mu_pred_array)):
				mu_pred = mu_pred_array[i,j]
				prec_pred = prec_pred_array[i,j]
				ensemble_integral += np.exp(
					Analysis.hierarchical_inference.gaussian_product_analytical(
						mu_pred,prec_pred,mu_omega_i,prec_omega_i,mu_omega,
						prec_omega))
			hand_integral += np.log(ensemble_integral/n_ensemble)

		ens_integral = prob_class_ens.log_integral_product(mu_pred_array,
			prec_pred_array,mu_omega_i,prec_omega_i,mu_omega,prec_omega)
		self.assertAlmostEqual(hand_integral,ens_integral)

	def test_log_post_omega(self):
		# Test that the log_post_omega calculation includes both the integral
		# and the prior.
		# Initialize the values we'll need for the probability class.
		n_lenses = 1000
		n_ensemble = 5
		mu_pred_array_input = np.tile(np.expand_dims(
			np.random.randn(n_lenses,10),axis=0),(n_ensemble,1,1))
		prec_pred_array_input = np.tile(np.expand_dims(np.expand_dims(
			np.identity(10),axis=0),axis=0),(n_ensemble,n_lenses,1,1))
		mu_omega_i = np.ones(10)
		cov_omega_i = np.identity(10)
		prec_omega_i = cov_omega_i

		@numba.njit()
		def eval_func_omega(hyperparameters):
			if np.any(hyperparameters[len(hyperparameters)//2:] < 0):
				return -np.inf
			return 0

		# Establish our ProbabilityClassAnalytical
		prob_class = Analysis.hierarchical_inference.ProbabilityClassEnsemble(
			mu_omega_i,cov_omega_i,eval_func_omega)
		prob_class.set_predictions(mu_pred_array_input,prec_pred_array_input)

		# Test a simple array of zeros
		hyperparameters = np.zeros(20)
		mu_omega = np.zeros(10)
		prec_omega = np.identity(10)
		hand_calc = prob_class.log_integral_product(mu_pred_array_input,
			prec_pred_array_input,mu_omega_i,prec_omega_i,mu_omega,prec_omega)
		self.assertAlmostEqual(hand_calc,
			prob_class.log_post_omega(hyperparameters))

		# Check that violating the prior returns -np.inf
		hyperparameters = -np.ones(20)
		self.assertEqual(-np.inf,prob_class.log_post_omega(hyperparameters))

		# Check a more complicated variance matrix
		hyperparameters = np.random.rand(20)
		mu_omega = hyperparameters[:10]
		prec_omega = np.linalg.inv(np.diag(np.exp(hyperparameters[10:])**2))
		hand_calc = prob_class.log_integral_product(mu_pred_array_input,
			prec_pred_array_input,mu_omega_i,prec_omega_i,mu_omega,prec_omega)
		self.assertAlmostEqual(hand_calc,
			prob_class.log_post_omega(hyperparameters))


class PdfFunctionsTests(unittest.TestCase):

	def test_eval_normal_logpdf_approx(self):
		# For a specific mu, sigma, upper, and lower, test that the log pdf
		# approximation gives the correct values inside the bounds, and then
		# suppressed values outside the bounds.
		mu = 1
		sigma = 5
		lower = -10
		upper = 10

		# Test within the bounds
		eval_at = np.linspace(-10,10,100)
		lpdf_approx = Analysis.pdf_functions.eval_normal_logpdf_approx(eval_at,
			mu,sigma,lower,upper)
		lpdf = truncnorm((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,
			scale=sigma).logpdf(eval_at)
		precision = 5
		np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

		# Test outside the bounds
		eval_at = np.linspace(-20,-10.0001,100)
		lpdf_approx = Analysis.pdf_functions.eval_normal_logpdf_approx(eval_at,
			mu,sigma,lower,upper)
		lpdf = truncnorm(-np.inf,np.inf,loc=mu,scale=sigma).logpdf(eval_at)
		np.testing.assert_array_less(lpdf_approx, lpdf)
		# assert greater because of the accept_norm
		np.testing.assert_array_less(lpdf-1000,lpdf_approx)

		eval_at = np.linspace(10.0001,20,100)
		lpdf_approx = Analysis.pdf_functions.eval_normal_logpdf_approx(eval_at,
			mu,sigma,lower,upper)
		lpdf = truncnorm(-np.inf,np.inf,loc=mu,scale=sigma).logpdf(eval_at)
		np.testing.assert_array_less(lpdf_approx, lpdf)
		# assert greater because of the accept_norm
		np.testing.assert_array_less(lpdf-1000,lpdf_approx)

		# Test that the default values work
		eval_at = np.linspace(-10,10,100)
		lpdf_approx = Analysis.pdf_functions.eval_normal_logpdf_approx(eval_at,
			mu,sigma)
		lpdf = truncnorm(-np.inf,np.inf,loc=mu,scale=sigma).logpdf(eval_at)
		np.testing.assert_almost_equal(lpdf_approx,lpdf)

	def test_eval_lognormal_logpdf_approx(self):
		# For a specific mu, sigma, upper, and lower, test that the log pdf
		# approximation gives the correct values inside the bounds, and then
		# suppressed values outside the bounds.
		mu = 1
		sigma = 5
		lower = 1
		upper = 10

		# Create a temp function for comparing
		def eval_lognormal_logpdf(eval_at,mu,sigma,lower=-np.inf,upper=np.inf):
			dist = lognorm(scale=np.exp(mu), s=sigma, loc=0.0)
			eval_unnormed_logpdf = dist.logpdf(eval_at)
			accept_norm = dist.cdf(upper) - dist.cdf(lower)
			eval_normed_logpdf = eval_unnormed_logpdf - np.log(accept_norm)
			eval_unnormed_logpdf[eval_at<lower] = -np.inf
			eval_unnormed_logpdf[eval_at>upper] = -np.inf
			return eval_normed_logpdf

		# Test within the bounds
		eval_at = np.linspace(1,10,100)
		lpdf_approx = Analysis.pdf_functions.eval_lognormal_logpdf_approx(
			eval_at,mu,sigma,lower,upper)
		lpdf = eval_lognormal_logpdf(eval_at,mu,sigma,lower,upper)
		precision = 5
		np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

		# Test outside the bounds
		eval_at = np.linspace(0.0000001,0.9999,100)
		lpdf_approx = Analysis.pdf_functions.eval_lognormal_logpdf_approx(
			eval_at,mu,sigma,lower,upper)
		lpdf = eval_lognormal_logpdf(eval_at,mu,sigma)
		np.testing.assert_array_less(lpdf_approx, lpdf)
		# assert greater because of the accept_norm
		np.testing.assert_array_less(lpdf-1000,lpdf_approx)

		eval_at = np.linspace(10.0001,20,100)
		lpdf_approx = Analysis.pdf_functions.eval_lognormal_logpdf_approx(
			eval_at,mu,sigma,lower,upper)
		lpdf = eval_lognormal_logpdf(eval_at,mu,sigma)
		np.testing.assert_array_less(lpdf_approx, lpdf)
		# assert greater because of the accept_norm
		np.testing.assert_array_less(lpdf-1000,lpdf_approx)

		# Check that without bounds the function behaves as expected.
		lpdf_approx = Analysis.pdf_functions.eval_lognormal_logpdf_approx(
			eval_at,mu,sigma)
		lpdf = eval_lognormal_logpdf(eval_at,mu,sigma)
		np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)

		# Check that the function doesn't fail if the lower is set to -np.inf
		lpdf_approx = Analysis.pdf_functions.eval_lognormal_logpdf_approx(
			eval_at,mu,sigma,lower=-np.inf)
		lpdf = eval_lognormal_logpdf(eval_at,mu,sigma)
		np.testing.assert_almost_equal(lpdf_approx, lpdf, precision)


class TrainModelTests(unittest.TestCase):

	def setUp(self):
		# Set the random seed so we don't run into trouble
		np.random.seed(20)

	def test_parse_args(self):
		# Check that the argument parser works as intended
		# We have to modify the sys.argv input which is bad practice
		# outside of a test
		old_sys = copy.deepcopy(sys.argv)
		sys.argv = ['test','train_config.py','--tensorboard_dir',
			'tensorboard_dir']
		args = Analysis.train_model.parse_args()
		self.assertEqual(args.training_config,'train_config.py')
		self.assertEqual(args.tensorboard_dir,'tensorboard_dir')
		sys.argv = old_sys

	def test_main(self):
		# Test that the main function runs and produces the expecteced
		# files.
		old_sys = copy.deepcopy(sys.argv)
		tensorboard_dir = 'test_data'
		sys.argv = ['test','test_data/train_config.py','--tensorboard_dir',
			tensorboard_dir]

		# Run a training with the first config file
		Analysis.train_model.main()

		# Cleanup the files we don't want
		os.remove('test_data/fake_model.h5')
		os.remove('test_data/fake_train/norms.csv')
		tensorboard_train = glob.glob('test_data/train/*')
		for f in tensorboard_train:
			os.remove(f)
		os.rmdir('test_data/train')
		tensorboard_val = glob.glob('test_data/validation/*')
		for f in tensorboard_val:
			os.remove(f)
		os.rmdir('test_data/validation')

		# Use a second config file that's a little different.
		sys.argv = ['test','test_data/train_config2.py','--tensorboard_dir',
			tensorboard_dir]
		Analysis.train_model.main()

		# Cleanup the files we don't want
		os.remove('test_data/fake_model.h5')
		os.remove('test_data/fake_train/norms.csv')
		os.remove('test_data/fake_train/data.tfrecord')
		tensorboard_train = glob.glob('test_data/train/*')
		for f in tensorboard_train:
			os.remove(f)
		os.rmdir('test_data/train')
		tensorboard_val = glob.glob('test_data/validation/*')
		for f in tensorboard_val:
			os.remove(f)
		os.rmdir('test_data/validation')

		# Use a config file that has some scalar inputs included.
		sys.argv = ['test','test_data/train_config_pai.py',
			'--tensorboard_dir',tensorboard_dir]
		Analysis.train_model.main()

		# Cleanup the files we don't want
		os.remove('test_data/fake_model.h5')
		os.remove('test_data/fake_train/norms.csv')
		os.remove('test_data/fake_train/data.tfrecord')
		tensorboard_train = glob.glob('test_data/train/*')
		for f in tensorboard_train:
			os.remove(f)
		os.rmdir('test_data/train')
		tensorboard_val = glob.glob('test_data/validation/*')
		for f in tensorboard_val:
			os.remove(f)
		os.rmdir('test_data/validation')

		sys.argv = old_sys
