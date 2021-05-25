import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from manada import Analysis
from scipy.stats import multivariate_normal
from lenstronomy.SimulationAPI.observation_api import SingleBand
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.Data.psf import PSF
import os
from shutil import copyfile
from matplotlib import pyplot as plt
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

	def test_normalize_inputs(self):
		# Test that normalizing inputs works as expected. First normalize
		# the metadata and make sure it agrees with hand computed values.
		metadata = pd.read_csv(self.fake_test_folder + 'metadata.csv')
		learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E',
			'subhalo_parameters_conc_beta']
		input_norm_path = self.fake_test_folder + 'norms.csv'
		norm_dict = Analysis.dataset_generation.normalize_inputs(metadata,
			learning_params,input_norm_path)

		# Check that the norms agree with what we would expect
		for lp in learning_params:
			self.assertAlmostEqual(np.mean(metadata[lp]),norm_dict['mean'][lp])
			self.assertAlmostEqual(np.std(metadata[lp]),norm_dict['std'][lp])

		# Change the metadata, but make sure the previous value is returned
		metadata['subhalo_parameters_conc_beta'] *= 2
		norm_dict = Analysis.dataset_generation.normalize_inputs(metadata,
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
			norm_dict = Analysis.dataset_generation.normalize_inputs(metadata,
				learning_params,input_norm_path)

		# Get rid of the file we made
		os.remove(input_norm_path)

	def test_unormalize_outputs(self):
		# Test that unormalizing the inputs works correctly
		# Create the normalization file
		learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los']
		metadata = pd.read_csv(self.fake_test_folder + 'metadata.csv')
		input_norm_path = self.fake_test_folder + 'norms.csv'
		norm_dict = Analysis.dataset_generation.normalize_inputs(metadata,
			learning_params,input_norm_path)

		mean = np.array([[1,2]]*2,dtype=np.float)
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

		image = Analysis.dataset_generation.rotate_image_batch(image,
			learning_params,output)

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
		_ = Analysis.dataset_generation.normalize_inputs(metadata,
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
		# specified in the baobab configuration file.
		kwargs_detector = {'pixel_scale':0.08,'ccd_gain':2.5,'read_noise':4.0,
			'magnitude_zero_point':25.9463,'exposure_time':540.0,
			'sky_brightness':17,'num_exposures':1, 'background_noise':None}
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

	def generate_rotations_dataset(self):
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
		_ = Analysis.dataset_generation.normalize_inputs(metadata,
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

			# Assert that does that should have changed have
			np.testing.assert_array_less(np.zeros(len(outputs)),
				np.abs(outputs[:,3]-rotated_outputs[:,3]))
			np.testing.assert_array_less(np.zeros(images.shape),
				np.abs(images-rotated_images))

		# Clean up the file now that we're done
		os.remove(input_norm_path)
		os.remove(tf_record_path)


class MSELossTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency.
		np.random.seed(2)

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


class DiagonalCovarianceTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)

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


class FullCovarianceLossTest(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)

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
				places=4)

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


class ConvModelsTest(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
		tf.random.set_seed(2)

	def test_build_resnet_50(self):
		# Just test that the model dimensions behave as we would expect
		image_size = (100,100,1)
		num_outputs = 8

		model = Analysis.conv_models.build_resnet_50(image_size,num_outputs)

		# Check shapes
		self.assertTupleEqual((None,100,100,1),model.input_shape)
		self.assertTupleEqual((None,100,100,1),model.input_shape)
		self.assertEqual(123,len(model.layers))

		# Check that the model compiles
		model.compile(loss='mean_squared_error')


class PosteriorFunctionsTest(unittest.TestCase):

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
		std_pred = np.random.normal(size=(batch_size,num_params))

		Analysis.posterior_functions.plot_coverage(y_pred,y_true,std_pred,
			parameter_names,block=False)
		plt.close('all')

		Analysis.posterior_functions.plot_coverage(y_pred,y_true,std_pred,
			parameter_names,block=False,show_error_bars=False)
		plt.close('all')
