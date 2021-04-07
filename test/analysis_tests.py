import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
from manada import Analysis
from scipy.stats import multivariate_normal
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DatasetGenerationTests(unittest.TestCase):

	def setUp(self):
		# Set up a random seed for consistency
		np.random.seed(2)
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

	def test_generate_tf_record(self):
		# Test that a reasonable tf record is generated.
		metadata = pd.read_csv(self.fake_test_folder + 'metadata.csv')
		learning_params = ['subhalo_parameters_sigma_sub',
			'los_parameters_delta_los','main_deflector_parameters_theta_E',
			'subhalo_parameters_conc_beta']
		metadata_path = self.fake_test_folder + 'metadata.csv'
		tf_record_path = self.fake_test_folder + 'tf_record_test'
		input_norm_path = self.fake_test_folder + 'norms.csv'
		Analysis.dataset_generation.generate_tf_record(self.fake_test_folder,
			learning_params,metadata_path,tf_record_path,
			input_norm_path=input_norm_path)
		self.assertTrue(os.path.exists(tf_record_path))

		# Probe the number of npy files to make sure the total number of files
		# each epoch matches what is expected
		num_npy = len(glob.glob(self.fake_test_folder+'image_*.npy'))
		print(num_npy)

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
		batch_size = 20
		dataset = raw_dataset.map(parse_image).batch(batch_size)
		self.dataset_comparison(metadata,learning_params,dataset,batch_size,
			num_npy)

		# Clean up the file now that we're done
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
