# -*- coding: utf-8 -*-
"""
Manipulate strong lensing images to be used with tensorflow models.

This module contains the functions that allow the numpy file and csv outputs
of generate to be transformed into a format that can be consumed by
tensorflow.
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import glob, os
from tqdm import tqdm
from lenstronomy.SimulationAPI.observation_api import SingleBand
import warnings
from scipy.ndimage import rotate

# Global filters on the python warnings. Using this since filter
# behaviour is a bit weird.
DEFAULTVALUEWARNING = True


def normalize_outputs(metadata,learning_params,input_norm_path,
	log_learning_params=None):
	""" Normalize the outputs of the network

	Args:
		metadata(pd.DataFrame): A pandas object containing the metadata
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		input_norm_path (str): The path to a csv that contains the
			normalization to be applied to the output parameters. If the
			file already exists it will be read, if no file exists it will
			be written.
		log_learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn the log of. Can
			be None.

	Returns:
		(pd.DataFrame): A pandas dataframe with the the mean and standard
		deviation for each parameter.
	"""
	# If normalization file is provided use / write it
	if os.path.isfile(input_norm_path):
		print('Using input normalization found at %s'%(input_norm_path))
		norm_dict = pd.read_csv(input_norm_path,index_col='parameter')
		if not all(elem in norm_dict.index for elem in learning_params):
			raise ValueError('Not all of the learning parameters are ' +
				'present in the input normalization dictionary.')
		if log_learning_params is not None:
			if not all(elem in norm_dict.index for elem in log_learning_params):
				raise ValueError('Not all of the log learning parameters are ' +
					'present in the input normalization dictionary.')
	else:
		print('Writing input normalization to %s'%(input_norm_path))
		norm_dict = pd.DataFrame(columns=['parameter','mean','std'])
		norm_dict['parameter'] = learning_params

		# Calculate the normalization for each parameter
		data = metadata[learning_params].to_numpy()
		norm_dict['mean'] = np.mean(data,axis=0)
		norm_dict['std'] = np.std(data,axis=0)

		# Append a dictionary of the log parameters if provided.
		if log_learning_params is not None:
			log_norm_dict = pd.DataFrame(columns=['parameter','mean','std'])
			log_norm_dict['parameter'] = log_learning_params
			# Calculate the normalization for each parameter
			log_data = metadata[log_learning_params].to_numpy()
			log_norm_dict['mean'] = np.mean(np.log(log_data),axis=0)
			log_norm_dict['std'] = np.std(np.log(log_data),axis=0)
			norm_dict = norm_dict.append(log_norm_dict)

		# Set parameter to the index
		norm_dict = norm_dict.set_index('parameter')
		norm_dict.to_csv(input_norm_path)

	return norm_dict


def unnormalize_outputs(input_norm_path,learning_params,mean,standard_dev=None,
	cov_mat=None):
	""" Given NN outputs, undo the normalization step and return the parameters
	in the original space

	Args:
		input_norm_path (str): The path to a csv that contains the
			normalization to be undone.
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn. Length n_params.
		mean (np.array): A numpy array with dimensions (batch_size,n_params)
			containing the mean estimate for each parameter
		standard_dev (np.array): A numpy array with dimensions (batch_size,
			n_params) containing the standard deviation estimate for each
			parameter.
		cov_mat (np.array): A numpy array with dimensions (batch_size,n_params,
			n_params) containing the covariance matrix estiamtes for each
			image.

	Notes:
		All values will be modified in place.
	"""
	# Read our normalization dictionary
	norm_dict = pd.read_csv(input_norm_path,index_col='parameter')
	# Iterate over our parameters
	for lpi, param in enumerate(learning_params):
		param_mean = norm_dict['mean'][param]
		param_std = norm_dict['std'][param]

		# We always want to correct the mean
		mean[:,lpi] *= param_std
		mean[:,lpi] += param_mean

		# If provided we want to correct the standard deviation
		if standard_dev is not None:
			standard_dev[:,lpi] *= param_std

		# If provided we want to correct the covariance matrix
		if cov_mat is not None:
			cov_mat[:,lpi,:] *= param_std
			cov_mat[:,:,lpi] *= param_std


def kwargs_detector_to_tf_noise(image,kwargs_detector):
	""" Add noise to the tf tensor provided in agreement with kwargs_detector

	Args:
		image (tf.Tensor): A tensorflow tensor containing the image
		kwargs_detector (dict): A dictionary containing the detector kwargs
			used to generate the noise on the fly.

	Returns:
		(tf.Tensor): A tensorflow tensor containing the noise.
	"""
	# Use lenstronomy to do our noise calculations (lenstronomy is well
	# tested and is very flexible in its noise inputs).
	single_band = SingleBand(**kwargs_detector)

	# Add the background noise.
	noise = tf.random.normal(tf.shape(image)) * single_band.background_noise

	# Add the poisson noise. We have to redo the functions to be tf friendly
	variance = tf.maximum(single_band.flux_iid(image),tf.zeros_like(image))
	variance = tf.sqrt(variance) / single_band.exposure_time
	if single_band._data_count_unit == 'ADU':
		variance /= single_band.ccd_gain
	noise += tf.random.normal(tf.shape(image)) * variance

	return noise


def generate_tf_record(npy_folder,learning_params,metadata_path,
	tf_record_path):
	""" Generate a TFRecord file from a directory of numpy files.

	Args:
		root_path (str): The path to the folder containing the numpy files.
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		metadata_path (str):  The path to the csv file containing the
			image metadata.
		tf_record_path (str): The path to which the TFRecord will be saved
	"""
	# Pull the list of numpy filepaths from the directory
	npy_file_list = glob.glob(os.path.join(npy_folder,'image_*.npy'))
	# Open label csv
	metadata = pd.read_csv(metadata_path, index_col=None)

	# Warn the user a default value of 0 will be used for parameters not
	# present in the metadata file.
	global DEFAULTVALUEWARNING
	for param in learning_params:
		if param not in metadata and DEFAULTVALUEWARNING:
			warnings.warn('One or more parameters in learning_params is not '+
				' present in the metadata. A default value of 0 will be used.',
				category=RuntimeWarning)
			DEFAULTVALUEWARNING = False

	# Initialize the writer object and write the lens data
	with tf.io.TFRecordWriter(tf_record_path) as writer:
		for npy_file in tqdm(npy_file_list):
			# Pull the index from the filename
			index = int(npy_file[-11:-4])
			image_shape = np.load(npy_file).shape
			# The image must be converted to a tf string feature
			image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
				value=[np.load(npy_file).astype(np.float32).tostring()]))
			# Initialize a feature dictionary with the image, the height,
			# and the width
			feature = {
				'image': image_feature,
				'height': tf.train.Feature(
					int64_list=tf.train.Int64List(value=[image_shape[0]])),
				'width': tf.train.Feature(
					int64_list=tf.train.Int64List(value=[image_shape[1]])),
				'index': tf.train.Feature(
					int64_list=tf.train.Int64List(value=[index]))
			}
			# Add all of the lens parameters to the feature dictionary
			for param in learning_params:
				if param in metadata:
					value = metadata[param][index]
				else:
					value = 0
				# Write the feature
				feature[param] = tf.train.Feature(
					float_list=tf.train.FloatList(value=[value]))
			# Create the tf example object
			example = tf.train.Example(features=tf.train.Features(
				feature=feature))
			# Write out the example to the TFRecord file
			writer.write(example.SerializeToString())


def generate_tf_dataset(tf_record_path,learning_params,batch_size,
	n_epochs,norm_images=False,input_norm_path=None,kwargs_detector=None,
	log_learning_params=None):
	"""	Generate a TFDataset that a model can be trained with.

	Args:
		tf_record_paths (str, or [str,...]) A string specifying the paths to
			the TFRecords that will be used in the dataset. Can also be a list
			of strings for specifying multiple tf_record_paths.
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		batch_size (int): The batch size that will be used for training
		n_epochs (int): The number of training epochs. The dataset object will
			deal with iterating over the data for repeated epochs.
		norm_images (bool): If True, images will be normalized to have std 1.
		input_norm_path (str): The path to a csv that contains the
			normalization to be applied to the output parameters. If None
			no normalization will be applied.
		kwargs_detector (dict): A dictionary containing the detector kwargs
			used to generate the noise on the fly. If None no additional
			noise will be added.
		log_learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn the log of. Can
			be None.

	Returns:
		(tf.Dataset): A tf.Dataset object that returns the input image and the
		output labels.

	Notes:
		Do not use kwargs_detector if noise was already added during dataset
		generation.
	"""
	# Read the TFRecords
	raw_dataset = tf.data.TFRecordDataset(tf_record_path)

	# If normalization file is provided use it
	if input_norm_path is not None:
		norm_dict = pd.read_csv(input_norm_path,index_col='parameter')
	else:
		norm_dict = None

	# Load a noise model if requested.
	if kwargs_detector is not None:
		noise_function = kwargs_detector_to_tf_noise
	else:
		warnings.warn('No noise will be added')
		noise_function = None

	# Create the feature decoder that will be used
	def parse_image_features(example):  # pragma: no cover
		data_features = {
			'image': tf.io.FixedLenFeature([],tf.string),
			'height': tf.io.FixedLenFeature([],tf.int64),
			'width': tf.io.FixedLenFeature([],tf.int64),
			'index': tf.io.FixedLenFeature([],tf.int64),
		}
		# Set the log learning params to an empy list if no value is provided.
		if log_learning_params is None:
			log_learning_params_list = []
		else:
			log_learning_params_list = log_learning_params

		for param in learning_params+log_learning_params_list:
			data_features[param] = tf.io.FixedLenFeature([],tf.float32)
		parsed_dataset = tf.io.parse_single_example(example,data_features)
		image = tf.io.decode_raw(parsed_dataset['image'],out_type=float)
		image = tf.reshape(image,(parsed_dataset['height'],
			parsed_dataset['width'],1))

		# Add the noise using the baobab noise function (which is a tf graph)
		if noise_function is not None:
			image += noise_function(image,kwargs_detector)

		# If the images must be normed divide by the std
		if norm_images:
			image = image / tf.math.reduce_std(image)

		# Log the parameter if needed
		for param in log_learning_params_list:
			parsed_dataset[param] = tf.math.log(parsed_dataset[param])

		# Normalize if requested
		if norm_dict is not None:
			for param in learning_params+log_learning_params_list:
				parsed_dataset[param] -= norm_dict['mean'][param]
				parsed_dataset[param] /= norm_dict['std'][param]

		lens_param_values = tf.stack([parsed_dataset[param] for param in
			learning_params+log_learning_params_list])
		return image,lens_param_values

	# Select the buffer size to be slightly larger than the batch
	buffer_size = int(batch_size*1.2)

	# Set the feature decoder as the mapping function. Drop the remainder
	# in the case that batch_size does not divide the number of training
	# points exactly
	dataset = raw_dataset.map(parse_image_features,
		num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(
		n_epochs).shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(
		tf.data.experimental.AUTOTUNE)
	return dataset


def rotate_params_batch(learning_params,output,rot_angle):
	""" Rotate a batch of lensing parameters according to a specified rotation
	angle.

	Args:
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		output (np.array): A numpy array of dimension (batch_size,n_outputs)
			containing the true parameter values for each image in the batch.
			Note that n_outputs should be the same as len(learning_params).
		rot_angle (float): The angle to rotate the image by in radians.

	Notes:
		output is modified in place.
	"""

	# Alter the parameters. Hardcoded for now.
	def rotate_param(x,y,theta):
		return (x*np.cos(theta)-y*np.sin(theta),
			x*np.sin(theta)+y*np.cos(theta))

	if 'main_deflector_parameters_center_x' in learning_params:
		xi = learning_params.index('main_deflector_parameters_center_x')
		yi = learning_params.index('main_deflector_parameters_center_y')
		x,y = rotate_param(output[:,xi],output[:,yi],rot_angle)
		output[:,xi] = x
		output[:,yi] = y
	if 'main_deflector_parameters_e1' in learning_params:
		xi = learning_params.index('main_deflector_parameters_e1')
		yi = learning_params.index('main_deflector_parameters_e2')
		x,y = rotate_param(output[:,xi],output[:,yi],2*rot_angle)
		output[:,xi] = x
		output[:,yi] = y
	if 'main_deflector_parameters_gamma1' in learning_params:
		xi = learning_params.index('main_deflector_parameters_gamma1')
		yi = learning_params.index('main_deflector_parameters_gamma2')
		x,y = rotate_param(output[:,xi],output[:,yi],2*rot_angle)
		output[:,xi] = x
		output[:,yi] = y


def rotate_covariance_batch(learning_params,coavriance_batch,rot_angle):
	""" Rotate a batch of lensing parameters according to a specified rotation
	angle.

	Args:
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		coavriance_batch (np.array): A numpy array of dimension
			(batch_size,n_outputs,n_outputs) containing the covariance values
			for each image in the batch.
		rot_angle (float): The angle to rotate the image by in radians.

	Notes:
		coavriance_batch is modified in place.
	"""
	# Calculate rotation matrix. Hardcoded for now.
	def rotation_matrix(theta):
		return np.array([[np.cos(theta),-np.sin(theta)],
			[np.sin(theta),np.cos(theta)]])

	# For each possible pair of parameters, conduct the rotation.
	if 'main_deflector_parameters_center_x' in learning_params:
		xi = learning_params.index('main_deflector_parameters_center_x')
		yi = learning_params.index('main_deflector_parameters_center_y')
		# Ensure the parameters are next to each other. Otherwise this
		# indexing will fail.
		assert xi == yi - 1
		yi+=1
		rot_mat = rotation_matrix(rot_angle)
		coavriance_batch[:,xi:yi,xi:yi] = np.dot(rot_mat,
			np.dot(coavriance_batch[:,xi:yi,xi:yi],rot_mat.T).T).T

	if 'main_deflector_parameters_e1' in learning_params:
		xi = learning_params.index('main_deflector_parameters_e1')
		yi = learning_params.index('main_deflector_parameters_e2')
		# Ensure the parameters are next to each other.
		assert xi == yi - 1
		yi+=1
		rot_mat = rotation_matrix(2*rot_angle)
		coavriance_batch[:,xi:yi,xi:yi] = np.dot(rot_mat,
			np.dot(coavriance_batch[:,xi:yi,xi:yi],rot_mat.T).T).T

	if 'main_deflector_parameters_gamma1' in learning_params:
		xi = learning_params.index('main_deflector_parameters_gamma1')
		yi = learning_params.index('main_deflector_parameters_gamma2')
		assert xi == yi - 1
		yi+=1
		rot_mat = rotation_matrix(2*rot_angle)
		coavriance_batch[:,xi:yi,xi:yi] = np.dot(rot_mat,
			np.dot(coavriance_batch[:,xi:yi,xi:yi],rot_mat.T).T).T


def rotate_image_batch(image_batch,learning_params,output,rot_angle):
	""" Rotate a batch of strong lensing images and the corresponding lensing
	parameters

	Args:
		image_batch (np.array): A numpy image array of shape (batch_size,
			height,width,n_channels) that will be rotated.
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		output (np.array): A numpy array of dimension (batch_size,n_outputs)
			containing the true parameter values for each image in the batch.
			Note that n_outputs should be the same as len(learning_params).
		rot_angle (float): The angle to rotate the image by in radians.

	Returns:
		(np.array): A numpy array containing the rotated images.

	Notes:
		output is changed in place.
	"""

	# Rotate the image
	image_batch = rotate(image_batch,-rot_angle*180/np.pi,reshape=False,
		axes=(2,1))

	rotate_params_batch(learning_params,output,rot_angle)

	return image_batch


def generate_rotations_dataset(tf_record_path,learning_params,batch_size,
	n_epochs,norm_images=False,input_norm_path=None,kwargs_detector=None,
	log_learning_params=None):
	"""	Returns a generator that builds off of a TFDataset by adding random
	rotations to the images and parameters.

	Args:
		tf_record_paths (str, or [str,...]) A string specifying the paths to
			the TFRecords that will be used in the dataset. Can also be a list
			of strings for specifying multiple tf_record_paths.
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		batch_size (int): The batch size that will be used for training
		n_epochs (int): The number of training epochs. The dataset object will
			deal with iterating over the data for repeated epochs.
		norm_images (bool): If True, images will be normalized to have std 1.
		input_norm_path (str): The path to a csv that contains the
			normalization to be applied to the output parameters. If None
			no normalization will be applied.
		kwargs_detector (dict): A dictionary containing the detector kwargs
			used to generate the noise on the fly. If None no additional
			noise will be added.
		log_learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn the log of. Can
			be None.

	Returns:
		(generator): A generator that returns a tuple with the rotated image
		and the parameter values.
	"""
	# Create our base tf dataset without normalization
	base_dataset = generate_tf_dataset(tf_record_path,learning_params,
		batch_size,n_epochs,norm_images=norm_images,
		kwargs_detector=kwargs_detector,log_learning_params=log_learning_params)

	# If normalization file is provided use it
	if input_norm_path is not None:
		norm_dict = pd.read_csv(input_norm_path,index_col='parameter')
	else:
		norm_dict = None

	# If there are no log_learning_params, set it to an empty list
	if log_learning_params is None:
		log_learning_params = []

	def rotation_generator(dataset):
		# Iterate through the images and parameters in the dataset
		for image_batch, lens_param_batch in dataset:
			image_batch = image_batch.numpy()
			lens_param_batch = lens_param_batch.numpy()
			# Pick a rotation angle
			rot_angle = np.random.uniform()*2*np.pi
			# Conduct the rotation
			image_batch = rotate_image_batch(image_batch,
				learning_params+log_learning_params,lens_param_batch,rot_angle)
			if norm_dict is not None:
				for lpi, param in enumerate(learning_params+log_learning_params):
					lens_param_batch[:,lpi] -= norm_dict['mean'][param]
					lens_param_batch[:,lpi] /= norm_dict['std'][param]
			# Yield the rotated image and parameters
			yield image_batch, lens_param_batch

	# Return a rotation generator on our base dataset.
	return rotation_generator(base_dataset)
