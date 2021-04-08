# -*- coding: utf-8 -*-
"""
Manipulate strong lensing images to use for tensorflow models.

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


def normalize_inputs(metadata,learning_params,input_norm_path):
	""" Normalize the inputs to the metadata

	Args:
		metadata(pd.DataFrame): A pandas object containing the metadata
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		input_norm_path (str): The path to a csv that contains the
			normalization to be applied to the output parameters. If the
			file already exists it will be read, if no file exists it will
			be written.

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
	else:
		print('Writing input normalization to %s'%(input_norm_path))
		norm_dict = pd.DataFrame(columns=['parameter','mean','std'])
		norm_dict['parameter'] = learning_params
		# Calculate the normalization for each parameter
		data = metadata[learning_params].to_numpy()
		norm_dict['mean'] = np.mean(data,axis=0)
		norm_dict['std'] = np.std(data,axis=0)
		# Set parameter to the index
		norm_dict = norm_dict.set_index('parameter')
		norm_dict.to_csv(input_norm_path)

	return norm_dict


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
	tf_record_path,input_norm_path=None):
	""" Generate a TFRecord file from a directory of numpy files.

	Args:
		root_path (str): The path to the folder containing the numpy files.
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		metadata_path (str):  The path to the csv file containing the
			image metadata.
		tf_record_path (str): The path to which the tf_record will be saved
		input_norm_path (str): The path to a csv that contains the
			normalization to be applied to the output parameters. If None
			no normalization, if the file already exists it will be read, if
			no file exists it will be written.
	"""
	# Pull the list of numpy filepaths from the directory
	npy_file_list = glob.glob(os.path.join(npy_folder,'image_*.npy'))
	# Open label csv
	metadata = pd.read_csv(metadata_path, index_col=None)

	# If normalization file is provided use / write it
	if input_norm_path is not None:
		norm_dict = normalize_inputs(metadata,learning_params,input_norm_path)
	else:
		norm_dict = None

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
				value = metadata[param][index]
				# Normalize if needed
				if norm_dict is not None:
					value -= norm_dict['mean'][param]
					value /= norm_dict['std'][param]
				# Write the feature
				feature[param] = tf.train.Feature(
					float_list=tf.train.FloatList(value=[value]))
			# Create the tf example object
			example = tf.train.Example(features=tf.train.Features(
				feature=feature))
			# Write out the example to the TFRecord file
			writer.write(example.SerializeToString())


def generate_tf_dataset(tf_record_path,learning_params,batch_size,
	n_epochs,norm_images=False,kwargs_detector=None):
	"""	Generate a TFDataset that a model can be trained with.

	Args:
		tf_record_paths (str) A string specifying the paths to the tf_records
			that will be used in the dataset.
		learning_params ([str,...]): A list of strings containing the
			parameters that the network is expected to learn.
		batch_size (int): The batch size that will be used for training
		n_epochs (int): The number of training epochs. The dataset object will
			deal with iterating over the data for repeated epochs.
		norm_images (bool): If True, images will be normalized to have std 1.
		kwargs_detector (dict): A dictionary containing the detector kwargs
			used to generate the noise on the fly. If None no additional
			noise will be added.

	Notes:
		Do not use kwargs_detector if noise was already added during dataset
		generation.
	"""
	# Read the TFRecords
	raw_dataset = tf.data.TFRecordDataset(tf_record_path)

	# Load a noise model from baobab using the baobab config file.
	if kwargs_detector is not None:
		noise_function = kwargs_detector_to_tf_noise
	else:
		warnings.warn('No noise will be added')
		noise_function = None

	# Create the feature decoder that will be used
	def parse_image_features(example):
		data_features = {
			'image': tf.io.FixedLenFeature([],tf.string),
			'height': tf.io.FixedLenFeature([],tf.int64),
			'width': tf.io.FixedLenFeature([],tf.int64),
			'index': tf.io.FixedLenFeature([],tf.int64),
		}
		for param in learning_params:
			data_features[param] = tf.io.FixedLenFeature([],tf.float32)
		parsed_dataset = tf.io.parse_single_example(example,data_features)
		image = tf.io.decode_raw(parsed_dataset['image'],out_type=float)
		image = tf.reshape(image,(parsed_dataset['height'],
			parsed_dataset['width'],1))
		# Add the noise using the baobab noise function (which is a tf graph)
		if noise_function is not None:
			image = noise_function(image,kwargs_detector)

		# If the images must be normed divide by the std
		if norm_images:
			image = image / tf.math.reduce_std(image)
		lens_param_values = tf.stack([parsed_dataset[param] for param in
			learning_params])
		return image,lens_param_values

	# Select the buffer size to be slightly larger than the batch
	buffer_size = int(batch_size*1.2)

	# Set the feature decoder as the mapping function. Drop the remainder
	# in the case that batch_size does not divide the number of training
	# points exactly
	dataset = raw_dataset.map(parse_image_features).repeat(n_epochs).shuffle(
		buffer_size=buffer_size).batch(batch_size)
	return dataset
