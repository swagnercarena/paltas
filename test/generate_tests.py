import numpy as np
import pandas as pd
import unittest
import glob, os
from paltas import generate

try:
	import tensorflow as tf
	tensorflow_installed = True
except ImportError:
	tensorflow_installed = False

# Define the cosmos path
cosmos_folder = 'test_data/cosmos/'


class GenerateTests(unittest.TestCase):

	def setUp(self):
		# Set the random seed so we don't run into trouble
		np.random.seed(20)

	def test_main(self):
		# Test that the main function makes some images
		output_folder = 'test_data/test_dataset'
		n_generate = 21
		generate.generate_from_config(
			config_path='test_data/config_dict.py',
			save_folder=output_folder,
			n=n_generate,
			save_png_too=True)

		image_file_list = glob.glob(os.path.join(output_folder,'image_*.npy'))

		self.assertEqual(len(image_file_list),n_generate)

		# Make sure all of the files are readable and have the correct size
		# for the config
		for image_file in image_file_list:
			img = np.load(image_file)
			self.assertTupleEqual(img.shape,(64,64))
			os.remove(image_file)

		# Make sure the metadata makes sense
		metadata = pd.read_csv(os.path.join(output_folder,'metadata.csv'))
		self.assertEqual(len(metadata),n_generate)
		self.assertListEqual(list(
			metadata['cosmology_parameters_cosmology_name']),
			['planck18']*n_generate)
		self.assertListEqual(list(
			metadata['detector_parameters_pixel_scale']),[0.08]*n_generate)
		self.assertListEqual(list(metadata['subhalo_parameters_c_0']),
			[18.0]*n_generate)
		self.assertListEqual(list(metadata['main_deflector_parameters_z_lens']),
			[0.5]*n_generate)
		self.assertListEqual(list(metadata['source_parameters_z_source']),
			[1.5]*n_generate)
		self.assertGreater(np.std(metadata['source_parameters_catalog_i']),0)
		self.assertListEqual(list(metadata['psf_parameters_fwhm']),
			[0.1]*n_generate)
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

		# Remove the images
		for i in range(n_generate):
			os.remove(os.path.join(output_folder,'image_%07d.png' % i))

		cleanup_cosmos_cache()
		os.rmdir(output_folder)


	def test_main_drizzle(self):
		# Test that the main function makes some images
		output_folder = 'test_data/test_dataset'
		generate.generate_from_config(
			config_path='test_data/config_dict_drizz.py',
			save_folder=output_folder,
			n=10,
			tf_record=tensorflow_installed)

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
		if tensorflow_installed:
			os.remove(os.path.join(output_folder,'data.tfrecord'))

		cleanup_cosmos_cache()
		os.rmdir(output_folder)


def cleanup_cosmos_cache():
	"""Clean up the test cosmos cache"""
	test_cosmo_folder = 'test_data/cosmos/'
	os.remove(test_cosmo_folder+'paltas_catalog.npy')
	for i in range(10):
		os.remove(test_cosmo_folder+'npy_files/img_%d.npy'%(i))
	os.rmdir(test_cosmo_folder+'npy_files')
