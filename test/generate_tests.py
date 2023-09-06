import numpy as np
import pandas as pd
import unittest
import sys, glob, copy, os
import h5py
from paltas import generate
import h5py
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

	def metadata_check(self,metadata,n_generate):
		# Function to check that the metadata gives reasonable values
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

	def test_main(self):
		# Test that the main function makes some images
		old_sys = copy.deepcopy(sys.argv)
		n_generate = 21
		output_folder = 'test_data/test_dataset'
		sys.argv = ['test','test_data/config_dict.py',output_folder,'--n',
			str(n_generate),'--save_png_too']
		generate.main()

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
		self.metadata_check(metadata,n_generate)

		# Remove the metadata file
		os.remove(os.path.join(output_folder,'metadata.csv'))
		os.remove(os.path.join(output_folder,'config_dict.py'))

		# Remove the images
		for i in range(n_generate):
			os.remove(os.path.join(output_folder,'image_%07d.png' % i))

		sys.argv = old_sys

		# Also clean up the test cosmos cache
		test_cosmo_folder = 'test_data/cosmos/'
		os.remove(test_cosmo_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(test_cosmo_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(test_cosmo_folder+'npy_files')
		os.rmdir(output_folder)

	def test_main_drizzle(self):
		# Test that the main function makes some images
		old_sys = copy.deepcopy(sys.argv)
		output_folder = 'test_data/test_dataset'
		sys.argv = ['test','test_data/config_dict_drizz.py',output_folder,
			'--n','10']
		if tensorflow_installed:
			sys.argv.append('--tf_record')
		generate.main()

		image_file_list = glob.glob(os.path.join(output_folder,'image_*.npy'))

		self.assertEqual(len(image_file_list),n_generate)

		# Make sure all of the files are readable and have the correct size
		# for the config
		for image_file in image_file_list:
			img = np.load(image_file)
			self.assertTupleEqual(img.shape,(85,85))
			os.remove(image_file)

		# Make sure the metadata makes sense
		metadata = pd.read_csv(os.path.join(output_folder,'metadata.csv'))
		self.metadata_check(metadata,n_generate)

		# Remove the metadata file
		os.remove(os.path.join(output_folder,'metadata.csv'))
		os.remove(os.path.join(output_folder,'config_dict_drizz.py'))
		if tensorflow_installed:
			os.remove(os.path.join(output_folder,'data.tfrecord'))

		sys.argv = old_sys

		# Also clean up the test cosmos cache
		test_cosmo_folder = 'test_data/cosmos/'
		os.remove(test_cosmo_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(test_cosmo_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(test_cosmo_folder+'npy_files')
		os.rmdir(output_folder)

	def test_h5(self):
		# Test that the h5 kwarg generates the required h5 files
		old_sys = copy.deepcopy(sys.argv)
		n_generate = 101
		output_folder = 'test_data/test_dataset'
		sys.argv = ['test','test_data/config_dict.py',output_folder,
			'--n',str(n_generate),'--h5']
		if tensorflow_installed:
			sys.argv.append('--tf_record')
		generate.main()

		image_file_list = glob.glob(os.path.join(output_folder,'*.h5'))
		# Check only 1 h5 file is made:
		self.assertEqual(len(image_file_list),1)
		with h5py.File(image_file_list[0],'r') as f0:
			image_data = f0['data'][()]
		image_shape = image_data.shape
		self.assertEqual(image_shape[0],n_generate)

		# Make sure all of the files are readable and have the correct size
		# for the config
		for n in range(n_generate):
			img = image_data[n,:,:]
			self.assertTupleEqual(img.shape,(64,64))

		# Make sure the metadata makes sense
		metadata = pd.read_csv(os.path.join(output_folder,'metadata.csv'))
		self.metadata_check(metadata,n_generate)

		# Remove the metadata file
		os.remove(os.path.join(output_folder,'metadata.csv'))
		os.remove(os.path.join(output_folder,'config_dict.py'))
		os.remove(os.path.join(output_folder,'data.tfrecord'))
		os.remove(image_file_list[0])
		sys.argv = old_sys

		# Also clean up the test cosmos cache
		test_cosmo_folder = 'test_data/cosmos/'
		os.remove(test_cosmo_folder+'paltas_catalog.npy')
		for i in range(10):
			os.remove(test_cosmo_folder+'npy_files/img_%d.npy'%(i))
		os.rmdir(test_cosmo_folder+'npy_files')
