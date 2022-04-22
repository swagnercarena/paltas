# -*- coding: utf-8 -*-
"""
Generate simulated strong lensing images using the classes and parameters of
an input configuration dictionary.

This script generates strong lensing images from paltas config dictionaries.

Example
-------
To run this script, pass in the desired config as argument::

	$ python -m generate.py path/to/config.py path/to/save_folder --n 1000

The parameters will be pulled from config.py and the images will be saved in
save_folder. If save_folder doesn't exist it will be created.
"""
import numpy as np
import argparse, os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from paltas.Configs.config_handler import ConfigHandler


def parse_args():
	"""Parse the input arguments by the user

	Returns:
		(argparse.Namespace): An instance of the Namespace object with the
		users provided values.

	"""
	# Initialize the parser and the possible inputs
	parser = argparse.ArgumentParser()
	parser.add_argument('config_dict', help='Path to paltas configuration dict')
	parser.add_argument('save_folder', help='Folder to save images to')
	parser.add_argument('--n', default=1, dest='n', type=int,
		help='Size of dataset to generate (default 1)')
	parser.add_argument('--save_png_too', action='store_true',
		help='Also save a PNG for each image, for debugging')
	parser.add_argument('--tf_record', action='store_true',
		help='Generate the tf record for the training set.')
	args = parser.parse_args()
	return args


def main():
	"""Generates the strong lensing images by drawing parameters values from
	the provided configuration dictionary.
	"""
	# Get the user provided arguments
	args = parse_args()

	# Make the directory if not already there
	if not os.path.exists(args.save_folder):
		os.makedirs(args.save_folder)
	print("Save folder path: {:s}".format(args.save_folder))

	# Copy out config dict
	shutil.copy(
		os.path.abspath(args.config_dict),
		args.save_folder)

	# Use a pandas dataframe to store the parameter values.
	metadata_csv = pd.DataFrame()
	metadata_path = os.path.join(args.save_folder,'metadata.csv')

	# Initialize our config handler
	config_handler = ConfigHandler(args.config_dict)

	# Generate our images
	pbar = tqdm(total=args.n)
	nt = 0
	tries = 0
	while nt < args.n:
		# We always try
		tries += 1

		# Attempt to draw our image
		image, metadata = config_handler.draw_image(new_sample=True)

		# Failed attempt if there is no image output
		if image is None:
			continue

		# Save the image and the metadata
		filename = os.path.join(args.save_folder, 'image_%07d' % nt)
		np.save(filename, image)
		if args.save_png_too:
			plt.imsave(filename + '.png', image)

		metadata_csv = metadata_csv.append(metadata,ignore_index=True)

		# Write out the metadata every 20 images
		if nt == 0:
			# Sort the keys lexographically to ensure consistent writes
			metadata_csv = metadata_csv.reindex(sorted(metadata_csv.columns), axis=1)
			metadata_csv.to_csv(metadata_path, index=None)
			metadata_csv = pd.DataFrame()
		elif nt%20 == 0:
			metadata_csv = metadata_csv.reindex(sorted(metadata_csv.columns), axis=1)
			metadata_csv.to_csv(metadata_path, index=None, mode='a',
				header=None)
			metadata_csv = pd.DataFrame()

		nt += 1
		pbar.update()

	# Make sure anything left in the metadata_csv DataFrame is written out
	metadata_csv = metadata_csv.reindex(sorted(metadata_csv.columns), axis=1)
	metadata_csv.to_csv(metadata_path, index=None, mode='a',header=None)
	pbar.close()
	print('Dataset generation complete. Acceptance rate: %.3f'%(args.n/tries))

	# Generate tf record if requested. Save all the parameters and use default
	# filename data.tfrecord
	if args.tf_record:
		# Delayed import, triggers tensorflow import
		from paltas.Analysis import dataset_generation

		# The path to save the TFRecord to.
		tf_record_path = os.path.join(args.save_folder,'data.tfrecord')
		# Generate the list of learning parameters. Only save learning
		# parameters with associated float values.
		learning_params = []
		for key in metadata:
			if (isinstance(metadata[key],float) or
				isinstance(metadata[key],int)):
				learning_params.append(key)
		# Generate the TFRecord
		dataset_generation.generate_tf_record(args.save_folder,learning_params,
			metadata_path,tf_record_path)


if __name__ == '__main__':
	main()
