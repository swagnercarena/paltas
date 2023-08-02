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
import h5py

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
	parser.add_argument('--h5', action='store_true',
		help='Save images as .h5 files rather than .npy')
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

	# Gather metadata in a list, will be written to dataframe later
	metadata_list = []
	metadata_path = os.path.join(args.save_folder,'metadata.csv')

	# Initialize our config handler
	config_handler = ConfigHandler(args.config_dict)

	# Generate our images
	pbar = tqdm(total=args.n)
	successes = 0
	tries = 0
	interim_image_array = []
	while successes < args.n:
		# We always try
		tries += 1

		# Attempt to draw our image
		image, metadata = config_handler.draw_image(new_sample=True)

		# Failed attempt if there is no image output
		if image is None:
			continue

		# Save the image and the metadata
		filename = os.path.join(args.save_folder, 'image_%07d' % successes)
		if not args.h5: np.save(filename, image)
		if args.save_png_too:
			plt.imsave(filename + '.png', image)

		metadata_list.append(metadata)

		# Write out the metadata every 20 images, and on the final write
		if len(metadata_list) > 20 or successes == args.n - 1:
			df = pd.DataFrame(metadata_list)
			# Sort the keys lexographically to ensure consistent writes
			df = df.reindex(sorted(df.columns), axis=1)
			first_write = successes <= len(metadata_list)
			df.to_csv(
				metadata_path,
				index=None,
				mode='w' if first_write else 'a',
				header=first_write)
			metadata_list = []
#
		successes += 1
		interim_image_array.append(image) 
		if args.h5:
		#Saves as h5 file every 100 images:
			if successes==1:
				with h5py.File(args.save_folder+'/image_data.h5', 'w') as hf:
					hf.create_dataset("data", data=np.array(interim_image_array),compression="gzip", maxshape=(None,np.array(interim_image_array).shape[1],np.array(interim_image_array).shape[2])) 
				interim_image_array=[]
			elif successes%100==0 or successes==args.n:
				interim_image_array = np.array(interim_image_array)
				with h5py.File(args.save_folder+'/image_data.h5', 'a') as hf:
				#Loads the h5 file, extends its shape, then appends the new images generated and saves the file:
					hf["data"].resize((hf["data"].shape[0] + interim_image_array.shape[0]), axis = 0)
					hf["data"][-interim_image_array.shape[0]:] = interim_image_array
				interim_image_array=[]
#
		pbar.update()

	# Make sure the list has been cleared out.
	assert not metadata_list
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
			metadata_path,tf_record_path,h5=args.h5)


if __name__ == '__main__':
	main()
