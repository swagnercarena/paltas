# -*- coding: utf-8 -*-
"""
Generate simulated strong lensing images using the classes and parameters of
an input configuration dictionary.

This script generates strong lensing images from manada config dictionaries.
Example
-------
To run this script, pass in the desired config as argument::

	$ python -m generate.py path/to/config.py path/to/save_folder --n 1000

The parameters will be pulled from config.py and the images will be saved in
save_folder. If save_folder doesn't exist it will be created.
"""
# TODO: Variable noise
import numpy as np
import argparse, os, sys
from importlib import import_module
from manada.Sampling.sampler import Sampler
from manada.Utils.cosmology_utils import get_cosmology
from manada.Sources.galaxy_catalog import GalaxyCatalog
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from lenstronomy.LensModel.profile_list_base import ProfileListBase
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.SimulationAPI.data_api import DataAPI
from lenstronomy.Data.psf import PSF
from lenstronomy.SimulationAPI.observation_api import SingleBand
import lenstronomy.Util.util as util


def parse_args():
	"""Parse the input arguments by the user

	Returns:
		(argparse.Namespace): An instance of the Namespace object with the
			users provided values.

	"""
	# Initialize the parser and the possible inputs
	parser = argparse.ArgumentParser()
	parser.add_argument('config_dict', help='Path to manada configuration dict')
	parser.add_argument('save_folder', help='Folder to save images to')
	parser.add_argument('--n', default=1, dest='n', type=int,
		help='Size of dataset to generate (default 1)')
	parser.add_argument('--save_png_too', action='store_true',
		help='Also save a PNG for each image, for debugging')
	args = parser.parse_args()
	return args


def main():
	"""Generates the strong lensing images by drawing parameters values from
	the provided configuration dictionary.
	"""
	# Get the user provided arguments
	args = parse_args()

	# Get the dictionary from the provided .py file
	config_dir, config_file = os.path.split(os.path.abspath(args.config_dict))
	sys.path.insert(0, config_dir)
	config_name, _ = os.path.splitext(config_file)
	config_module = import_module(config_name)
	config_dict = config_module.config_dict

	# Set a random seed if provided
	if hasattr(config_module,'seed'):
		np.random.seed(config_module.seed)

	# Make the directory if not already there
	if not os.path.exists(args.save_folder):
		os.makedirs(args.save_folder)
	print("Save folder path: {:s}".format(args.save_folder))

	# Set up our sampler and draw an initial sample for initialization
	sampler = Sampler(config_dict)
	sample = sampler.sample()

	# Get the numerical kwargs numpix from the config
	kwargs_numerics = config_module.kwargs_numerics
	numpix = config_module.numpix

	# Set up the manada objects we'll use
	multi_plane = False
	if 'los' in config_dict:
		los_class = config_dict['los']['class'](sample['los_parameters'],
			sample['main_deflector_parameters'],sample['source_parameters'],
			sample['cosmology_parameters'])
		multi_plane = True
	if 'subhalo' in config_dict:
		subhalo_class = config_dict['subhalo']['class'](
			sample['subhalo_parameters'],sample['main_deflector_parameters'],
			sample['source_parameters'],sample['cosmology_parameters'])
	if 'main_deflector' in config_dict:
		main_model_list = config_dict['main_deflector']['models']

	source_class = config_dict['source']['class'](
		sample['cosmology_parameters'],sample['source_parameters'])

	# Use a pandas dataframe to store the parameter values.
	metadata = pd.DataFrame()
	metadata_path = os.path.join(args.save_folder,'metadata.csv')

	# Generate our images
	pbar = tqdm(total=args.n)
	nt = 0
	tries = 0
	while nt < args.n:
		# We always try
		tries += 1
		# Save the parameter values
		meta_values = {}
		# Draw our parameters
		sample = sampler.sample()
		z_lens = sample['main_deflector_parameters']['z_lens']
		z_source = sample['source_parameters']['z_source']
		cosmo = get_cosmology(sample['cosmology_parameters'])

		# Populate the list of models and kwargs we need
		complete_lens_model_list = []
		complete_lens_model_kwargs = []
		complete_z_list = []

		# Get the remaining observation kwargs from the sampler and
		# initialize our observational objects.
		kwargs_psf = sample['psf_parameters']
		kwargs_detector = sample['detector_parameters']
		psf_model = PSF(**kwargs_psf)
		data_api = DataAPI(numpix=numpix,**kwargs_detector)
		single_band = SingleBand(**kwargs_detector)

		# For each lensing object that's present, add them to the model and
		# kwargs list
		if 'los' in config_dict:
			los_class.update_parameters(
				sample['los_parameters'],sample['main_deflector_parameters'],
				sample['source_parameters'],sample['cosmology_parameters'])
			los_model_list, los_kwargs_list, los_z_list = los_class.draw_los()
			interp_model_list, interp_kwargs_list, interp_z_list = (
				los_class.calculate_average_alpha(numpix*2))
			complete_lens_model_list += los_model_list + interp_model_list
			complete_lens_model_kwargs += los_kwargs_list + interp_kwargs_list
			complete_z_list += los_z_list + interp_z_list
		if 'subhalo' in config_dict:
			subhalo_class.update_parameters(
				sample['subhalo_parameters'],
				sample['main_deflector_parameters'],
				sample['source_parameters'],sample['cosmology_parameters'])
			sub_model_list, sub_kwargs_list, sub_z_list = (
				subhalo_class.draw_subhalos())
			complete_lens_model_list += sub_model_list
			complete_lens_model_kwargs += sub_kwargs_list
			complete_z_list += sub_z_list
		if 'main_deflector' in config_dict:
			complete_lens_model_list += main_model_list
			# Get the parameters we need to pull from the main deflector list
			# from lenstronomy
			for model in main_model_list:
				p_names = ProfileListBase._import_class(model,None).param_names
				model_kwargs = {}
				for param in p_names:
					model_kwargs[param] = (
						sample['main_deflector_parameters'][param])
				complete_lens_model_kwargs += [model_kwargs]
			# All of the main deflector components are at z_lens
			complete_z_list += [z_lens]*len(main_model_list)

		complete_lens_model = LensModel(complete_lens_model_list, z_lens,
			z_source,complete_z_list, cosmo=cosmo.toAstropy(),
			multi_plane=multi_plane)

		# Now get the model and kwargs from the source class and turn it
		# into a source model
		source_class.update_parameters(
			cosmology_parameters=sample['cosmology_parameters'],
			source_parameters=sample['source_parameters'])
		# For catalog objects we also want to save the catalog index
		# and the (possibly randomized) additional rotation angle
		if isinstance(source_class,GalaxyCatalog):
			catalog_i, phi = source_class.fill_catalog_i_phi_defaults()
			meta_values['source_parameters_catalog_i'] = catalog_i
			meta_values['source_parameters_phi'] = phi
			source_model_list, source_kwargs_list = source_class.draw_source(
				catalog_i=catalog_i, phi=phi, z_new=z_source)
		else:
			source_model_list, source_kwargs_list = source_class.draw_source(
				z_new=z_source)
		source_light_model = LightModel(source_model_list)

		# Put it together into an image model
		complete_image_model = ImageModel(data_api.data_class, psf_model,
			complete_lens_model, source_light_model, None, None,
			kwargs_numerics=kwargs_numerics)

		# Generate our image with noise
		image = complete_image_model.image(complete_lens_model_kwargs,
			source_kwargs_list, None, None)

		# Check for magnification cut and apply
		if hasattr(config_module, 'mag_cut'):
			mag = np.sum(image)/source_light_model.total_flux(
				source_kwargs_list)
			if mag < config_module.mag_cut:
				continue

		if (not hasattr(config_module, 'magically_no_noise')
				or not config_module.magically_no_noise):
			image += single_band.noise_for_model(image)

		# Mask out an interior region of the image if requested
		if hasattr(config_module,'mask_radius'):
			x_grid, y_grid = util.make_grid(numPix=numpix,
			deltapix=kwargs_detector['pixel_scale'])
			r = util.array2image(np.sqrt(x_grid**2+y_grid**2))
			image[r<=config_module.mask_radius] = 0

		# Save the image and the metadata
		filename = os.path.join(args.save_folder, 'image_%07d' % nt)
		np.save(filename, image)
		if args.save_png_too:
			plt.imsave(filename + '.png', image)
		for component in sample:
			for key in sample[component]:
				meta_values[component+'_'+key] = sample[component][key]
		metadata = metadata.append(meta_values,ignore_index=True)

		# Write out the metadata every 20 images
		if nt == 0:
			# Sort the keys lexographically to ensure consistent writes
			metadata = metadata.reindex(sorted(metadata.columns), axis=1)
			metadata.to_csv(metadata_path, index=None)
			metadata = pd.DataFrame()
		elif nt%20 == 0:
			metadata.to_csv(metadata_path, index=None, mode='a',
				header=None)
			metadata = pd.DataFrame()

		nt += 1
		pbar.update()

	# Make sure anything left in the metadata DataFrame is written out
	metadata.to_csv(metadata_path, index=None, mode='a',header=None)
	pbar.close()
	print('Dataset generation complete. Acceptance rate: %.3f'%(args.n/tries))


if __name__ == '__main__':
	main()
