# -*- coding: utf-8 -*-
"""
Turn real galaxies into Lenstronomy inputs.

This module contains the default class for transforming the objects of a
source catalog into sources to be passed to lenstronomy.
"""
import numpy as np
from ..Utils.cosmology_utils import get_cosmology
import copy

DEFAULT_Z = 2.


class GalaxyCatalog:
	"""Base class for turning real galaxy images into Lenstronomy inputs.

	Args:
		cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	"""
	required_parameters = ('random_rotation',)

	def __init__(self, cosmology_parameters, source_parameters):
		self.cosmo = get_cosmology(cosmology_parameters)
		self.source_parameters = copy.deepcopy(source_parameters)

		# Check that all the required parameters are present
		self.check_parameterization(GalaxyCatalog.required_parameters)

	def __len__(self):
		"""Returns the length of the catalog"""
		raise NotImplementedError

	def check_parameterization(self,required_params):
		""" Check that all the required parameters are present in the
		source_parameters.

		Args:
			required_params ([str,...]): A list of strings containing the
				required parameters.
		"""
		if not all(elem in self.source_parameters.keys() for
			elem in required_params):
			raise ValueError('Not all of the required parameters for the ' +
				'parameterization are present.')

	def update_parameters(self,cosmology_parameters=None,
		source_parameters=None):
		"""Updated the class parameters

		Args:
			cosmology_parameters (str,dict, or
				colossus.cosmology.cosmology.Cosmology): Either a name
				of colossus cosmology, a dict with 'cosmology name': name of
				colossus cosmology, an instance of colussus cosmology, or a
				dict with H0 and Om0 ( other parameters will be set to
				defaults).
			source_parameters (dict): A dictionary containing all the parameters
				needed to draw sources.
		"""
		if source_parameters is not None:
			self.source_parameters.update(source_parameters)
		if cosmology_parameters is not None:
			self.cosmo = get_cosmology(cosmology_parameters)

	def image_and_metadata(self, catalog_i):
		"""Returns the image array and metadata for one galaxy

		Parameters:
			catalog_i (int): The catalog index

		Returns
			([np.array, np.void]) A numpy array containing the image
			metadata and a numpy void type that acts as a dictionary with
			the metadata.
		"""
		raise NotImplementedError

	def iter_lightmodel_kwargs_samples(self,n_galaxies,z_new=DEFAULT_Z,
		**selection_kwargs):
		"""Yields dicts of lenstronomy LightModel kwargs for n_galaxies,
		placed at redshift z_new

		Args:
			n_galaxies (int): Number of galaxies to draw
			z_new (float): Redshift to place galaxies at
			**selection_kwargs: Kwargs that can be passed to the
				sample_indices method.

		Returns:
			(generator): A generator that can be iterated over to give
			lenstronomy model lists and kwargs.
		"""
		for catalog_i in self.sample_indices(n_galaxies,**selection_kwargs):
			yield self.draw_source(catalog_i, z_new=z_new)

	def iter_image_and_metadata(self, message=''):
		"""Yields the image array and metadata for all of the images
		in the catalog.

		Args:
			message (str): If the iterator uses tqdm, this message
				will be displayed.

		Returns:
			(generator): A generator that can be iterated over to give
			lenstronomy kwargs.
		"""
		for catalog_i in range(len(self)):
			yield self.image_and_metadata(catalog_i)

	def sample_indices(self, n_galaxies):
		"""Returns n_galaxies array of ints, catalog indices to sample

		Args:
			n_galaxies (int): Number of indices to return

		Returns:
			(np.array): Array of ints to sample
		"""
		return np.random.randint(0, len(self), size=n_galaxies)

	def draw_source(self, catalog_i=None, z_new=DEFAULT_Z):
		"""Creates lenstronomy interpolation lightmodel kwargs from
			a catalog image.

		Args:
			catalog_i (int): Index of image in catalog
			z_new (float): Redshift to place image at

		Returns:
			(list,list) A list containing the model ['INTERPOL'] and
				the kwargs for an instance of the class
				lenstronomy.LightModel.Profiles.interpolation.Interpol

		Notes:
			If not catalog_i is provided, one that meets the cuts will be
			selected at random.
		"""
		# If no index is provided pick one at random
		if catalog_i is None:
			catalog_i = self.sample_indices(1)
		img, metadata = self.image_and_metadata(catalog_i)
		pixel_width = metadata['pixel_width']

		# With this, lenstronomy will preserve the scale/units of
		# the input image (in a configuration without lensing,
		# same pixel widths, etc.)
		img = img / pixel_width**2

		pixel_width *= self.z_scale_factor(metadata['z'], z_new)

		# Convert to kwargs for lenstronomy
		return (
			['INTERPOL'],
			[dict(
				image=img,
				center_x=0, center_y=0,
				phi_G=self.draw_phi(),
				scale=pixel_width)])

	def draw_phi(self, old_phi=0.):
		if self.source_parameters['random_rotation']:
			phi = np.random.rand() * 2 * np.pi
		else:
			phi = 0
		return (phi + old_phi) % (2 * np.pi)

	def z_scale_factor(self, z_old, z_new):
		"""Return multiplication factor for object/pixel size
		for moving its redshift from z_old to z_new.
		"""
		# Pixel length ~ angular diameter distance
		# (colossus uses funny /h units, but for ratios it
		#  fortunately doesn't matter)
		return (
			self.cosmo.angularDiameterDistance(z_old)
			/ self.cosmo.angularDiameterDistance(z_new))
