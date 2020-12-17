# -*- coding: utf-8 -*-
"""
Turn real galaxies into Lenstronomy inputs.

This module contains the default class for transforming the objects of a
source catalog into sources to be passed to lenstronomy.
"""
import numpy as np
from ..Utils.cosmology_utils import get_cosmology

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

	def __init__(self, cosmology_parameters):
		self.cosmo = get_cosmology(cosmology_parameters)

	def __len__(self):
		"""Returns the length of the catalog"""
		raise NotImplementedError

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
			lenstronomy kwargs.
		"""
		for catalog_i in self.sample_indices(n_galaxies,**selection_kwargs):
			yield self.lightmodel_kwargs(catalog_i, z_new=z_new)

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

	def lightmodel_kwargs(self, catalog_i, z_new=DEFAULT_Z):
		"""Creates lenstronomy interpolation lightmodel kwargs from
			a catalog image.

		Args:
			catalog_i (int): Index of image in catalog
			z_new (float): Redshift to place image at

		Returns:
			(dict) kwargs for
			lenstronomy.LightModel.Profiles.interpolation.Interpol
		"""
		img, metadata = self.image_and_metadata(catalog_i)
		z, pixel_width = metadata['z'], metadata['pixel_width']

		# With this, lenstronomy will preserve the scale/units of
		# the input image (in a configuration without lensing,
		# same pixel widths, etc.)
		img = img / pixel_width**2

		# Pixel length ~ angular diameter distance
		# (colossus uses funny /h units, but for ratios it
		#  fortunately doesn't matter)
		pixel_width *= (self.cosmo.angularDiameterDistance(z)
						/ self.cosmo.angularDiameterDistance(z_new))

		# Convert to kwargs for lenstronomy
		return dict(image=img,center_x=0,center_y=0,phi_G=0,scale=pixel_width)
