# -*- coding: utf-8 -*-
"""
Turn real galaxies into Lenstronomy inputs.

This module contains the default class for transforming the objects of a
source catalog into sources to be passed to lenstronomy.
"""
import numpy as np
from .source_base import SourceBase
from ..Utils.cosmology_utils import absolute_to_apparent, get_k_correction
from lenstronomy.Util.data_util import magnitude2cps


class GalaxyCatalog(SourceBase):
	"""Base class for turning real galaxy images into Lenstronomy inputs.

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters (dict): A dictionary containing all the parameters
			needed to draw sources (in this case random_rotation).

	Notes:

	Required Parameters

	- 	random_rotation - boolean dictating if COSMOS sources will be rotated
		randomly when drawn
	- 	output_ab_zeropoint - AB magnitude zeropoint of the detector
	- 	z_source - source redshift
	- 	center_x - x-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	center_y- y-coordinate lens center for COSMOS source in units of
		arcseconds

	Optional Parameters

	-	source_absolute_magnitude - AB absolute magnitude of the source. The
		light from the catalog galaxy will be rescaled to match this
		magnitude.
	"""
	required_parameters = ('random_rotation','output_ab_zeropoint',
		'z_source','center_x','center_y')
	optional_parameters = ('source_absolute_magnitude')
	# This parameter must be set by class inheriting GalaxyCatalog
	ab_zeropoint = None

	def __len__(self):
		"""Returns the length of the catalog"""
		raise NotImplementedError

	def check_parameterization(self, required_params):
		""" Check that all the required parameters are present in the
		source_parameters. Also checks ab_zeropoint is set.

		Args:
			required_params ([str,...]): A list of strings containing the
				required parameters.
		"""
		# Run the base check
		super().check_parameterization(required_params)
		# Check ab magnitude zeropoint has been set
		if (self.__class__.__name__ != 'GalaxyCatalog' and
			self.__class__.ab_zeropoint is None):
			raise ValueError('ab_zeropoint must be set by class inheriting '+
				'GalaxyCatalog.')

	def image_and_metadata(self, catalog_i):
		"""Returns the image array and metadata for one galaxy

		Args:
			catalog_i (int): The catalog index

		Returns
			([np.array, np.void]) A numpy array containing the image
			metadata and a numpy void type that acts as a dictionary with
			the metadata.
		"""
		raise NotImplementedError

	def iter_lightmodel_kwargs_samples(self,n_galaxies,z_new,
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
			yield self.draw_source(catalog_i)[:2]

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

	def fill_catalog_i_phi_defaults(self, catalog_i=None, phi=None):
		"""Return catalog index and source rotation angle.

		Args:
			catalog_i (int): Index of image in catalog
				If not provided or None, will be sampled from catalog.
			phi (float): Rotation to apply to the image.
				If not provided or None, will either randomize or use 0,
				depending on source_parameters['random_rotation'].
		"""
		# If no index is provided pick one at random
		if catalog_i is None:
			catalog_i = self.sample_indices(1).item()
		# If no rotation is provided, pick one at random or use original
		# orientation
		if phi is None:
			if self.source_parameters['random_rotation']:
				phi = self.draw_phi()
			else:
				phi = 0
		return catalog_i, phi

	def draw_source(self, catalog_i=None, phi=None):
		"""Creates lenstronomy lightmodel kwargs from a catalog image.

		Args:
			catalog_i (int): Index of image in catalog
			z_new (float): Redshift to place image at
			phi (float): Rotation to apply to the image.
				If not provided, randomize or use 0, depending on
				source_parameters['random_rotation']

		Returns:
			(list,list,list): A list containing the model ['INTERPOL'],
			the kwargs for an instance of the class
			lenstronomy.LightModel.Profiles.interpolation.Interpol,
			and the redshift of the model.

		Notes:
			If not catalog_i is provided, one that meets the cuts will be
			selected at random.
		"""
		catalog_i, phi = self.fill_catalog_i_phi_defaults(catalog_i, phi)
		img, metadata = self.image_and_metadata(catalog_i)
		pixel_width = metadata['pixel_width']
		z_new = self.source_parameters['z_source']

		# With this, lenstronomy will preserve the scale/units of
		# the input image (in a configuration without lensing,
		# same pixel widths, etc.)
		img = img / pixel_width**2

		# Take into account the difference in the magnitude zeropoints
		# of the input survey and the output survey. Note this doesn't
		# take into account the color of the object!
		img *= 10**((self.source_parameters['output_ab_zeropoint']-
			self.__class__.ab_zeropoint)/2.5)

		pixel_width *= self.z_scale_factor(metadata['z'], z_new)

		# Apply the k correction to the image from the redshifting
		self.k_correct_image(img,metadata['z'],z_new)

		# If a desired absolute magnitude was specified, scale the image
		# accordingly
		if 'source_absolute_magnitude' in self.source_parameters:
			mag_apparent = absolute_to_apparent(
				self.source_parameters['source_absolute_magnitude'],
				self.source_parameters['z_source'],self.cosmo)
			self.normalize_to_mag(img,mag_apparent,
				self.source_parameters['output_ab_zeropoint'],pixel_width)

		# Convert to kwargs for lenstronomy
		return (
			['INTERPOL'],
			[dict(
				image=img,
				center_x=self.source_parameters['center_x'],
				center_y=self.source_parameters['center_y'],
				phi_G=phi,
				scale=pixel_width)],
			[z_new])

	@staticmethod
	def k_correct_image(image,z_original,z_new):
		"""Apply the k-correction to the image at the pixel level.

		Args:
			image (np.array): The image that needs to be k-corrected
			z_original (float): The original redshift of the object
			z_new (float): The new redshift of the object

		Notes:
			image will be changed in place
		"""
		# Calculate the k-correction for the change in redshift
		mag_k_correct = get_k_correction(z_new) - get_k_correction(z_original)

		# Apply the correction
		image *= 10**(-mag_k_correct/2.5)

	@staticmethod
	def normalize_to_mag(image,mag_apparent,mag_zero_point,pixel_width):
		"""Renormalizes and image so that it has the specified apparent
		magnitude.

		Args:
			image (np.array): Image to be renormalized
			mag_apparent (float): TThe desired apparent magnitude
			mag_zero_point (float): The magnitude zero-point of the detector
			pixel_width (float): The width of a pixel in arcseconds
		Notes:
			Image changed in place.
		"""
		# Get the total flux of the image
		flux_total = np.sum(image) * pixel_width**2

		# Get the cps of an image of the desired magnitude
		flux_true = magnitude2cps(mag_apparent, mag_zero_point)

		# Rescale the image so that both values match
		image *= flux_true/flux_total

	@staticmethod
	def draw_phi():
		"""Draws a random rotation angle for the interpolation of the source.

		Returns:
			(float): The new angle to use in the interpolation class.
		"""
		return np.random.rand() * 2 * np.pi

	def z_scale_factor(self, z_old, z_new):
		"""Return multiplication factor for object/pixel size for moving its
		redshift from z_old to z_new.

		Args:
			z_old (float): The original redshift of the object.
			z_new (float): The redshift the object will be placed at.

		Returns:
			(float): The multiplicative pixel size.
		"""
		# Pixel length ~ angular diameter distance
		# (colossus uses funny /h units, but for ratios it
		#  fortunately doesn't matter)
		return (
			self.cosmo.angularDiameterDistance(z_old)
			/ self.cosmo.angularDiameterDistance(z_new))

	def draw(self, result, *, sample, lens_light, **kwargs):
		# For catalog objects we also want to save the catalog index
		# and the (possibly randomized) additional rotation angle. We will
		# therefore push these back into the sample object.
		catalog_i, phi = self.fill_catalog_i_phi_defaults()
		if lens_light:
			result.add_lens_light(
				*self.draw_source(catalog_i=catalog_i, phi=phi))
			sample['lens_light_parameters']['catalog_i'] = catalog_i
			sample['lens_light_parameters']['phi'] = phi
		else:
			result.add_sources(
				*self.draw_source(catalog_i=catalog_i, phi=phi))
			sample['source_parameters']['catalog_i'] = catalog_i
			sample['source_parameters']['phi'] = phi
