# -*- coding: utf-8 -*-
"""
Utility functions for simplifying interactions with lenstronomy classes.
"""
from lenstronomy.ImSim.Numerics.numerics_subframe import NumericsSubFrame


class PSFHelper():
	"""Class for simplifying interactions with the lenstronomy PSF class
	when using 2d images generated outside lenstronomy.

	Args:
		data_class (lenstronomy.Data.imaging_data.ImageData): An instance of
			the lenstronomy ImageData class, likely spawned from the
			lenstronomy DataAPI class. It will be used to define the grid
			onto which the PSF will be mapped.
		psf_model (lenstronomy.Data.psf.PSF): An instance of the lenstronomy
			PSF class that will be used to convolve the image.
		kwargs_numerics (dict): A dict containing the numerics kwargs to
			pass to lenstronomy
	"""

	def __init__(self,data_class,psf_model,kwargs_numerics):
		# Just initialize the NumericsSubframe object that we will later call
		# to perform the convolutions
		psf_model.set_pixel_size(data_class.pixel_width)
		self.image_numerics = NumericsSubFrame(pixel_grid=data_class,
			psf=psf_model,**kwargs_numerics)
		self.data_class = data_class

	def psf_model(self,image):
		"""Apply the psf model to the given image.

		Args:
			image (np.array): The image that will be convolved with the psf.
				It must have dimensions and resolution matching those specified
				in the production of the data_class provided when PSFHelper was
				initialized.

		Returns:
			(np.array): The image convolved with the psf.
		"""
		# Store the image shape for later
		im_shape = image.shape

		# Convolve the image in the 1d format required by the lenstronomy
		# class.
		conv_class = self.image_numerics.convolution_class

		if conv_class is None:
			# The psf type was NONE so no convolution should be done
			return image
		else:
			conv_flat = conv_class.convolution2d(image)
			# Reshape the image to be 2d if a 1d image was returned.
			return conv_flat.reshape(im_shape)
