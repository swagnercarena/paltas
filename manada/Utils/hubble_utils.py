# -*- coding: utf-8 -*-
"""
Utility functions for producing Hubble drizzled images.
"""
import copy
import numpy as np
from astropy.io import fits
from astropy.wcs import wcs
from drizzle.drizzle import Drizzle
from scipy.interpolate import RectBivariateSpline


def image_grid(shape, pixel_width,x0=0, y0=0):
	"""Return x, y edges of image coordinates

	Args:
		shape (tuple): The shape of the image
		pixel_scale (float): pixel width in units of arcseconds
		x0 (float): The center x coordinate of the grid
		y0 (float): The center y coordiante of the grid
		edges (bool)
	"""
	nx, ny = shape
	dx = nx * pixel_width
	dy = nx * pixel_width
	x = np.linspace(-dx / 2, dx / 2, nx) + x0
	y = np.linspace(-dy / 2, dy / 2, ny) + y0
	return x, y


def offset_wcs(w,pixel_offset):
	"""Return wcs w shifted by pixel_offset pixels.

	Args:
		w (astropy.wcs.wcs.WCS): An istance of the WCS class that will
			be copied and offset.
		pixel_offset (tuple): The x,y offset to apply in pixel units.

	Returns:
		(astropy.wcs.wcs.WCS): The offset WCS object.
	"""
	# Copy the object and add the pixel offset to the reference pixel
	# position.
	w_out = copy.deepcopy(w)
	w_out.wcs.crpix += np.array(pixel_offset)
	return w_out


def distort_image(img_high_res,w_high_res,w_dither,offset_pattern):
	"""Create distorted Hubble image in _flt frame

	Args:
	 - sky_image: ndarray, undistorted, noise-free high-res image
		 of the sky in electrons/s
	 - w: WCS of *output* image (including any
		 distortions you'd like to apply)
	 - pixel_width_sky: Width in arcsec of a pixel in sky_image
	 - pixel_offset: Position  of the center of the image
		 in pixels, relative to the center (crval/crpix) of the wcs
		 the crval of distorted_wcs
	"""
	try:
		out_shape = (w.naxis2, w.naxis1)
	except AttributeError:
		out_shape = tuple(w._naxis)

	# Coordinate grid of the sky image
	x_sky, y_sky = image_grid(
		sky_image.shape,
		# Convert from arcseconds to degrees
		pixel_width=pixel_width_sky / 3600,
		edges=False)
	center_sky = w.wcs.crval
	x_sky += center_sky[0]
	y_sky += center_sky[1]

	# Interpolator mapping sky position to image value
	img_sky_itp = RectBivariateSpline(x_sky, y_sky, sky_image)

	# Coordinate grid of the _flt pixel image
	# (the one we are generating)
	out_img_shape = out_shape
	center_pixel = w.wcs.crpix
	x_pixel = np.arange(out_img_shape[0])
	y_pixel = np.arange(out_img_shape[1])
	x_pixel += int(center_pixel[0]) - len(x_pixel)//2 - pixel_offset[0]
	y_pixel += int(center_pixel[1]) - len(y_pixel)//2 - pixel_offset[1]

	# Compute image in distorted frame
	xx, yy = np.meshgrid(x_pixel, y_pixel, indexing='ij')
	pixel_on_sky = w.all_pix2world(np.stack([xx.ravel(), yy.ravel()]).T, 1).T
	img_out = img_sky_itp(*pixel_on_sky, grid=False).reshape(out_img_shape)

	return img_out


def generate_downsampled_wcs(high_res_shape,high_res_pixel_scale,
		low_res_pixel_scale,wcs_distortion):
	"""Generate a wcs mapping onto the specified lower resolution. If
	specified, geometric distortions will be included in the wcs object.

	Args:
		high_res_shape (tuple): The shape of the high resolution image
		high_res_pixel_scale (float): The pixel width of the high
			resolution image in units of arcseconds.
		low_res_pixel_scale (float): The pixel width of the lower
			resolution target image in units of arcseconds.
		wcs_distortion (astropy.wcs.wcs.WCS): An instance of the
			WCS class that includes the desired gemoetric distortions as
			SIP coefficients. Note that the parameters relating to
			pixel scale, size of the image, and reference pixels will be
			overwritten. If None no gemoetric distortions will be applied.

	Returns:
		astropy.wcs.wcs.WCS: The wcs object corresponding to the
			downsampled image.
	"""
	# If wcs distortion was specified, use it to seed the fits file we
	# will later use to create the wcs object. This is a little
	# circular, but it's difficult to modify a wcs once it's been
	# created.
	if wcs_distortion is not None:
		hdul = wcs_distortion.to_fits()
	else:
		hdul = fits.HDUList([fits.PrimaryHDU()])

	# Get the shape of the expected data
	scaling = high_res_pixel_scale/low_res_pixel_scale
	low_res_shape = copy.copy(high_res_shape)
	low_res_shape[0] = low_res_shape[0]//scaling
	low_res_shape[1] = low_res_shape[1]//scaling

	# Modify the fits header to match our desired low resolution
	# configuration.
	hdul[0].header['WCSAXES'] = 2
	hdul[0].header['CRPIX1'] = (low_res_shape[0]/2,
		'Pixel coordinate of reference point')
	hdul[0].header['CRPIX2'] = (low_res_shape[1]/2,
		'Pixel coordinate of reference point')
	hdul[0].header['CDELT1'] = (low_res_pixel_scale/3600,
		'[deg] Coordinate increment at reference point')
	hdul[0].header['CDELT2'] = (low_res_pixel_scale/3600,
		'[deg] Coordinate increment at reference point')
	hdul[0].header['CUNIT1'] = ('deg',
		'Units of coordinate increment and value ')
	hdul[0].header['CUNIT2'] = ('deg',
		'Units of coordinate increment and value ')
	if wcs_distortion:
		hdul[0].header['CTYPE1'] = 'RA-SIP'
		hdul[0].header['CTYPE2'] = 'DEC-SIP'
	else:
		hdul[0].header['CTYPE1'] = 'RA'
		hdul[0].header['CTYPE2'] = 'DEC'
	hdul[0].header['CRVAL1'] = (20,
		'[deg] Coordinate value at reference point')
	hdul[0].header['CRVAL2'] = (-70,
		'[deg] Coordinate value at reference point')

	# Use the first object to generate our WCS object
	return wcs.WCS(fobj=hdul,header=hdul[0].header)


def hubblify(img_high_res,high_res_pixel_scale,detector_pixel_scale,
	drizzle_pixel_scale,noise_model,psf_model,offset_pattern,
	wcs_distortion=None):
	"""Generates a simulated drizzled HST image, accounting for
	gemoetric distortions, the dithering pattern, and correlated
	read noise from drizzling.

	Args:
		img_high_res (np.array): A high resolution image that will be
			downsampled to produce each of the dithered images.
		high_res_pixel_scale (float): The pixel width of the
			high resolution image in units of arcseconds.
		detector_pixel_scale (float): The pixel width of the detector
			in units of arcseconds.
		drizzle_pixel_scale (float): The pixel width of the final
			drizzled product in units of arcseconds.
		noise_model (function): A function that maps from an input
			numpy array of the image to a realization of the noise. This
			should operate on the degraded images.
		psf_model (function): A function that maps from an input image
			to an output psf-convovled image. This should operate on the
			degraded images.
		offset_pattern ([tuple,...]): A list of x,y coordinate pairs
			specifying the offset of each dithered image from the coordinate
			frame used to generate the high resolution image. Specifying shifts
			that place the degraded image outside of the high resolution image
			will cause interpolation errors.
		wcs_distortion (astropy.wcs.wcs.WCS): An instance of the
			WCS class that includes the desired gemoetric distortions.
			Note that the parameters relating to pixel scale, size of
			the image, and reference pixels will be overwritten. If None
			no gemoetric distortions will be applied.
	"""
	# Create our three base WCS systems, one for the input high res, one for the
	# input highres, one for the drizzled image, and one for the final dithered
	# image.
	w_high_res = generate_downsampled_wcs(img_high_res.shape,
		high_res_pixel_scale,high_res_pixel_scale,None)
	w_driz = generate_downsampled_wcs(img_high_res.shape,high_res_pixel_scale,
		drizzle_pixel_scale,None)
	w_dither = generate_downsampled_wcs(img_high_res.shape,
		high_res_pixel_scale,detector_pixel_scale,wcs_distortion)

	# Initialize our drizzle class with the target output wcs.
	driz = Drizzle(outwcs=w_driz)

	# Get the distorted sub images to which we will add noise and then add
	# them to our drizzle.
	img_dither_array = distort_image(img_high_res,w_high_res,w_dither,
		offset_pattern)

	for d_i,image_dither in enumerate(img_dither_array):

		# Add the psf and the noise to each dithered image.
		image_dither = psf_model(image_dither)
		image_dither += noise_model(image_dither)

		# Feed to drizzle, note WCS translation
		driz.add_image(image_dither,offset_wcs(w_dither, d_i))

	# Get final image from drizzle
	return driz.outsci
