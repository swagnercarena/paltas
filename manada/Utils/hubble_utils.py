from copy import deepcopy
import os
import tempfile
from astropy.coordinates.attributes import Attribute

import numpy as np

from astropy.io import fits
from astropy.wcs import wcs

from drizzle.drizzle import Drizzle

from scipy.interpolate import RectBivariateSpline

WFC3_PIXEL_WIDTH = 0.04


def image_grid(
		shape, pixel_width=WFC3_PIXEL_WIDTH,
		x0=0, y0=0, edges=True):
	"""Return x, y edges of image coordinates
	Args:
		- pixel_width: pixel width (in whatever units you want, e.g. arcsec)
		- x0, y0: center of grid
		- edges: If True (default), returns pixel edges.
			If False, returns pixel centers
	"""
	nx, ny = shape
	dx = nx * pixel_width
	dy = nx * pixel_width
	extra = 1 if edges else 0
	x = np.linspace(-dx / 2, dx / 2, nx + extra) + x0
	y = np.linspace(-dy / 2, dy / 2, ny + extra) + y0
	return x, y


def distort_image(
		sky_image: np.ndarray,
		w: wcs.WCS,
		pixel_width_sky=WFC3_PIXEL_WIDTH / 2,
		pixel_offset=(0,0)):
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


def offset_wcs(w: wcs.WCS, pixel_offset=(0,0)):
	"""Return wcs w shifted by pixel_offset pixels
		in some possibly consistent direction
	"""
	if pixel_offset == (0,0):
		w_out = w
	else:
		w_out = deepcopy(w)
		w_out.wcs.crpix += np.array(pixel_offset)
	return w_out


def sky_bg(img,
		   pixel_width=WFC3_PIXEL_WIDTH / 2,
		   exposure_sec=500,
		   bg_per_as2_sec=20.778):
	"""Return sky background in electrons/pixel given an image img
	"""
	return np.random.poisson(
		bg_per_as2_sec * pixel_width**2 * exposure_sec,
		size=img.shape)


def simple_wcs(
		pixel_width=WFC3_PIXEL_WIDTH,
		npix=64,
		center=(337.5202808, -20.83333306),
		offset=(0,0)):
	"""Return basic RA-DEC WCS

	Args:
	 - pixel_width: width of pixel in arcseconds
	 - npix: length (and width) of image in pixels
	 - center: center in RA-DEC coordinates
	 - offset: offset to center pixel in pixels
	"""
	arcsec_to_deg = 1/3600
	wcs_input_dict = {
		'CTYPE1': 'RA-TAN',
		'CTYPE2': 'DEC-TAN',

		'CUNIT1': 'deg',
		'CUNIT2': 'deg',

		'CDELT1': -pixel_width * arcsec_to_deg,
		'CDELT2': pixel_width * arcsec_to_deg,

		'CRPIX1': 1 + offset[0],
		'CRPIX2': 1 + offset[1],

		# Just some standard reference location
		'CRVAL1': center[0],
		'CRVAL2': center[1],

		'NAXIS1': npix,
		'NAXIS2': npix
	}
	return wcs.WCS(wcs_input_dict)


def hubblify(
		img_sky,
		w_flt=None,
		w_drz=None,
		pixel_width_sky=WFC3_PIXEL_WIDTH / 2,
		exposure_sec=500):
	"""Return WFC3 UVIS Hubble image given an undistorted sky image;
		applying distortions, an empirical PSF, and drizzling for
		a four-point dither observation pattern.

	Args:
	 - sky_image: ndarray, undistorted, noise-free high-res image
		 of the sky in electrons/s
	 - w_flt: astropy.wcs.WCS of single exposure in _flt/_flc frame
	 - w_drz: astropy.wcs.WCS of final, drizzled image
	 - pixel_width_sky: Width in arcsec of a pixel in sky_image
	 - exposure_sec: Exposure. Total exposure is four times this
		 due to the four-point dither/drizzle pattern.
	"""
	if w_flt is None:
		w_flt = simple_wcs(WFC3_PIXEL_WIDTH / 2, npix=128)
	if w_drz is None:
		w_drz = simple_wcs(WFC3_PIXEL_WIDTH, npix=64)

	driz = Drizzle(outwcs=w_drz)
	for offset in [(0,0), (0, .5), (.5, 0), (.5, .5)]:
		img_sky_with_noise = (
			img_sky * exposure_sec   # Convert from e/s to e
			+ sky_bg(img_sky, pixel_width_sky, exposure_sec))

		# Construct image in _flt / _flc frame via interpolation
		# This should produce an image with WFC3's pixel size.
		img_in_flt = distort_image(img_sky_with_noise, w_flt, pixel_width_sky)

		# TODO: Convolve with empirical PSF

		# TODO: Apply readout noise

		# Feed to drizzle, note WCS translation
		driz.add_image(
			img_in_flt,
			offset_wcs(w_flt, offset))

	# Get final image from drizzle
	try:
		_, tmp_fn = tempfile.mkstemp()
		driz.write(tmp_fn)
		with fits.open(tmp_fn) as hdul:
			return hdul[1].data
	finally:
		os.remove(tmp_fn)
