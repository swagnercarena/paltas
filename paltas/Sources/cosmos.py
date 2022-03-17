# -*- coding: utf-8 -*-
"""
Turn COSMOS galaxies into Lenstronomy inputs.

This module contains the default class for transforming the objects of the
COSMOS catalog into sources to be passed to lenstronomy.
"""
from pathlib import Path
import astropy
import astropy.table
from astropy.io import fits
from lenstronomy.Util.param_util import phi_q2_ellipticity
import numpy as np
import numpy.lib.recfunctions
from tqdm import tqdm
from .galaxy_catalog import GalaxyCatalog
import scipy

HUBBLE_ACS_PIXEL_WIDTH = 0.03   # Arcsec


class COSMOSCatalog(GalaxyCatalog):
	"""Class to interface to the COSMOS/GREAT3 23.5 magnitude catalog

	This is the catalog used for real galaxies in galsim, see
	https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data.
	The catalog must be downloaded and unzipped.

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters (dict): A dictionary containing all the parameters
			needed to draw sources.

	Notes:

	Required Parameters

	- 	minimum_size_in_pixels - smallest cutout size for COSMOS sources in pixels
	- 	faintest_apparent_mag - faintest apparent AB magnitude for COSMOS source
	- 	max_z - highest redshift for COSMOS source
	- 	smoothing_sigma - smoothing kernel to apply to COSMOS source in pixels
	- 	cosmos_folder - path to COSMOS images
	- 	random_rotation - boolean dictating if COSMOS sources will be rotated
		randomly when drawn
	- 	min_flux_radius - minimum half-light radius for COSMOS source in pixels
	- 	center_x - x-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	center_y- y-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	output_ab_zeropoint - AB magnitude zeropoint of the detector
	- 	z_source - source redshift
	"""
	required_parameters = ('minimum_size_in_pixels','faintest_apparent_mag',
		'max_z','smoothing_sigma','cosmos_folder','random_rotation',
		'min_flux_radius','output_ab_zeropoint','z_source','center_x','center_y')
	# Average AB magnitude zeropoint for the COSMOS run.
	ab_zeropoint = 25.95

	def __init__(self, cosmology_parameters, source_parameters):
		super().__init__(cosmology_parameters,source_parameters)

		# Store the path as a Path object.
		self.folder = Path(source_parameters['cosmos_folder'])

		# Check if we've already populated the catalog
		self.catalog_path = self.folder/'paltas_catalog.npy'
		self.npy_files_path = self.folder/'npy_files'
		if self.catalog_path.exists() and self.npy_files_path.exists():
			self.catalog = np.load(str(self.catalog_path))

		else:
			# Make the directory where I'm going to save the npy files
			self.npy_files_path.mkdir()
			# Combine all partial catalog files
			catalogs = [unfits(str(self.folder / fn)) for fn in [
				'real_galaxy_catalog_23.5.fits',
				'real_galaxy_catalog_23.5_fits.fits'
			]]

			# Duplicate IDENT field crashes numpy's silly merge function.
			catalogs[1] = numpy.lib.recfunctions.drop_fields(catalogs[1],
				'IDENT')

			# Custom fields
			catalogs += [
				np.zeros(len(catalogs[0]),
					dtype=[('size_x', np.int),('size_y', np.int),('z', float),
					('pixel_width', float)])]

			self.catalog = numpy.lib.recfunctions.merge_arrays(
				catalogs, flatten=True)

			self.catalog['pixel_width'] = HUBBLE_ACS_PIXEL_WIDTH
			self.catalog['z'] = self.catalog['zphot']

			# Loop over the images to find their sizes.
			catalog_i = 0
			for img, meta in self.iter_image_and_metadata_bulk(
				message='One-time COSMOS startup'):
				# Grab the shape of each image.
				meta['size_x'], meta['size_y'] = img.shape
				# Save the image as its own image.
				img = img.astype(np.float)
				np.save(str(self.npy_files_path/('img_%d.npy'%(catalog_i))),img)
				catalog_i += 1

			np.save(self.catalog_path,self.catalog)

	def __len__(self):
		return len(self.catalog)

	def _passes_cuts(self):
		""" Return a boolean mask of the sources that pass the source
		parameter cuts.

		Returns:
			(np.array): Array of bools of catalog indices to use for
				sampling.
		"""
		is_ok = np.ones(len(self), dtype=np.bool_)
		# Grab the parameter to cut on.
		faintest_apparent_mag = self.source_parameters['faintest_apparent_mag']
		minimum_size_in_pixels = self.source_parameters['minimum_size_in_pixels']
		max_z = self.source_parameters['max_z']
		min_flux_radius = self.source_parameters['min_flux_radius']

		# Get the images that match the cuts.
		if faintest_apparent_mag is not None:
			is_ok &= self.catalog['mag_auto'] < faintest_apparent_mag
		if minimum_size_in_pixels is not None:
			min_size = np.minimum(self.catalog['size_x'],
				self.catalog['size_y'])
			is_ok &= min_size >= minimum_size_in_pixels
		if max_z is not None:
			is_ok &= self.catalog['z'] < max_z
		if min_flux_radius is not None:
			is_ok &= self.catalog['flux_radius'] > min_flux_radius

		return is_ok

	def sample_indices(self,n_galaxies):
		"""Return n_galaxies array of catalog indices to sample

		Args:
			n_galaxies (int): Number of indices to return

		Returns:
			(np.array): Array of ints of catalog indices to sample.

		Notes:
			The minimum apparent magnitude, minimum size in pixels, and
			minimum redshift are all set by the source parameters dict.
		"""
		is_ok = self._passes_cuts()
		return np.random.choice(np.where(is_ok)[0],size=n_galaxies,
			replace=True)

	@staticmethod
	def _file_number(fn):
		"""Return integer X in blah_nX.fits filename fn.
		X can be more than one digit, not necessarily zero padded.
		"""
		return int(str(fn).split('_n')[-1].split('.')[0])

	def iter_image_and_metadata_bulk(self, message=''):
		"""Yields the image array and metadata for all of the images
		in the catalog.

		Args:
			message (str): If the iterator uses tqdm, this message
				will be displayed.

		Returns:
			(generator): A generator that can be iterated over to give
			lenstronomy kwargs.

		Notes:
			This will read the fits files.
		"""
		catalog_i = 0
		_pattern = f'real_galaxy_images_23.5_n*.fits'  # noqa: F541, F999
		files = list(sorted(self.folder.glob(_pattern),
			key=self._file_number))

		# Iterate over all the matching files.
		for fn in tqdm(files, desc=message):
			with fits.open(fn) as hdul:
				for img in hdul:
					yield img.data, self.catalog[catalog_i]
					catalog_i += 1

	def image_and_metadata(self, catalog_i):
		"""Returns the image array and metadata for one galaxy

		Args:
			catalog_i (int): The catalog index

		Returns:
			([np.array, np.void]) A numpy array containing the image
			metadata and a numpy void type that acts as a dictionary with
			the metadata.

		Notes:
			This will read the numpy files made during initialization. This is
			much faster on average than going for the fits files.
		"""
		img = np.load(str(self.npy_files_path/('img_%d.npy'%(catalog_i))))
		smoothing_sigma = self.source_parameters['smoothing_sigma']
		# Filter the images if that was requested
		if smoothing_sigma > 0:
			img = scipy.ndimage.gaussian_filter(img,
				sigma=smoothing_sigma/HUBBLE_ACS_PIXEL_WIDTH)
		return img, self.catalog[catalog_i]


class COSMOSSersicCatalog(COSMOSCatalog):
	"""Class identical to COSMOSCatalog, but produces the best-fit single
	elliptic Sersic profiles instead of real galaxy images

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters (dict): A dictionary containing all the parameters
			needed to draw sources.

	Notes:

	Required Parameters

	- 	minimum_size_in_pixels - smallest cutout size for COSMOS sources in pixels
	- 	faintest_apparent_mag - faintest apparent AB magnitude for COSMOS source
	- 	max_z - highest redshift for COSMOS source
	- 	smoothing_sigma - smoothing kernel to apply to COSMOS source in pixels
	- 	cosmos_folder - path to COSMOS images
	- 	random_rotation - boolean dictating if COSMOS sources will be rotated
		randomly when drawn
	- 	min_flux_radius - minimum half-light radius for COSMOS source in pixels
	- 	center_x - x-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	center_y- y-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	output_ab_zeropoint - AB magnitude zeropoint of the detector
	- 	z_source - source redshift
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

		_fit_results = self.catalog['sersicfit'].astype(np.float)
		self.sercic_info = {
			p: _fit_results[:, i]
			for i, p in enumerate(
				'intensity r_half n q boxiness x0 y0 phi'.split())}
		# Convert half-light radius from pixels to arcseconds
		self.sercic_info['r_half'] *= HUBBLE_ACS_PIXEL_WIDTH

	def draw_source(self, catalog_i=None, phi=None):
		"""Creates lenstronomy interpolation lightmodel kwargs from
		a catalog image.

		Args:
			catalog_i (int): Index of image in catalog
			z_new (float): Redshift to place image at
			phi (float): Rotation to apply to the image.
				If not provided, use random or original rotation
				depending on source_parameters['random_rotation']

		Returns:
			(list,list) A list containing the model ['INTERPOL'] and
			the kwargs for an instance of the class
			lenstronomy.LightModel.Profiles.interpolation.Interpol

		Notes:
			If not catalog_i is provided, one that meets the cuts will be
			selected at random.
		"""
		catalog_i, phi = self.fill_catalog_i_phi_defaults(catalog_i, phi)
		metadata = self.catalog[catalog_i]
		z_new = self.source_parameters['z_source']

		z_scaling = self.z_scale_factor(metadata['z'], z_new)

		# Get sercic info for this particular galaxy
		sercic_info = {
			p: self.sercic_info[p][catalog_i]
			for p in self.sercic_info}

		# Convert (phi, q) -> (e1, e2), after applying possible random rotation
		# Using py_func to avoid numba caching trouble
		# (and it's a trivial func anyway)
		e1, e2 = phi_q2_ellipticity.py_func(
			(sercic_info['phi'] + phi) % (2 * np.pi),
			sercic_info['q'])

		# Convert to kwargs for lenstronomy
		return (
			['SERSIC_ELLIPSE'],
			[dict(
				# Scale by pixel area before z-scaling, as in parent draw_source
				amp=sercic_info['intensity'] / metadata['pixel_width'] ** 2,
				e1=e1,
				e2=e2,
				R_sersic=sercic_info['r_half'] * z_scaling,
				n_sersic=sercic_info['n'])])

	def image_and_metadata(self, catalog_i):
		"""Returns the image array and metadata for one galaxy

		Args:
			catalog_i (int): The catalog index

		Returns:
			([np.array, np.void]) A numpy array containing the image
			metadata and a numpy void type that acts as a dictionary with
			the metadata.

		Notes:
			This will read the numpy files made during initialization. This is
			much faster on average than going for the fits files.
		"""
		raise NotImplementedError

	def iter_image_and_metadata_bulk(self, message=''):
		"""Yields the image array and metadata for all of the images
		in the catalog.

		Args:
			message (str): If the iterator uses tqdm, this message
				will be displayed.

		Returns:
			(generator): A generator that can be iterated over to give
			lenstronomy kwargs.

		Notes:
			This will read the fits files.
		"""
		raise NotImplementedError


class COSMOSExcludeCatalog(COSMOSCatalog):
	"""	Class identical to COSMOSCatalog, but allows for specific source indexes
	to be excluded from the analysis.

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters (dict): A dictionary containing all the parameters
			needed to draw sources.

	Notes:

	Required Parameters

	- 	minimum_size_in_pixels - smallest cutout size for COSMOS sources in pixels
	- 	faintest_apparent_mag - faintest apparent AB magnitude for COSMOS source
	- 	max_z - highest redshift for COSMOS source
	- 	smoothing_sigma - smoothing kernel to apply to COSMOS source in pixels
	- 	cosmos_folder - path to COSMOS images
	- 	random_rotation - boolean dictating if COSMOS sources will be rotated
		randomly when drawn
	- 	min_flux_radius - minimum half-light radius for COSMOS source in pixels
	- 	center_x - x-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	center_y- y-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	output_ab_zeropoint - AB magnitude zeropoint of the detector
	- 	z_source - source redshift
	-	source_exclusion_list - A list of COSMOS source indices to exclude,
		even if they pass the other cuts.
	"""

	required_parameters = ('minimum_size_in_pixels','faintest_apparent_mag',
		'max_z','smoothing_sigma','cosmos_folder','random_rotation',
		'min_flux_radius','source_exclusion_list','output_ab_zeropoint',
		'center_x','center_y','z_source')

	def __init__(self, cosmology_parameters, source_parameters):
		super().__init__(cosmology_parameters,source_parameters)

	def _passes_cuts(self):
		""" Return a boolean mask of the sources that pass the source
		parameter cuts.

		Returns:
			(np.array): Array of bools of catalog indices to use for
				sampling.
		"""
		is_ok = super()._passes_cuts()

		# Drop any catalog indices that are in the exclusion list
		is_ok &= np.invert(np.in1d(np.arange(len(self)),
			self.source_parameters['source_exclusion_list']))

		return is_ok


class COSMOSIncludeCatalog(COSMOSCatalog):
	"""Class identical to COSMOSExcludeCatalog, but only allows for specific
	source indeces to be included for the analysis.

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology):
			Either a name of colossus cosmology, a dict with 'cosmology name':
			name of colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters (dict): A dictionary containing all the parameters
			needed to draw sources.

		Notes:

	Required Parameters

	- 	minimum_size_in_pixels - smallest cutout size for COSMOS sources in pixels
	- 	faintest_apparent_mag - faintest apparent AB magnitude for COSMOS source
	- 	max_z - highest redshift for COSMOS source
	- 	smoothing_sigma - smoothing kernel to apply to COSMOS source in pixels
	- 	cosmos_folder - path to COSMOS images
	- 	random_rotation - boolean dictating if COSMOS sources will be rotated
		randomly when drawn
	- 	min_flux_radius - minimum half-light radius for COSMOS source in pixels
	- 	center_x - x-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	center_y- y-coordinate lens center for COSMOS source in units of
		arcseconds
	- 	output_ab_zeropoint - AB magnitude zeropoint of the detector
	- 	z_source - source redshift
	-	source_inclusion_list - A list of COSMOS source indices to include,
		cutting any source not in this list.
	"""

	required_parameters = ('minimum_size_in_pixels','faintest_apparent_mag',
		'max_z','smoothing_sigma','cosmos_folder','random_rotation',
		'min_flux_radius','source_inclusion_list','output_ab_zeropoint',
		'center_x','center_y','z_source')

	def __init__(self, cosmology_parameters, source_parameters):
		super().__init__(cosmology_parameters,source_parameters)

	def _passes_cuts(self):
		""" Return a boolean mask of the sources that pass the source
		parameter cuts.

		Returns:
			(np.array): Array of bools of catalog indices to use for
				sampling.
		"""
		is_ok = super()._passes_cuts()

		# Drop any catalog indices that are in the exclusion list
		is_ok &= np.in1d(np.arange(len(self)),
			self.source_parameters['source_inclusion_list'])

		return is_ok


def unfits(fn, pandas=False):
	"""Returns numpy record array from fits catalog file fn

	Args:
		fn (str): filename of fits file to load
		pandas (bool): If True, return pandas DataFrame instead of an array

	Returns:
		(np.array):
	"""
	if pandas:
		astropy.table.Table.read(fn, format='fits').to_pandas()
	else:
		with fits.open(fn) as hdul:
			data = hdul[1].data
			# Remove fitsyness from record array
			return np.array(data, dtype=data.dtype)
