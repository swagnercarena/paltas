from pathlib import Path
import astropy
import astropy.table
from astropy.io import fits
import numpy as np
import numpy.lib.recfunctions
from tqdm import tqdm
from .galaxy_catalog import GalaxyCatalog

HUBBLE_ACS_PIXEL_WIDTH = 0.03   # Arcsec


class COSMOSCatalog(GalaxyCatalog):
	"""Interface to the COSMOS/GREAT3 23.5 magnitude catalog

	This is the catalog used for real galaxies in galsim, see
	https://github.com/GalSim-developers/GalSim/wiki/RealGalaxy%20Data.
	The catalog must be downloaded and unzipped.

	Args:
		folder (str): Path to the folder with the catalog files
		cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
	"""

	def __init__(self, folder, cosmology_parameters):
		super().__init__(cosmology_parameters)

		# Store the path as a Path object.
		self.folder = Path(folder)

		# Combine all partial catalog files
		catalogs = [unfits(str(self.folder / fn)) for fn in [
			'real_galaxy_catalog_23.5.fits',
			'real_galaxy_catalog_23.5_fits.fits'
		]]

		# Duplicate IDENT field crashes numpy's silly merge function.
		catalogs[1] = numpy.lib.recfunctions.drop_fields(catalogs[1], 'IDENT')

		# Custom fields
		catalogs += [
			np.zeros(len(catalogs[0]),
				dtype=[('size_x', np.int),('size_y', np.int),('z', np.float),
				('pixel_width', np.float)])]

		self.catalog = numpy.lib.recfunctions.merge_arrays(
			catalogs, flatten=True)

		self.catalog['pixel_width'] = HUBBLE_ACS_PIXEL_WIDTH
		self.catalog['z'] = self.catalog['zphot']

		# Loop over the images to find their sizes.
		# Why wasn't this just stored in the catalog?
		for img, meta in self.iter_image_and_metadata(
			message='Retrieving image sizes'):
			# Grab the shape of each image.
			meta['size_x'], meta['size_y'] = img.shape

	def __len__(self):
		return len(self.catalog)

	def sample_indices(self,n_galaxies,min_apparent_mag=None,
		minimum_size_in_pixels=None):
		"""Return n_galaxies array of catalog indices to sample

		Args:
			n_galaxies (int): Number of indices to return
			min_apparent_mag (float): minimum apparent magnitude of COSMOS
				image
			minimum_size_in_pixels (int): minimum image width and height
				in pixels

		Returns:
			(np.array): Array of ints of catalog indices to sample.
		"""
		is_ok = np.ones(len(self), dtype=np.bool_)
		if min_apparent_mag is not None:
			is_ok &= self.catalog['mag_auto'] < min_apparent_mag
		if minimum_size_in_pixels is not None:
			min_size = np.minimum(self.catalog['size_x'],
				self.catalog['size_y'])
			is_ok &= min_size >= minimum_size_in_pixels
		return np.random.choice(np.where(is_ok)[0],size=n_galaxies,
			replace=True)

	@staticmethod
	def _file_number(fn):
		"""Return integer X in blah_nX.fits filename fn.
		X can be more than one digit, not necessarily zero padded.
		"""
		return int(str(fn).split('_n')[-1].split('.')[0])

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
		catalog_i = 0
		_pattern = f'real_galaxy_images_23.5_n*.fits'  # noqa: F999
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

		Parameters:
			catalog_i (int): The catalog index

		Returns
			([np.array, np.void]) A numpy array containing the image
			metadata and a numpy void type that acts as a dictionary with
			the metadata.
		"""
		fn, index = self.catalog[catalog_i][['GAL_FILENAME', 'GAL_HDU']]
		fn = self.folder / fn.decode()  # 'real_galaxy_images_23.5_n1.fits'
		img = fits.getdata(fn, ext=index)
		# For some reason the COSMOS images are in big endian..??
		# This would cause some functions to fail, others to do weird things
		# silently! Let's cast it back to normal floats...
		img = img.astype(np.float)
		return img, self.catalog[catalog_i]


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
