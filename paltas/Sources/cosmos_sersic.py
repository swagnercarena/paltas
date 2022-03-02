# -*- coding: utf-8 -*-
"""
Generate Lenstronomy inputs for a COSMOS galaxy + a Sersic Ellipse

This module contains a class that combines a COSMOS galaxy and a Sersic light 
distribution as the two sources for paltas.
"""
from .cosmos import COSMOSCatalog
from .sersic import SingleSersicSource


class COSMOSSersic(COSMOSCatalog):
	"""Class to Combine COSMOS galaxy with Sersic light source

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
	- 	mag_sersic - AB absolute magnitude of the sersic
	- 	output_ab_zeropoint - AB magnitude zeropoint of the detector
	- 	R_sersic - Sersic radius in units of arcseconds
	- 	n_sersic - Sersic index
	- 	e1_sersic - x-direction ellipticity eccentricity for sersic
	- 	e2_sersic - xy-direction ellipticity eccentricity for sersic
	- 	center_x_sersic - x-coordinate lens center for sersic in units of
		arcseconds
	- 	center_y_sersic - y-coordinate lens center for sersic in units of
		arcseconds
	- 	z_source - source redshift
	"""

	required_parameters = ('minimum_size_in_pixels','faintest_apparent_mag',
		'max_z','smoothing_sigma','cosmos_folder','random_rotation',
		'min_flux_radius','output_ab_zeropoint','z_source', 'mag_sersic',
		'center_x','center_y','R_sersic', 'n_sersic','e1_sersic', 'e2_sersic',
		'center_x_sersic','center_y_sersic')

	def draw_source(self, catalog_i=None, phi=None):
		"""Creates lenstronomy kwargs for a COSMOS catalog image and a 
		Sersic source

		Args:
			catalog_i (int): Index of image in catalog
			phi (float): Rotation to apply to the image.
				If not provided, use random or original rotation
				depending on source_parameters['random_rotation']

		Returns:
			(list,list) A list containing the models ['INTERPOL',
			'SERSIC_ELLIPSE'] and a list of two kwarg dicts for the
			instances of the lenstronomy classes

		Notes:
			If catalog_i is not provided, one that meets the cuts will be
			selected at random.
		"""
		model_list, kwargs_list = COSMOSCatalog.draw_source(self, catalog_i, phi)
		# look up difference between append & extend
		model_list.append('SERSIC_ELLIPSE')

		# create dict for sersic
		sersic_kwargs_dict = {}
		suffix = '_sersic'
		for param_name in self.__class__.required_parameters:
			if suffix in param_name:
				# R_sersic & n_sersic keep suffix
				if (param_name == 'R_sersic' or param_name == 'n_sersic'):
					sersic_kwargs_dict[param_name] = (
						self.source_parameters[param_name])
				else:
					sersic_kwargs_dict[param_name[:-7]] = (
						self.source_parameters[param_name])

		# mag to amp conversion
		sersic_kwargs_dict.pop('mag')
		sersic_kwargs_dict['amp'] = SingleSersicSource.mag_to_amplitude(
			self.source_parameters['mag_sersic'],
			self.source_parameters['output_ab_zeropoint'], sersic_kwargs_dict)

		kwargs_list.append(sersic_kwargs_dict)

		return model_list, kwargs_list
