# -*- coding: utf-8 -*-
"""
Generate Lenstronomy inputs for a COSMOS galaxy + a Sersic Ellipse

This module contains a class that combines a COSMOS galaxy and a Sersic light 
distribution as the two sources for paltas.
"""
from .cosmos import COSMOSCatalog
from .sersic import SingleSersicSource


class COSMOSSersic(COSMOSCatalog):
	"""Combines COSMOS galaxy with Sersic light source

	Args:
		cosmology_parameters (str,dict, or colossus.cosmology.Cosmology): 
			Either a name of colossus cosmology, a dict with 'cosmology name': name
			of colossus cosmology, an instance of colussus cosmology, or a dict 
			with H0 and Om0 (other parameters will be set to defaults)
		source_parameters (dict): A dictionary containing all the parameters 
			needed to draw sources.

	"""

	required_parameters = ('minimum_size_in_pixels','min_apparent_mag','max_z',
		'smoothing_sigma','cosmos_folder','random_rotation','min_flux_radius',
		'output_ab_zeropoint','z_source', 'mag_sersic', 'R_sersic', 'n_sersic',
		'e1_sersic', 'e2_sersic', 'center_x_sersic', 'center_y_sersic')

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
