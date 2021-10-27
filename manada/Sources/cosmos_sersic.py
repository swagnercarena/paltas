# -*- coding: utf-8 -*-
"""
Generate Lenstronomy inputs for a COSMOS galaxy + a Sersic Ellipse

This module contains a class that combines a COSMOS galaxy and a Sersic light 
distribution as the two sources for manada.
"""
from .cosmos import COSMOSCatalog
from .sersic import SingleSersicSource


class COSMOSSersic(COSMOSCatalog):

		required_parameters = ('minimum_size_in_pixels','min_apparent_mag','max_z',
		'smoothing_sigma','cosmos_folder','random_rotation','min_flux_radius',
		'output_ab_zeropoint','z_source', 'mag_sersic', 'R_sersic', 'n_sersic',
		'e1_sersic', 'e2_sersic', 'center_x_sersic', 'center_y_sersic')

		def draw_source(self, catalog_i=None, phi=None):

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
							sersic_kwargs_dict[param_name] = self.source_parameters[param_name]
						else:
							sersic_kwargs_dict[param_name[:-7]] = (
								self.source_parameters[param_name])

				# mag to amp conversion
				sersic_kwargs_dict.pop('mag')
				sersic_kwargs_dict['amp'] = SingleSersicSource.mag_to_amplitude(
					self.source_parameters['mag_sersic'],
					self.source_parameters['output_ab_zeropoint'],
					sersic_kwargs_dict)

				kwargs_list.append(sersic_kwargs_dict)

				return model_list, kwargs_list
