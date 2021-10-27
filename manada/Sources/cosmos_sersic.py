# -*- coding: utf-8 -*-
"""
Generate Lenstronomy inputs for a COSMOS galaxy + a Sersic Ellipse

This module contains a class that combines a COSMOS galaxy and a Sersic light 
distribution as the two sources for manada.
"""
from .cosmos import COSMOSCatalog
from .sersic import SingleSersicSource


class COSMOSSersic(COSMOSCatalog):
<<<<<<< HEAD
  """Class to generate COSMOS Galaxy + Sersic model

  Args: 
    cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters: dictionary with source-specific parameters.
  """
    
  required_parameters = ('minimum_size_in_pixels','min_apparent_mag','max_z',
	'smoothing_sigma','cosmos_folder','random_rotation','min_flux_radius',
	'output_ab_zeropoint','z_source', 'mag_sersic', 'R_sersic', 'n_sersic', 
  'e1_sersic', 'e2_sersic', 'center_x_sersic', 'center_y_sersic')

  def draw_source(self, catalog_i=None, phi=None):
    """Return lenstronomy LightModel kwargs

		Returns:
		(list,list) A list containing the model names(s), and
			a list containing the model kwargs dictionaries.
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
        if ( param_name == 'R_sersic' or param_name == 'n_sersic') :
          sersic_kwargs_dict[param_name] = self.source_parameters[param_name]
        else :
          sersic_kwargs_dict[param_name[:-7]] = self.source_parameters[param_name]

    # mag to amp conversion
    sersic_kwargs_dict.pop('mag')
    sersic_kwargs_dict['amp'] = SingleSersicSource.mag_to_amplitude(
      self.source_parameters['mag_sersic'], 
      self.source_parameters['output_ab_zeropoint'], sersic_kwargs_dict)

    kwargs_list.append(sersic_kwargs_dict)

    return model_list, kwargs_list
=======

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
>>>>>>> 55c9e1352de287a01a46156b96f83641cb77f97a
