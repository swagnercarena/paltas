# -*- coding: utf-8 -*-
"""
Provides class for specifying a single point source

This module contains the class required to provide a single point source
as the source for manada.
"""
from .point_source_base import PointSourceBase
from lenstronomy.Util.data_util import magnitude2cps


class SinglePointSource(PointSourceBase):
    """Class to generate single point source model

	Args:
		point_source_parameters: dictionary with source-specific parameters.
	"""

    required_parameters = ('x_point_source', 'y_point_source', 'magnitude', 
        'mag_zeropoint')

    def draw_point_source(self) :
        """Return lenstronomy PointSource kwargs

		Returns:
			(list,list) A list containing the model names(s), and
				a list containing the model kwargs dictionaries.
		"""

        point_source_kwargs = {}
        point_source_kwargs['ra_source'] = self.point_source_parameters[
            'x_point_source']
        point_source_kwargs['dec_source'] = self.point_source_parameters[
            'y_point_source']
        
        # flux = amplitude for point source
        point_source_kwargs['point_amp'] = magnitude2cps(
            self.point_source_parameters['magnitude'], 
            self.point_source_parameters['mag_zeropoint'])

        return ['SOURCE_POSITION'], [point_source_kwargs]