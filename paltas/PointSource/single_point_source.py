# -*- coding: utf-8 -*-
"""
Provides class for specifying a single point source

This module contains the class required to provide a single point source
as the source for paltas.
"""
from .point_source_base import PointSourceBase
from lenstronomy.Util.data_util import magnitude2cps


class SinglePointSource(PointSourceBase):
	"""Class to generate single point source model

	Args:
		point_source_parameters (dict): dictionary with source-specific 
			parameters.

	Notes:
		The required parameters are the x and y location of the source
		(x_point_source [float],y_point_source [float]), the magnitude
		(magnitude [float]), the absolute magnitude zeropoint of the detector
		(mag_zeropoint [float]), and wether or not to add time delays to the
		metadata (compute_time_delays [bool]). If adding time delays to metadata
		user must define kappa_ext.

		The optional parameters are a list of fractional magnification
		pertubations that will be applied to each image (mag_pert [float,...]),
		the external convergence used to calculate time delays(kappa_ext [float]),
		and the negative or positive error in days to add to the time delay
		calculation (time_delay_error [float]).

	"""

	required_parameters = ('x_point_source', 'y_point_source', 'magnitude',
		'output_ab_zeropoint', 'compute_time_delays')

	def draw_point_source(self):
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
		if('mag_pert' in self.point_source_parameters.keys()):
			point_source_kwargs['mag_pert'] = self.point_source_parameters[
				'mag_pert']

		# mag to amp conversion
		# note: flux = amplitude for point source
		point_source_kwargs['point_amp'] = magnitude2cps(
			self.point_source_parameters['magnitude'],
			self.point_source_parameters['output_ab_zeropoint'])

		return ['SOURCE_POSITION'], [point_source_kwargs]
