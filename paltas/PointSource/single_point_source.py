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
		Required parameters: 
			x_point_source (float),
			y_point_source (float),
			magnitude (float),
			mag_zeropoint (float): absolute magnitude zeropoint of detector
			compute_time_delays (bool): will add time delays to metadata if 
				True. Must define kappa_ext (see below) if True.
		Optional parameters: 
			mag_pert (list of floats): List of fractional magnification 
				pertubations that will be applied to each image.
			kappa_ext (float): External convergence used to calculate time 
				delays. If compute_time_delays = True, this parameter must be 
				defined.
			time_delay_error (float): error in days will be added to time delay
				calculation (can be negative or positive error)

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
