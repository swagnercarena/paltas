# -*- coding: utf-8 -*-
"""
Provides class for specifying a single point source

This module contains the class required to provide a single point source
as the source for paltas.
"""
from .point_source_base import PointSourceBase
from lenstronomy.Util.data_util import magnitude2cps
from ..Utils.cosmology_utils import absolute_to_apparent


class SinglePointSource(PointSourceBase):
	"""Class to generate single point source model

	Args:
		point_source_parameters (dict): dictionary with source-specific 
			parameters.

	Notes:

	Required Parameters

	- 	magnitude - AB absolute magnitude of the point source
	-	x_point_source - x-coordinate lens center in units of arcseconds
	-	y_point_source - y-coordinate lens center in units of arcseconds
	-	output_ab_zeropoint - AB magnitude zeropoint of the detector
	- 	compute_time_delay - bool determining if time delays will be computed
		and added to the returned kwargs
    -   z_point_source - point source redshift

	Optional Parameters

	- 	mag_pert - list of 4 fractional magnification perturbations that will be
		applied to each point source image
	-	kappa_ext - external convergence used to calculate time delays
	-	time_delay_error - list of errors on the time delay measurements in units
		of days.
	"""

    # removing magnitude, but still requires one of: mag_abs or mag_app
	required_parameters = ('x_point_source', 'y_point_source',
		'output_ab_zeropoint', 'compute_time_delays', 'z_point_source')

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
		if 'mag_abs' in self.point_source_parameters.keys():
			mag_apparent = absolute_to_apparent(self.point_source_parameters['mag_abs'],
				self.point_source_parameters['z_source'],self.cosmo)
		elif 'mag_app' in self.point_source_parameters.keys():
			mag_apparent = self.point_source_parameters['mag_app']
		else:
			raise ValueError('Not all of the required parameters for the ' +
				'parameterization are present: missing mag_abs or mag_app')
		# note: flux = amplitude for point source
		point_source_kwargs['source_amp'] = magnitude2cps(mag_apparent,
			self.point_source_parameters['output_ab_zeropoint'])

		return ['SOURCE_POSITION'], [point_source_kwargs]