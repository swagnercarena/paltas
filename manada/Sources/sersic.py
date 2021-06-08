from copy import copy

from .source_base import SourceBase, DEFAULT_Z_SOURCE


class SingleSersicSource(SourceBase):
	"""Class to generate single Sersic profile light models

	Args:
		cosmology_parameters (str,dict, or
			colossus.cosmology.cosmology.Cosmology): Either a name
			of colossus cosmology, a dict with 'cosmology name': name of
			colossus cosmology, an instance of colussus cosmology, or a
			dict with H0 and Om0 ( other parameters will be set to defaults).
		source_parameters: dictionary with source-specific parameters:
			amp, R_sersic n_sersic, e1, e2, center_x, center_y
			As described by lenstronomy at:
			https://lenstronomy.readthedocs.io/en/latest/lenstronomy.LightModel.Profiles.html#lenstronomy.LightModel.Profiles.sersic.SersicElliptic
	"""

	required_parameters = tuple(
		'amp R_sersic n_sersic e1 e2 center_x center_y'.split())

	def draw_source(self, **kwargs):
		sersic_params ={
			k: v
			for k, v in self.source_parameters.items()
			if k in self.required_parameters}
		return (
			['SERSIC_ELLIPSE'],
			[sersic_params])
