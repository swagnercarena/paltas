from copy import copy

from .source_base import SourceBase


class SingleSersicSource(SourceBase):
	required_parameters = tuple(
		'amp R_sersic n_sersic e1 e2 center_x center_y'.split())

	def draw_source(self, **kwargs):
		return (['SERSIC_ELLIPSE'],
				[copy(self.source_parameters)])
