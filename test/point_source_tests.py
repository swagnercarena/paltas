import numpy as np
import unittest
from paltas.PointSource.point_source_base import PointSourceBase
from paltas.PointSource.single_point_source import SinglePointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util.simulation_util import data_configure_simple
from lenstronomy.Util.data_util import magnitude2cps
from lenstronomy.Data.psf import PSF


class PointSourceBaseTests(unittest.TestCase):

	def setUp(self):
		self.c = PointSourceBase(point_source_parameters=dict())

	def test_update_parameters(self):
		# test passing None
		self.c.update_parameters(None)
		self.assertDictEqual(self.c.point_source_parameters, dict())
		# test passing an element that wasn't there yet
		self.c.update_parameters({'radius':1.})
		self.assertDictEqual(self.c.point_source_parameters, {'radius':1.})

	def test_draw_point_source(self):
		# Just test that the not implemented error is raised.
		with self.assertRaises(NotImplementedError):
			self.c.draw_point_source()


class SinglePointSourceTests(PointSourceBaseTests):
	def setUp(self):
		self.point_source_parameters=dict(
			x_point_source=0.001,
			y_point_source=0.001,
			magnitude=22,
			output_ab_zeropoint=25,
			compute_time_delays=False
		)
		self.c = SinglePointSource(
			point_source_parameters=self.point_source_parameters)

	def test_check_parameterization(self):
		# test that the base class actually checks for missing parameters
		failed_parameters = dict(x_point_source=0.001,y_point_source=0.001,
			magnitude=22)
		with self.assertRaises(ValueError):
			SinglePointSource(point_source_parameters=failed_parameters)

	def test_update_parameters(self):
		# test a parameter originally set in setUp
		self.point_source_parameters['magnitude'] = 10
		self.c.update_parameters(self.point_source_parameters)
		self.assertEqual(self.c.point_source_parameters['magnitude'], 10)

	def test_draw_point_source(self):
		list_model, list_kwargs = self.c.draw_point_source()

		# test list_model for ['SOURCE_POSITION']
		self.assertTrue('SOURCE_POSITION' in list_model)

		# test that all needed parameters are in list_kwargs
		params = ('ra_source', 'dec_source', 'point_amp')
		for p in params:
			self.assertTrue(p in list_kwargs[0].keys())

		# now, test with no lens mass
		list_ps_model, list_ps_kwargs = self.c.draw_point_source()

		# set up lens, source light, point source models
		lens_model = LensModel(['SPEP'])
		lens_kwargs = [{'theta_E': 0.0, 'e1': 0., 'e2': 0., 'gamma': 0.,
			'center_x': 0, 'center_y': 0}]
		source_light_model = LightModel(['SERSIC_ELLIPSE'])
		source_kwargs = [{'amp':70, 'R_sersic':0.1, 'n_sersic':2.5,
			'e1':0., 'e2':0., 'center_x':0.01, 'center_y':0.01}]

		point_source_model = PointSource(list_ps_model)

		# define PSF class, data class
		n_pixels = 64
		pixel_width = 0.08
		psf_class = PSF(psf_type='NONE')
		data_class = ImageData(**data_configure_simple(numPix=n_pixels,
			deltaPix=pixel_width))

		# draw image with point source
		complete_image_model = ImageModel(data_class=data_class,
			psf_class=psf_class,lens_model_class=lens_model,
			source_model_class=source_light_model,
			point_source_class=point_source_model)
		image_withPS = complete_image_model.image(kwargs_lens=lens_kwargs,
		kwargs_source=source_kwargs, kwargs_ps=list_ps_kwargs)

		# draw image without point source
		complete_image_model = ImageModel(data_class=data_class,
			psf_class=psf_class,lens_model_class=lens_model,
			source_model_class=source_light_model,
			point_source_class=None)
		image_noPS = complete_image_model.image(kwargs_lens=lens_kwargs,
		kwargs_source=source_kwargs, kwargs_ps=None)

		# take difference to isolate point source
		im_diff = image_withPS - image_noPS

		# make sure we get a nonzero image out
		self.assertTrue(np.sum(im_diff) > 0)

		# make sure the flux is what we expect
		flux_true = magnitude2cps(self.c.point_source_parameters['magnitude'],
			self.c.point_source_parameters['output_ab_zeropoint'])
		flux_image = np.sum(im_diff)
		self.assertAlmostEqual(flux_true,flux_image)

		# make sure light is in the center of the image 128 x 128 image
		self.assertTrue(np.sum(im_diff[30:34,30:34]) == flux_image)

		# test draw image with mag_pert
		self.point_source_parameters['mag_pert'] = [1, 1, 1, 1, 1]
		self.c.update_parameters(self.point_source_parameters)
		list_model, list_kwargs = self.c.draw_point_source()
		# make sure mag_pert is passed to lenstronomy
		self.assertTrue('mag_pert' in list_kwargs[0].keys())
