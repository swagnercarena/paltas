import numpy as np
import unittest
import os
from manada.PointSource.point_source_base import PointSourceBase
from manada.PointSource.single_point_source import SinglePointSource
from manada.Sources.sersic import SingleSersicSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util.simulation_util import data_configure_simple
from lenstronomy.Util.data_util import magnitude2cps
from lenstronomy.Data.psf import PSF
import scipy
import copy

class PointSourceBaseTests(unittest.TestCase):
    
    def setUp(self):
        self.c = PointSourceBase(point_source_parameters=dict())

class SinglePointSourceTests(PointSourceBaseTests):
    def setUp(self):
        self.c = SinglePointSource(point_source_parameters=dict(
            x_point_source=0.001,
            y_point_source=0.001,
            magnitude=22,
            mag_zeropoint=25
        ))

    def test_draw_point_source(self):
        list_model, list_kwargs = self.c.draw_point_source()

        # test list_model for ['SOURCE_POSITION']
        self.assertTrue('SOURCE_POSITION' in list_model)

        # test that all needed parameters are in list_kwargs
        params = ('ra_source', 'dec_source', 'point_amp')
        for p in params :
            self.assertTrue(p in list_kwargs[0].keys())

    def test_no_lens_mass(self):

        list_ps_model, list_ps_kwargs = self.c.draw_point_source()
        
        # set up lens, source light, point source models
        lens_model = LensModel(['SPEP'])
        lens_kwargs = [{'theta_E': 0.0, 'e1': 0., 'e2': 0., 'gamma': 0.,
			'center_x': 0, 'center_y': 0}]
        source_light_model = LightModel(['SERSIC_ELLIPSE'])
        source_kwargs = [{'amp':70, 'R_sersic':0.1, 'n_sersic':2.5,
            'e1':0., 'e2':0., 'center_x':0.01, 'center_y':0.01}]

        point_source_model = PointSource(list_ps_model)
        
        # define PSF class
        #psf_class = PSF(psf_type='GAUSSIAN', fwhm=0.1 * pixel_width)
        psf_class = PSF(psf_type='NONE')

        # draw image with point source
        n_pixels = 128
        pixel_width = 0.08
        complete_image_model = ImageModel(data_class=
            ImageData(**data_configure_simple(numPix=n_pixels,deltaPix=pixel_width)),
			psf_class=psf_class,
            lens_model_class=lens_model, source_model_class=source_light_model,
            point_source_class=point_source_model)
        image_withPS = complete_image_model.image(kwargs_lens=lens_kwargs,
		kwargs_source=source_kwargs, kwargs_ps=list_ps_kwargs)

        # draw image without point source
        complete_image_model = ImageModel(data_class=
            ImageData(**data_configure_simple(numPix=n_pixels,deltaPix=pixel_width)),
			psf_class=psf_class,
            lens_model_class=lens_model, source_model_class=source_light_model,
            point_source_class=None)
        image_noPS = complete_image_model.image(kwargs_lens=lens_kwargs,
		kwargs_source=source_kwargs, kwargs_ps=None)

        # take difference to isolate point source
        im_diff = image_withPS - image_noPS

        # make sure we get a nonzero image out
        self.assertTrue(np.sum(im_diff) > 0)

        # make sure the flux is what we expect
        flux_true = magnitude2cps(self.c.point_source_parameters['magnitude'], 
            self.c.point_source_parameters['mag_zeropoint'])
        flux_image = np.sum(im_diff)
        self.assertAlmostEqual(flux_true,flux_image)

        # with no PSF, make sure only 4 nonzero pixels
        self.assertTrue(np.count_nonzero(im_diff) < 5)
        