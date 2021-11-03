import numpy as np
import unittest
import os
from manada.PointSource.point_source_base import PointSourceBase
from manada.PointSource.single_point_source import SinglePointSource
from manada.Sources.sersic import SingleSersicSource
from manada.Sources.cosmos import COSMOSCatalog
from manada.Sources.cosmos_sersic import COSMOSSersic
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
    
    def test_update_parameters(self):
        # test passing None
        self.c.update_parameters(None)
        self.assertDictEqual(self.c.point_source_parameters, dict())
        # test passing an element that wasn't there yet
        prev = self.c.point_source_parameters
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
            mag_zeropoint=25
        )
        self.c = SinglePointSource(point_source_parameters=
            self.point_source_parameters)

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
        for p in params :
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
        n_pixels = 128
        pixel_width = 0.08
        psf_class = PSF(psf_type='NONE')
        data_class = ImageData(**data_configure_simple(numPix=n_pixels,
            deltaPix=pixel_width))

        # draw image with point source
        complete_image_model = ImageModel(data_class=
            data_class,psf_class=psf_class,lens_model_class=lens_model, 
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
            self.c.point_source_parameters['mag_zeropoint'])
        flux_image = np.sum(im_diff)
        self.assertAlmostEqual(flux_true,flux_image)

        # make sure light is in the center of the image
        # 128 x 128 image
        self.assertTrue( np.sum(im_diff[60:68,60:68]) > 0 )

        # test image positions compared to a sersic w/ high n_sersic
        # define lens, psf, data classes
        lens_kwargs[0]['gamma'] = 2.0
        lens_kwargs[0]['theta_E'] = 0.5
        psf_class = PSF(psf_type='GAUSSIAN', fwhm=0.1 * pixel_width)

        # set up COSMOS
        test_cosmo_folder = (os.path.dirname(
			os.path.abspath(__file__))+'/test_data/cosmos/')
        # set high n sersic, low R sersic
        cosmossersic_params = {
			'smoothing_sigma':0, 'max_z':None, 'minimum_size_in_pixels':None,
			'min_apparent_mag':None,'cosmos_folder':test_cosmo_folder,
			'random_rotation':False, 'min_flux_radius':None,
			'output_ab_zeropoint':25, 'z_source':1.5,
			'mag_sersic':21, 'R_sersic':0.001, 'n_sersic':6.5, 
  			'e1_sersic':0, 'e2_sersic':0, 'center_x_sersic':0, 
			  'center_y_sersic':0}
        cosmos = COSMOSCatalog(cosmology_parameters='planck18',
			source_parameters=cosmossersic_params)
        light_model_list, light_kwargs = cosmos.draw_source(0)

        # PointSource
        self.point_source_parameters['magnitude'] = 18.75
        self.point_source_parameters['x_point_source'] = 0.
        self.point_source_parameters['y_point_source'] = 0.
        self.c.update_parameters(self.point_source_parameters)
        list_ps_model, point_source_kwargs = self.c.draw_point_source()
        point_source_model = PointSource(list_ps_model)
    
        # COSMOS + Point Source
        point_source_model = PointSource(list_ps_model)
        light_model = LightModel(light_model_list)
        image_model = ImageModel(data_class=data_class, psf_class=psf_class,
            lens_model_class=lens_model, source_model_class=light_model,
            point_source_class=point_source_model)
        im_cosmos_ps = image_model.image(kwargs_lens=lens_kwargs, 
            kwargs_source=light_kwargs,kwargs_ps=point_source_kwargs)
        
        # COSMOS + Sersic
        cosmossersic = COSMOSSersic(cosmology_parameters='planck18', 
            source_parameters=cosmossersic_params)
        light_model_list, light_kwargs = cosmos.draw_source(0)
        light_model = LightModel(light_model_list)
        image_model = ImageModel(data_class=data_class, psf_class=psf_class,
            lens_model_class=lens_model,source_model_class=light_model)
        im_cosmossersic = image_model.image(kwargs_lens=lens_kwargs, 
            kwargs_source=light_kwargs)

        # TODO: see if images are in the same location
        # maybe ask to see if this is even a good idea??

        # tear down COSMOS
        os.remove(test_cosmo_folder+'manada_catalog.npy')
        for i in range(10):
            os.remove(test_cosmo_folder+'npy_files/img_%d.npy'%(i))
        os.rmdir(test_cosmo_folder+'npy_files')