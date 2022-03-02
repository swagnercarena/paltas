import numpy as np
import unittest
from paltas.MainDeflector.main_deflector_base import MainDeflectorBase
from paltas.MainDeflector.simple_deflectors import PEMD, PEMDShear
from paltas.Utils.cosmology_utils import get_cosmology
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util.simulation_util import data_configure_simple
from lenstronomy.Data.psf import PSF


class SourceBaseTests(unittest.TestCase):

	def setUp(self):
		self.c = MainDeflectorBase(
			cosmology_parameters='planck18',
			main_deflector_parameters=dict())
		self.cosmo = get_cosmology('planck18')

	def test_update_parameters(self):
		# Check that the update parameter call updates the cosmology
		h = self.c.cosmo.h
		self.c.update_parameters(cosmology_parameters='WMAP9')
		self.assertNotEqual(h,self.c.cosmo.h)

	def test_draw_source(self):
		# Just test that the not implemented error is raised.
		with self.assertRaises(NotImplementedError):
			self.c.draw_main_deflector()


class PEMDTests(unittest.TestCase):

	def setUp(self):
		self.main_deflector_parameters = {'z_lens': 0.5,'gamma': 2.0,
			'theta_E': 1.0,'e1': 0.2,'e2': 0.1,'center_x': 0.2,'center_y': 0.1}
		self.c = PEMD(cosmology_parameters='planck18',
			main_deflector_parameters=self.main_deflector_parameters)
		self.cosmo = get_cosmology('planck18')

	def test_check_parameterization(self):
		# Check that trying to initialize a class without the correct
		# parameters raises a value error.
		with self.assertRaises(ValueError):
			PEMD(cosmology_parameters='planck18',
				main_deflector_parameters={})

	def test_update_parameters(self):
		# Check that the update parameter call works
		self.main_deflector_parameters['center_x'] = 0.1
		self.c.update_parameters(
			main_deflector_parameters=self.main_deflector_parameters)
		self.assertEqual(self.c.main_deflector_parameters['center_x'],
			0.1)

	def test_draw_main_deflector(self):
		# Check that the model list and kwargs we get from the draw function
		# are what we expect and work with lenstronomy.
		md_model_list, md_kwargs_list, md_z_list = self.c.draw_main_deflector()

		self.assertListEqual(md_model_list,['EPL_NUMBA'])
		self.assertListEqual(md_z_list,[0.5])

		# Check that the kwargs are what lenstronomy needs
		lens_model = LensModel(md_model_list)
		light_model = LightModel(['SERSIC_ELLIPSE'])
		light_kwargs = [{'amp':1.,'R_sersic':1.,'n_sersic':2.,'e1':0.,'e2':0.,
			'center_x':0.,'center_y':0.}]
		image_model = ImageModel(
			data_class=ImageData(**data_configure_simple(numPix=64,
				deltaPix=0.08)),
			psf_class=PSF(psf_type='NONE'),
			lens_model_class=lens_model,source_model_class=light_model)
		image = image_model.image(kwargs_lens=md_kwargs_list,
			kwargs_source=light_kwargs)
		self.assertTrue(isinstance(image, np.ndarray))
		self.assertTrue(image.sum() > 0)


class PEMDTShearests(unittest.TestCase):

	def setUp(self):
		self.main_deflector_parameters = {'z_lens': 0.5,'gamma': 2.0,
			'theta_E': 1.0,'e1': 0.2,'e2': 0.1,'center_x': 0.2,'center_y': 0.1,
			'gamma1':0.0,'gamma2':0.0,'ra_0':0.0,'dec_0':0.0}
		self.c = PEMDShear(cosmology_parameters='planck18',
			main_deflector_parameters=self.main_deflector_parameters)
		self.cosmo = get_cosmology('planck18')

	def test_check_parameterization(self):
		# Check that trying to initialize a class without the correct
		# parameters raises a value error.
		with self.assertRaises(ValueError):
			PEMD(cosmology_parameters='planck18',
				main_deflector_parameters={})

	def test_update_parameters(self):
		# Check that the update parameter call works
		self.main_deflector_parameters['center_x'] = 0.1
		self.c.update_parameters(
			main_deflector_parameters=self.main_deflector_parameters)
		self.assertEqual(self.c.main_deflector_parameters['center_x'],
			0.1)

	def test_draw_main_deflector(self):
		# Check that the model list and kwargs we get from the draw function
		# are what we expect and work with lenstronomy.
		md_model_list, md_kwargs_list, md_z_list = self.c.draw_main_deflector()

		self.assertListEqual(md_model_list,['EPL_NUMBA','SHEAR'])
		self.assertListEqual(md_z_list,[0.5,0.5])

		# Check that the kwargs are what lenstronomy needs
		lens_model = LensModel(md_model_list)
		light_model = LightModel(['SERSIC_ELLIPSE'])
		light_kwargs = [{'amp':1.,'R_sersic':1.,'n_sersic':2.,'e1':0.,'e2':0.,
			'center_x':0.,'center_y':0.}]
		image_model = ImageModel(
			data_class=ImageData(**data_configure_simple(numPix=64,
				deltaPix=0.08)),
			psf_class=PSF(psf_type='NONE'),
			lens_model_class=lens_model,source_model_class=light_model)
		image = image_model.image(kwargs_lens=md_kwargs_list,
			kwargs_source=light_kwargs)
		self.assertTrue(isinstance(image, np.ndarray))
		self.assertTrue(image.sum() > 0)
