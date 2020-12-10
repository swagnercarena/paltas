import numpy as np
import unittest

from manada.Sources.galaxy_catalog import DummyCatalog


class GalaxyMakerTests(unittest.TestCase):

    def setUp(self):
        self.c = DummyCatalog(cosmology='planck18')

    def test_get_image(self):
        img, meta = self.c.image_and_metadata(0)
        assert isinstance(img, np.ndarray)
        assert isinstance(meta, (np.void, dict))   # Either type is fine

    def test_lightmodel_prep(self):
        _check_lightmodel_kwargs(self.c.lightmodel_kwargs(0))

    def test_sampling(self):
        kwargs_samples = list(self.c.iter_lightmodel_kwargs_samples(10))
        assert len(kwargs_samples) == 10
        for kwargs in kwargs_samples:
            _check_lightmodel_kwargs(kwargs)


def _check_lightmodel_kwargs(kwargs):
    assert isinstance(kwargs, dict)
    assert 'image' in kwargs
    img = kwargs['image']
    assert isinstance(img, np.ndarray)
    assert 0 < img[0, 0] < np.inf
