import numpy as np


from manada.Utils.cosmology_utils import get_cosmology

DEFAULT_Z = 2.


class GalaxyCatalog:
    """Base class for real galaxy inputs

    Kwargs:
        - cosmology: anything get_cosmology from utils will accept
    """

    def __init__(self, *, cosmology):
        self.cosmo = get_cosmology(cosmology)

    def __len__(self):
        raise NotImplementedError

    def image_and_metadata(self, catalog_i):
        """Return (image array, metadata record) for one galaxy"""
        raise NotImplementedError

    def iter_lightmodel_kwargs_samples(self,
                                       n_galaxies,
                                       z_new=DEFAULT_Z,
                                       **selection_kwargs):
        """Yield dicts of lenstronomy LightModel kwargs for n_galaxies,
        placed at redshift z_new

        Args:
            n_galaxies (int): Number of galaxies to draw

        Kwargs:
            z_new (float): Redshift to place galaxies at
            Other kwargs will be passed to the sample_indices method.
        """
        for catalog_i in self.sample_indices(n_galaxies):
            yield self.lightmodel_kwargs(catalog_i, z_new=z_new)

    def iter_image_and_metadata(self, message=''):
        """Yield (image, metadata) for all images in catalog"""
        for catalog_i in range(len(self)):
            yield self.image_and_metadata(catalog_i)

    def sample_indices(self, n_galaxies):
        """Return n_galaxies array of ints, catalog indices to sample

        Args:
            n_galaxies (int): Number of indices to return
        """
        return np.random.randint(0, len(self), size=n_galaxies)

    def lightmodel_kwargs(self, catalog_i, z_new=DEFAULT_Z):
        """Create lenstronomy interpolation lightmodel kwargs from
            a catalog image.

        Args:
            catalog_i: Index of image in catalog
            z_new: Redshift to place image at

        Returns:
            dict with kwargs for
            lenstronomy.LightModel.Profiles.interpolation.Interpol
        """
        img, metadata = self.image_and_metadata(catalog_i)
        z, pixel_width = metadata['z'], metadata['pixel_width']

        # Convert image to flux / arcsec^2
        # TODO: do we also need to convert to a magnitude scale?
        img = img / pixel_width**2

        # Pixel length ~ angular diameter distance
        # (colossus uses funny /h units, but for ratios it
        #  fortunately doesn't matter)
        pixel_width *= (self.cosmo.angularDiameterDistance(z_new)
                        / self.cosmo.angularDiameterDistance(z))

        # (flux/arcsec^2) ~ 1/(luminosity distance)^2
        img *= (self.cosmo.luminosityDistance(z)
                / self.cosmo.luminosityDistance(z_new))**2

        # Assuming image is centered, compute center in angular coordinates
        center = np.array(img.shape) * pixel_width / 2

        # Convert to kwargs for lenstronomy
        return dict(
            image=img,
            center_x=center[0],
            center_y=center[1],
            phi_G=0,
            scale=pixel_width)


class DummyCatalog(GalaxyCatalog):
    """
    Dummy catalog consisting of one 4x4 image filled with ones

    Useful for unit testing.
    """

    def __len__(self):
        return 1

    @staticmethod
    def image_and_metadata(catalog_i):
        assert catalog_i == 0, "Dummy catalog has only one image"
        img = np.ones((4, 4), dtype=np.float)
        metadata = dict(
            z=1,
            size=img.shape,
            pixel_width=0.3)
        return img, metadata
