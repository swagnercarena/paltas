# -*- coding: utf-8 -*-
"""
Add to the functionality of colossus

Useful functions for extra cosmology calculations.
"""
import typing

from colossus.cosmology import cosmology
import numpy as np


def get_cosmology(x: typing.Union[str, dict, cosmology.Cosmology]
                  ) -> cosmology.Cosmology:
    """Return colossus cosmology

    Argument must be either:
        - str: name of colossus cosmology
        - dict with 'cosmology name': name of colossus cosmology
        - colussus cosmology: returned unchanged
        - dict with H0 and Om0; other parameters set to defaults.

    Returns: colossus cosmology
    """
    if isinstance(x, cosmology.Cosmology):
        return x
    if isinstance(x, str):
        return cosmology.setCosmology(x)
    if isinstance(x, dict):
        if 'cosmology_name' in x:
            return get_cosmology(x['cosmology_name'])
        else:
            # Leave some parameters to their default values so the user only has
            # to specify H0 and Om0.
            col_params = dict(
                flat=True,
                H0=x['H0'],
                Om0=x['Om0'],
                Ob0=0.049,
                sigma8=0.81,
                ns=0.95)
            return cosmology.setCosmology('temp_cosmo', col_params)


def kpc_per_arcsecond(z,cosmo):
    """
    Calculate the physical kpc per arcsecond at a given redshift and cosmology

    Parameters:
        z (float): The redshift to calculate the distance at
        cosmo (colossus.cosmology.cosmology.Cosmology): An instance of the
            colossus cosmology object.
    Returns:
        (float): The kpc per arcsecond
    """
    h = cosmo.h
    kpc_per_arcsecond = (cosmo.angularDiameterDistance(z) *np.pi/180/3600 /
        h * 1e3)
    return kpc_per_arcsecond
