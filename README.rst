==========================================================================
|logo| paltas
==========================================================================

.. |logo| image:: https://raw.githubusercontent.com/swagnercarena/paltas/main/docs/figures/logo.png
    :target: https://raw.githubusercontent.com/swagnercarena/paltas/main/docs/figures/logo.png
    :width: 100

.. image:: https://github.com/swagnercarena/paltas/workflows/CI/badge.svg
    :target: https://github.com/swagnercarena/paltas/actions

.. image:: https://coveralls.io/repos/github/swagnercarena/paltas/badge.svg?branch=main
	:target: https://coveralls.io/github/swagnercarena/paltas?branch=main

.. image:: https://readthedocs.org/projects/paltas/badge/?version=latest
    :target: https://paltas.readthedocs.io/en/latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/swagnercarena/paltas/main/LICENSE

``paltas`` is a package for conducting simulation-based inference on strong gravitational lensing images. The package builds on ``lenstronomy`` to create large datasets of strong lensing images with realistic low-mass halos, Hubble Space Telescope (HST) observational effects, and galaxy light from HST's COSMOS field. ``paltas`` also includes the capability to easily train neural posterior estimators of the parameters of the lensing system and to run hierarchical inference on test populations.

Installation
------------

``paltas`` is installable via pip:

.. code-block:: bash

    $ pip install paltas

The default ``paltas`` requirements do not include ``tensorflow``, but if you are interested in using the modules contained in the analysis folder, you will have to install ``tensorflow``:

.. code-block:: bash

    $ pip install tensorflow

Attribution
-----------
If you use ovejero or its datasets for your own research, please cite the ``paltas`` package (`Wagner-Carena et al. 2022 <https://arxiv.org/abs/xxxx.yyyy>`_) as well as the``lenstronomy`` package (`Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_,`Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_).
