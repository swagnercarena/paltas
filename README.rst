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

The default ``paltas`` requirements do not include ``tensorflow``, but if you are interested in using the modules contained in the Analysis folder, you will have to install ``tensorflow``:

.. code-block:: bash

    $ pip install tensorflow

Usage
-----

The main functionality of ``paltas`` is to generate realistic datasets of strong gravitational lenses in a way that's modular, scalable, and user-friendly. To make a dataset with platas all you need is a configuration file which you can then pass to the generate.py script:

.. code-block:: bash

    $ python generate.py path/to/config/file path/to/output/folder --n 100

Running the line of code above would generate 100 lenses and output them in the specified folder. ``paltas``  comes preloaded with a number of configuration files which are described in Configs/README.rst. To use one just run:

.. code-block:: bash

    $ python generate.py Configs/config_all.py example --n 100

We provide a tutorial notebook that describes how to 1generate your own config file <https://https://github.com/swagnercarena/paltas/tree/main/notebooks/ConfigTutorial.ipynb>`_.

Attribution
-----------
If you use ``paltas`` or its datasets for your own research, please cite the ``paltas`` package (`Wagner-Carena et al. 2022 <https://arxiv.org/abs/xxxx.yyyy>`_) as well as the``lenstronomy`` package (`Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_,`Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_).
