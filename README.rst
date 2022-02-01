==========================================================================
|logo| paltas
==========================================================================

.. |logo| image:: https://raw.githubusercontent.com/swagnercarena/paltas/main/docs/figures/logo.png
    :target: https://raw.githubusercontent.com/swagnercarena/paltas/main/docs/figures/logo.png
    :width: 100

.. image:: https://github.com/swagnercarena/paltas/workflows/CI/badge.svg
    :target: https://github.com/swagnercarena/paltas/actions

.. image:: https://coveralls.io/repos/github/swagnercarena/paltas/badge.svg?branch=main&amp
	:target: https://coveralls.io/github/swagnercarena/paltas?branch=main

.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/swagnercarena/paltas/main/LICENSE

``paltas`` is a package for conducting simulation-based inference on strong gravitational lensing images. The package builds on ``lenstronomy`` to create large datasets of strong lensing images with realistic low-mass halos, Hubble Space Telescope (HST) observational effects, and galaxy light from HST's COSMOS field. ``paltas`` also includes the capability to easily train neural posterior estimators of the parameters of the lensing system and to run hierarchical inference on test populations.

Installation
------------

Lenstronomy requires an additional fortran package (fastell) to run lens models with elliptical mass distributions. Thankfully, installing the package is fairly easy (although a fortran compiler is required).

.. code-block:: bash

    $ git clone https://github.com/sibirrer/fastell4py.git <desired location>
    $ cd <desired location>
    $ python setup.py install --user


In the future, paltas will be a pypi package. For now, it can be installed by cloning the git repo.

.. code-block:: bash

	$ git clone https://github.com/swagnercarena/paltas.git
	$ cd paltas/
	$ pip install -e . -r requirements.txt

The addition of the -e option will allow you to pull paltas updates and have them work automatically.