==========================================================================
|logo| paltas
==========================================================================

.. |logo| image:: https://raw.githubusercontent.com/swagnercarena/paltas/main/docs/figures/logo.png
    	:target: https://raw.githubusercontent.com/swagnercarena/paltas/main/docs/figures/logo.png
    	:width: 100
.. |pypi| image:: https://badge.fury.io/py/paltas.svg
    	:target: https://badge.fury.io/py/paltas
.. |ci| image:: https://github.com/swagnercarena/paltas/workflows/CI/badge.svg
    	:target: https://github.com/swagnercarena/paltas/actions
.. |coverage| image:: https://coveralls.io/repos/github/swagnercarena/paltas/badge.svg?branch=main
	:target: https://coveralls.io/github/swagnercarena/paltas?branch=main
.. |docs| image:: https://readthedocs.org/projects/paltas/badge/?version=latest
    	:target: https://paltas.readthedocs.io/en/latest
    	:alt: Documentation Status
.. |arxiv| image:: https://img.shields.io/badge/arXiv-2203.00690%20-yellowgreen.svg
    	:target: https://arxiv.org/abs/2203.00690
.. |license| image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    	:target: https://github.com/swagnercarena/paltas/main/LICENSE

|pypi| |ci| |coverage| |docs| |arxiv| |license|

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

The main functionality of ``paltas`` is to generate realistic datasets of strong gravitational lenses in a way that's modular, scalable, and user-friendly. To make a dataset with ``paltas`` all you need is a configuration file which you can then pass to the generate.py script:

.. code-block:: bash

    $ python generate.py path/to/config/file path/to/output/folder --n 100

Running the line of code above would generate 100 lenses and output them in the specified folder. ``paltas``  comes preloaded with a number of configuration files which are described in ``Configs/Examples/README.rst``. For example, to create a dataset with HST observational effects, subhalos, and line-of-sight halos run:

.. code-block:: bash

    $ python generate.py Configs/Examples/config_all.py example --n 100

We provide a tutorial notebook that describes how to `generate your own config file <https://github.com/swagnercarena/paltas/tree/main/notebooks/Config_Tutorial.ipynb>`_.

Demos
-----

``paltas`` comes with a tutorial notebook for users interested in modifying the simulation classes.

* `Implement your own source, line-of-sight, subhalo, or main deflector model <https://github.com/swagnercarena/paltas/tree/main/notebooks/Understanding_Pipeline.ipynb>`_.
* `Training a neural posterior estimator of simulation parameters <https://github.com/swagnercarena/paltas/tree/main/notebooks/Network_Training.ipynb>`_.
* `Running hierarchical inference on a population of strong lenses <https://github.com/swagnercarena/paltas/tree/main/notebooks/Population_Analysis.ipynb>`_.

Figures
-------

Code for generating the plots included in some of the publications using ``paltas`` can be found under the corresponding arxiv number in the ``notebooks/papers/`` folder.

Contributing
------------

We welcome improvements via GitHub pull request! The workflow for contributing to ``paltas`` is the same as it is for ``lenstronomy``, please see the `lenstronomy CONTRIBUTING guidelines <https://github.com/lenstronomy/lenstronomy/blob/main/CONTRIBUTING.rst>`_. for the rundown.

In brief, to contribute to ``paltas`` you will need to fork the repo on GitHub, clone it, and pip install with editable option, like this:

.. code-block:: bash

    $ git clone git@github.com:<your-account>/paltas.git
    $ cd paltas
    $ pip install -e .

You should then work in a local development branch, while keeping your ``main`` branch up to date with the base repo on GitHub. (That means you'll need to ``git remote add base https://github.com/swagnercarena/paltas.git`` so that you can ``git pull base main`` from your ``main`` branch).

See the `lenstronomy CONTRIBUTING guidelines <https://github.com/lenstronomy/lenstronomy/blob/main/CONTRIBUTING.rst>`_. for help with python style, docstring formats, creating pull requests, and so on. We recommend that you start your pull request (PR) when you _begin_ work on a new feature or bug fix, so that the rest of the ``paltas`` community can see what you are doing and coordinate with you - when your improvement is ready you can request code review from others via the PR interface. Try and restrict PRs to one feature per branch, to help keep them small and manageable. (Note that ``paltas`` uses `Google docstring style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.)

Attribution
-----------
If you use ``paltas`` or its datasets for your own research, please cite the ``paltas`` package (`Wagner-Carena et al. 2022 <https://arxiv.org/abs/2203.00690>`_) as well as the ``lenstronomy`` package (`Birrer & Amara 2018 <https://arxiv.org/abs/1803.09746v1>`_, `Birrer et al. 2021 <https://joss.theoj.org/papers/10.21105/joss.03283>`_).

Zenodo Uploads
--------------
The following is a list of the zenodo uploads associated to papers using paltas. These uploads will include additional chains, test sets, and model weights required to reproduce the paper results.

* `From Images to Dark Matter: End-To-End Inference of Substructure From Hundreds of Strong Gravitational Lenses -- Data <https://zenodo.org/record/6326743#.Yo_4qBPML0o>`_.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6326743.svg
   :target: https://doi.org/10.5281/zenodo.6326743
