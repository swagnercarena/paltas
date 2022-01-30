==========================================================================
paltas
==========================================================================

.. image:: https://github.com/swagnercarena/paltas/workflows/CI/badge.svg
    :target: https://github.com/swagnercarena/paltas/actions

.. image:: https://coveralls.io/repos/github/swagnercarena/paltas/badge.svg?branch=main
	:target: https://coveralls.io/github/swagnercarena/paltas?branch=main

.. image:: docs/logo.png
  :width: 200

''paltas'' is not yet sure what it does

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