==========================================================================
manada - Substructure Inference of Strong Gravitational Lenses
==========================================================================

''manada'' is not yet sure what it does

Installation
------------

Lenstronomy requires an additional fortran package (fastell) to run lens models with elliptical mass distributions. Thankfully, installing the package is fairly easy (although a fortran compiler is required).

.. code-block:: bash

    $ git clone https://github.com/sibirrer/fastell4py.git <desired location>
    $ cd <desired location>
    $ python setup.py install --user


In the future, manada will be a pypi package. For now, it can be installed by cloning the git repo.

.. code-block:: bash

	$ git clone https://github.com/swagnercarena/manada.git
	$ cd manada/
	$ pip install -e . -r requirements.txt

The addition of the -e option will allow you to pull manada updates and have them work automatically.

TODO: Label the units for all functions