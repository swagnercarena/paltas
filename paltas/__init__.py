__author__ = 'Sebastian Wagner-Carena'
__email__ = 'sebaswagner@outlook.com'
__version__ = '0.2.0'

# Analysis is not imported by default because it required tensorflow.
try:
    import tensorflow as tf
    del tf
except ImportError:
    print("paltas.Analysis disabled since tensorflow is missing")
else:
    from . import Analysis
from . import Configs
from . import Sampling
from . import Sources
from . import Substructure
from . import Utils
from . import MainDeflector
from . import generate
