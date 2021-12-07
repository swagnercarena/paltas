__author__ = 'Sebastian Wagner-Carena'
__email__ = 'sebaswagner@outlook.com'
__version__ = '0.0.1'

try:
    import tensorflow as tf
    del tf
except ImportError:
    print("manada.Analysis disabled since tensorflow is missing")
else:
    from . import Analysis

from . import Configs
from . import Sampling
from . import Sources
from . import Substructure
from . import Utils
from . import generate
