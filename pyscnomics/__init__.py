""" Specify callable packages from pyscnomics """

from . import contracts as contracts
from . import dataset as dataset
from . import econ as econ
from . import io as io
from . import optimize as optimize
from . import tools as tools

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyscnomics")
except PackageNotFoundError:
    __version__ = "unknown"
