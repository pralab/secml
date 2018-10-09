import os
import io
from pkg_resources import parse_version

from ._globals import _NoValue
from .core.settings import SECML_CONFIG


# From `VERSION` file (see https://packaging.pypa.io/en/latest/version/)
version_p = os.path.join(os.path.dirname(__file__), 'VERSION')
try:
    with io.open(version_p, 'r') as _version_f:
        __version__ = parse_version(_version_f.read().strip())
except:
    raise RuntimeError("Unable to find version string.")
