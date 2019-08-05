import os
from io import open  # TODO: REMOVE AFTER TRANSITIONING TO PYTHON 3
import sys
import subprocess
from pkg_resources import parse_version

from ._globals import _NoValue

# Logger for this module only. Use `secml.utils.CLog` elsewhere
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger_handle = logging.StreamHandler(sys.stdout)
_logger_handle.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_logger.addHandler(_logger_handle)

__all__ = ['_NoValue', '__version__', 'global_filterwarnings']

_here = os.path.abspath(os.path.dirname(__file__))


if sys.version_info < (3, 0):
    _logger.warn("DEPRECATION: Python 2.7 is deprecated, please use "
                 "Python >= 3.5. Support for Python 2.7 will be dropped "
                 "in a future release without advanced notice.")


def _read(*path_parts):
    with open(os.path.join(_here, *path_parts), 'r', encoding='ascii') as fp:
        return fp.read().strip()


def _write_rev(v, *path_parts):
    """Write revision id to file."""
    a = open(os.path.join(_here, *path_parts), 'w', encoding='ascii')
    try:
        a.write(v)
    finally:
        a.close()


# Return the git revision as a string
# Thanks to Numpy GitHub: https://github.com/numpy/numpy
def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        # Execute in the current dir
        out = subprocess.Popen(cmd, cwd=_here, env=env,
                               stdout=subprocess.PIPE,
                               stderr=open(os.devnull, 'w')).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
        if len(GIT_REVISION) == 0:
            raise OSError
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


# Check if we want to building a release package
is_release = False
try:
    is_release = bool(os.environ['SECML_ISRELEASE'])
except KeyError:
    pass

# For version string format see: https://packaging.pypa.io/en/latest/version/
try:
    _v_f = _read('VERSION')  # Read main version file
    if not is_release:  # Override for is_release checks
        _v_git = git_version()
        if _v_git == 'Unknown':
            try:  # Try to read rev from file. May not exists
                _v_git = _read('VERSION_REV')
            except:
                pass  # _v_git will stay "Unknown"
        else:
            _write_rev(_v_git, 'VERSION_REV')
        # Append rev number only if available
        _v = _v_f if _v_git == 'Unknown' else _v_f + '+' + _v_git
    else:
        _v = _v_f  # release package
    _v = parse_version(_v)  # Integrity checks
    # Display rev number (if available) for dev releases only
    if _v._version.dev is not None:
        __version__ = str(_v)
    else:  # alpha/beta/rc/final only display public
        __version__ = _v.public
except:
    raise RuntimeError("Unable to find version string.")


# The following are global filters for warnings
def global_filterwarnings():

    import warnings

    # TODO: REMOVE WHEN SCIPY MIN VERSION WILL BE 1.3
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning,
        message="the matrix subclass is not the recommended way to represent "
                "matrices or deal with linear algebra*"
    )

    # Warnings related to data-type size changed. # TODO: fixed in numpy 1.16.1
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="numpy.dtype size changed*"
    )
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="numpy.ufunc size changed*"
    )

    # TODO: REMOVE AFTER SWITCHING TO PYTHON 3
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message="The SafeConfigParser class has been renamed to "
                "ConfigParser in Python 3.2.*"
    )

    # TODO: fixed in scipy 1.3.1
    warnings.filterwarnings(
        "ignore", category=FutureWarning,
        message="future versions will not create a writeable array "
                "from broadcast_array*"
    )
    # TODO: fixed in scipy 1.3.1
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message="Numpy has detected that you (may be)*")  # same as before

    # TODO: check after upgrading to matplotlib 3
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message="Using or importing the ABCs from 'collections' instead of "
                "from 'collections.abc' is deprecated*")


# Call the filterwarnings method to make it active project-wide
global_filterwarnings()
