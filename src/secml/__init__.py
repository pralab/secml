import os
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

    # Warnings related to data-type size changed. # TODO: fixed in numpy ??
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="numpy.dtype size changed*"
    )
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="numpy.ufunc size changed*"
    )

    # TODO: check after upgrading to tensorflow 2
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message="Using or importing the ABCs from 'collections' instead of "
                "from 'collections.abc' is deprecated*")

    # TODO: check after upgrading to tensorflow 2
    warnings.filterwarnings(
        "ignore", category=PendingDeprecationWarning,
        message="the imp module is deprecated in favour of importlib*")
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message="the imp module is deprecated in favour of importlib*")

    # TODO: check after upgrading to tensorflow 2
    warnings.filterwarnings(
        "ignore", category=FutureWarning, message="Passing (type, 1)*")

    # TODO: check after cleverhans fix this (post 3.0.1)
    try:  # For some reason we are not able to filter tf warnings
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except ImportError:
        pass

    # TODO: check after upgrading to tensorflow 2 (related to numpy v0.19)
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message=r"tostring\(\) is deprecated\. Use tobytes\(\) instead\.")

    # TODO: warning raised by torchvision mnist loader first time you download
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torchvision.datasets.mnist",
        message=r"The given NumPy array is not writeable")

    # TODO: cures https://github.com/pytorch/pytorch/issues/47038
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=r"CUDA initialization")


# Call the filterwarnings method to make it active project-wide
global_filterwarnings()
