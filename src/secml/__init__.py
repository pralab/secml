import os
import io
import subprocess
from pkg_resources import parse_version

from ._globals import _NoValue
from .core.settings import SECML_CONFIG


__all__ = ['_NoValue', 'SECML_CONFIG', '__version__']


_here = os.path.abspath(os.path.dirname(__file__))


def _read(*path_parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with io.open(os.path.join(_here, *path_parts), 'r') as fp:
        return fp.read().strip()


def _write_rev(v, *path_parts):
    """Write revision id to file."""
    a = open(os.path.join(_here, *path_parts), 'w')
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


# For version string format see: https://packaging.pypa.io/en/latest/version/
try:
    _v_f = _read('VERSION')  # Read main version file
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
    _v = parse_version(_v)  # Integrity checks
    # Display rev number (if available) for prereleases only
    __version__ = str(_v) if _v.is_prerelease else _v.public
except:
    raise RuntimeError("Unable to find version string.")
