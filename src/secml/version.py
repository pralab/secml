import os
import subprocess
from pkg_resources import parse_version


_VERSION = '0.1-dev0'


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
        cwd = os.path.dirname(__file__)
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=cwd, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', '--short', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


_v_git = git_version()  # Retrieve VCS revision

# Append VCS revision to raw version string
_VERSION = _VERSION if _v_git == 'Unknown' else _VERSION + '+' + _v_git

# Parse final version string for validation
VERSION = parse_version(_VERSION)
