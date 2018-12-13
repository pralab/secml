from setuptools import setup, find_packages
from pkg_resources import parse_version
import os
import io
import re
import subprocess

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with io.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read().strip()


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


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^_VERSION = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        _v_raw = version_match.group(1)
        _v_git = git_version()
        _v_raw = _v_raw if _v_git == 'Unknown' else _v_raw + '+' + _v_git
        return str(parse_version(_v_raw))

    raise RuntimeError("Unable to find version string.")


LONG_DESCRIPTION = read('README.md')

setup(
    name='SecML-Lib',
    version=find_version("src", "secml", "version.py"),
    description='A library for Secure Machine Learning',
    long_description=LONG_DESCRIPTION,
    license='GNU GPLv3',
    url='https://sec-ml.pluribus-one.it/lib/',
    download_url='https://pypi.org/project/secml-lib/#files',
    maintainer='Marco Melis',
    maintainer_email='marco.melis@diee.unica.it',
    packages=find_packages('src', exclude=["*.tests", "*.tests.*",
                                           "tests.*", "tests"]),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires='==2.7.*'
)
