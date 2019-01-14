from setuptools import setup, find_packages
from pkg_resources import parse_version
import os
import io
import subprocess

here = os.path.abspath(os.path.dirname(__file__))


def read(*path_parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with io.open(os.path.join(here, *path_parts), 'r') as fp:
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
        out = subprocess.Popen(cmd, cwd=here, env=env,
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


def find_version(*path_parts):
    """This function read version number and revision number if available."""
    try:
        _v_f = read(*path_parts)  # Read main version file
        _v_git = git_version()
        if _v_git == 'Unknown':
            try:  # Try to read rev from file. May not exists
                _v_git = read('src', 'secml', 'VERSION_REV')
            except:
                pass  # _v_git will stay "Unknown"
        else:
            write_rev(_v_git, 'src', 'secml', 'VERSION_REV')
        # Append rev number only if available
        _v = _v_f if _v_git == 'Unknown' else _v_f + '+' + _v_git
        _v = parse_version(_v)  # Integrity checks
        # Display rev number (if available) for prereleases only
        return str(_v) if _v.is_prerelease else _v.public
    except:
        raise RuntimeError("Unable to find version string.")


def write_rev(v, *path_parts):
    """Write revision id to file."""
    a = open(os.path.join(here, *path_parts), 'w')
    try:
        a.write(v)
    finally:
        a.close()


def install_deps():
    """Reads requirements.txt and preprocess it
    to be feed into setuptools.

    This is the only possible way (we found)
    how requirements.txt can be reused in setup.py
    using dependencies from private github repositories.

    Links must be appended by `-{StringWithAtLeastOneNumber}`
    or something like that, so e.g. `-9231` works as well as
    `1.1.0`. This is ignored by the setuptools, but has to be there.

    Warnings
    --------
    To make pip respect the links, you have to use
    `--process-dependency-links` switch. So e.g.:
    `pip install --process-dependency-links {git-url}`

    Returns
    -------
    List of packages and dependency links.

    Notes
    -----
    Thanks to knykda for this implementation:
        https://github.com/pypa/pip/issues/3610#issuecomment-356687173

    """
    default = io.open(os.path.join(here, 'requirements.txt'), 'r').readlines()
    new_pkgs = []
    links = []
    for resource in default:
        if 'git+ssh' in resource:
            pkg = resource.split('#')[-1]
            links.append(resource.strip() + '-9876543210')
            new_pkgs.append(pkg.replace('egg=', '').rstrip())
        else:
            new_pkgs.append(resource.strip())
    return new_pkgs, links


REQ_PKGS, DEP_LINKS = install_deps()

LONG_DESCRIPTION = read('README.md')

# List of classifiers: https://pypi.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 2
Programming Language :: Python :: 2.7
Programming Language :: Python :: Implementation :: PyPy
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""

setup(
    name='SecML-Lib',
    version=find_version("src", "secml", "VERSION"),
    description='A library for Secure Machine Learning',
    long_description=LONG_DESCRIPTION,
    license='GNU GPLv3',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["Linux", "Mac OS-X", "Unix"],
    url='https://sec-ml.pluribus-one.it/lib/',
    download_url=
        'git+ssh://git@pragit.diee.unica.it/secml/secml-lib.git#egg=secml-lib',
    maintainer='Marco Melis',
    maintainer_email='marco.melis@diee.unica.it',
    packages=find_packages('src', exclude=["*.tests", "*.tests.*",
                                           "tests.*", "tests"]),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires='==2.7.*',
    install_requires=REQ_PKGS,
    extras_require={
        'pytorch': ["torch>=0.4.*", "torchvision>=0.1.8"],
        'cleverhans': ["tensorflow>=1.5.*,<2", "cleverhans"]
    },
    zip_safe=False
)
