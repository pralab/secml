from setuptools import setup, find_packages
from pkg_resources import parse_version
import os
import subprocess

here = os.path.abspath(os.path.dirname(__file__))

# Check if we want to building a release package
is_release = False
try:
    is_release = bool(os.environ['SECML_ISRELEASE'])
except KeyError:
    pass


def read(*path_parts):
    with open(os.path.join(here, *path_parts), 'r', encoding='ascii') as fp:
        return fp.read().strip()


def parse_readme(*path_parts):  # For README.md we accept utf-8 chars
    with open(os.path.join(here, *path_parts), 'r', encoding='utf-8') as fp:
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
        res = subprocess.Popen(cmd, cwd=here, env=env,
                               stdout=subprocess.PIPE,
                               stderr=open(os.devnull, 'w')).communicate()[0]
        return res

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
        if not is_release:  # Override for is_release checks
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
        else:
            _v = _v_f  # release package
        _v = parse_version(_v)  # Integrity checks
        # Display rev number (if available) for dev releases only
        # alpha/beta/rc/final only display public
        return str(_v) if _v._version.dev is not None else _v.public
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
    default = open(os.path.join(here, 'requirements.txt'),
                   'r', encoding='ascii').readlines()
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

LONG_DESCRIPTION = parse_readme('README.md')

# List of classifiers: https://pypi.org/pypi?%3Aaction=list_classifiers
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Programming Language :: Python :: Implementation :: PyPy
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Operating System :: Microsoft :: Windows
"""

setup(
    name='secml',
    version=find_version("src", "secml", "VERSION"),
    description='A library for Secure and Explainable Machine Learning',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license='Apache License 2.0',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["Linux", "Mac OS-X", "Unix", "Windows"],
    url='https://secml.gitlab.io',
    download_url="https://pypi.python.org/pypi/secml#files",
    project_urls={
        "Bug Tracker": "https://gitlab.com/secml/secml/-/issues",
        "Source Code": "https://gitlab.com/secml/secml",
    },
    maintainer='Marco Melis',
    maintainer_email='marco.melis@unica.it',
    packages=find_packages('src', exclude=[
        "*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={'': 'src'},
    include_package_data=True,
    python_requires='>=3.5, <3.9',
    install_requires=REQ_PKGS,
    extras_require={
        'pytorch': ["torch>=1.4,!=1.5.*", "torchvision>=0.5,!=0.6.*"],
        'cleverhans': ["tensorflow>=1.14,<2", "cleverhans"],
        'tf-gpu': ["tensorflow-gpu>=1.14,<2"],
        'foolbox': ["foolbox>=3.3.0", "torch>=1.4,!=1.5.*", "torchvision>=0.5,!=0.6.*"],
        'unittests': ['pytest>=5,<5.1',
                      'pytest-cov>=2.9', 'coverage<5',
                      'jupyter', 'nbval', 'requests-mock']
    },
    zip_safe=False
)
