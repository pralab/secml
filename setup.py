from setuptools import setup, find_packages
from pkg_resources import parse_version
import io
import os

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with io.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read().strip()


def find_version(*file_paths):
    try:
        return parse_version(read(*file_paths)).public
    except:
        raise RuntimeError("Unable to find version string.")


long_description = read('README.md')

setup(
    name='SecML-Lib',
    version=find_version("src", "secml", "VERSION"),
    description='A library for Secure Machine Learning',
    long_description=long_description,
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
