from distutils.core import setup
from setuptools import find_packages
import os

setup(
    name='SecML-Lib',
    version='0.1',
    packages=find_packages('src', exclude=["*.tests", "*.tests.*",
                                           "tests.*", "tests"]),
    package_dir={'': 'src'},
    url='http://pralab.diee.unica.it',
    license='GNU GPLv3',
    author='PRALab',
    author_email='pralab@diee.unica.it',
    description='A library for Secure Machine Learning',
    data_files=[
        (os.path.join(os.path.expanduser('~'), 'pralib'), ['settings.txt'])
    ]
)
