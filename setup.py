from distutils.core import setup
import os

setup(
    name='SecML-Lib',
    version='0.1',
    packages=['', 'secml', 'secml.core', 'secml.data', 'secml.data.tests',
              'secml.data.loader', 'secml.data.loader.test',
              'secml.data.selection', 'secml.data.selection.tests',
              'secml.data.splitter', 'secml.data.splitter.tests',
              'secml.array', 'secml.array.tests', 'secml.peval',
              'secml.peval.tests', 'secml.peval.metrics',
              'secml.peval.metrics.tests', 'secml.stats', 'secml.utils',
              'secml.figure', 'secml.figure.plots', 'secml.figure.tests',
              'secml.kernel', 'secml.kernel.tests',
              'secml.kernel.numba_kernel', 'secml.kernel.numba_kernel.tests',
              'secml.features', 'secml.features.reduction',
              'secml.features.reduction.tests', 'secml.features.normalization',
              'secml.features.normalization.tests', 'secml.parallel',
              'secml.similarity', 'secml.classifiers',
              'secml.classifiers.loss', 'secml.classifiers.loss.numba',
              'secml.classifiers.tests', 'secml.classifiers.multiclass',
              'secml.classifiers.multiclass.tests',
              'secml.classifiers.regularizer', 'secml.optimization',
              'secml.optimization.tests', 'secml.adv', 'secml.adv.attacks',
              'secml.adv.attacks.evasion'],
    package_dir={'': 'src'},
    url='http://pralab.diee.unica.it',
    license='GNU GPLv3',
    author='PRALab',
    author_email='pralab@diee.unica.it',
    description='A library for Secure Machine Learning',
    data_files=[
        (os.path.join(os.path.expanduser('~'), 'secml'), ['settings.txt'])
    ]
)
