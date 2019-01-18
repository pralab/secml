import os

from secml.settings import SECML_CONFIG, SECML_HOME_DIR
from secml.settings import _parse_env_config


__all__ = ['SECML_PYTORCH_USE_CUDA',
           'SECML_PYTORCH_DATA_DIR', 'SECML_PYTORCH_MODELS_DIR']


"""Main directory for storing PyTorch data, subdirectory of SECML_HOME_DIR.

This is set by default to: 'SECML_HOME_DIR/pytorch-data'

"""
SECML_PYTORCH_DATA_DIR = _parse_env_config(
    'SECML_PYTORCH_DATA_DIR', SECML_CONFIG, 'pytorch', 'data_dir',
    dtype=str, default=os.path.join(SECML_HOME_DIR, 'pytorch-data')
)

"""Main directory for storing PyTorch models, subdirectory of SECML_PYTORCH_DATA_DIR.

This is set by default to: 'SECML_HOME_DIR/SECML_PYTORCH_DATA_DIR/models'

"""
SECML_PYTORCH_MODELS_DIR = os.path.join(SECML_PYTORCH_DATA_DIR, 'models')

"""True if CUDA should be used in PyTorch wrappers.

PyTorch may use CUDA too speed up computations when a 
compatible device is found. 
Set this to False will force PyTorch to use CPU anyway.

This can be set globally or per-script.
To be effective, use in the head of your script. Example:
>>> from secml.core import settings
>>> settings.SECML_PYTORCH_USE_CUDA = False
>>>
>>> **OTHER IMPORTS**
>>> **REST OF CODE**

"""
SECML_PYTORCH_USE_CUDA = _parse_env_config(
    'SECML_PYTORCH_USE_CUDA', SECML_CONFIG, 'pytorch', 'use_cuda',
    dtype=bool, default=True)
