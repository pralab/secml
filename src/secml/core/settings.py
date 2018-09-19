"""
.. module:: Settings
   :synopsis: System settings for PraLib

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import os
from ConfigParser import SafeConfigParser, NoSectionError, NoOptionError, Error


__all__ = ['HOME_DIR', 'EXP_DIR', 'DATA_DIR',
           'USE_NUMBA',
           'USE_CUDA']


def parse_setting(params_file, section, parameter, default=None, dtype=None):
    """Parse input `parameter` from parameters file under `section`.

    Parameters file must have the following structure:

        [section1]
        param1=xxx
        param2=xxx

        [section2]
        param1=xxx
        param2=xxx

    Parameters
    ----------
    params_file : str
        Path to the parameters file to parse.
    section : str
        Section under which look for specified parameter.
    parameter : str
        Name of the parameter. This is not case-sensitive.
    default : any
        Set default value of parameter.
        If None (default), parameter is considered required and
        so must be defined in the input configuration file.
        If not None, the value will be used if configuration file
        does not exists, section is not defined, or the parameter
        is not defined under section.
    dtype : type
        Expected dtype of the parameter.
        If None (default), parameter will be parse as a string.
        Other accepted values are: float, int, bool.

    """
    # Parsing parameters
    _config = SafeConfigParser()

    # Parse configuration file (even if not exists)
    _config.read(params_file)

    # Try to parse the parameter from section
    try:
        # Call the get function appropriate to specified dtype
        if dtype is None:
            param = _config.get(section, parameter)
        elif dtype == int:
            param = _config.getint(section, parameter)
        elif dtype == float:
            param = _config.getfloat(section, parameter)
        elif dtype == bool:
            param = _config.getboolean(section, parameter)
        else:
            raise ValueError("accepted dtypes are int, float, bool (or None)")
    except NoSectionError:
        if default is not None:
            # Use default if config file does not exists
            # or does not have the desired section
            return default
        raise Error("check that section `[{:}]` exists in {:}"
                    "".format(section, params_file))
    except NoOptionError:
        if default is not None:
            # Use default if desired parameter is not specified under section
            return default
        raise Error("parameter `{:}` not found under section `[{:}]` of {:}"
                    "".format(parameter, section, params_file))
    except Error as e:  # Any other parser error (don't return default here)
        raise Error(e)

    return param


"""Main directory for storing datasets, experiments, temporary files.

This is set by default to:
    * Unix -> '$HOME/pralib'
    * Windows -> '$HOME/$USERPROFILE/pralib'

"""
HOME_DIR = os.path.join(os.path.expanduser('~'), 'pralib')


"""Main directory of experiments data, subdirectory of HOME_DIR.

This is set by default to: 'HOME_DIR/experiments'

"""
EXP_DIR = os.path.join(HOME_DIR, 'experiments')


"""Main directory for storing datasets, subdirectory of HOME_DIR.

This is set by default to: 'HOME_DIR/datasets'

"""
DATA_DIR = os.path.join(HOME_DIR, 'datasets')


"""Main directory of parameters `settings.txt` file."""
PRLIB_SETTINGS = os.path.join(HOME_DIR, 'settings.txt')

# [PRLIB]

"""True if functions optimized with Numba library should be used.

This can be globally set inside params file or per-script.
To be effective, use in the head of your script. Example:
>>> from secml.core import settings
>>> settings.USE_NUMBA = True
>>>
>>> **OTHER IMPORTS**
>>> **REST OF CODE**

"""
USE_NUMBA = parse_setting(
    PRLIB_SETTINGS, 'prlib', 'use_numba', dtype=bool, default=True)

# [PYTORCH]

"""True if CUDA should be used in PyTorch wrappers.

PyTorch may use CUDA too speed up computations when a 
compatible device is found. 
Set this to False will force PyTorch to use CPU anyway.

This can be set globally or per-script.
To be effective, use in the head of your script. Example:
>>> from secml.core import settings
>>> settings.USE_CUDA = False
>>>
>>> **OTHER IMPORTS**
>>> **REST OF CODE**

"""
USE_CUDA = parse_setting(
    PRLIB_SETTINGS, 'pytorch', 'use_cuda', dtype=bool, default=True)
