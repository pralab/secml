"""
.. module:: Settings
   :synopsis: System settings for SecML

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import os
import sys
import shutil
from configparser import ConfigParser, NoSectionError, NoOptionError

# Logger for this module only. Use `secml.utils.CLog` elsewhere
import logging
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger_handle = logging.StreamHandler(sys.stdout)
_logger_handle.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
_logger.addHandler(_logger_handle)


__all__ = ['SECML_HOME_DIR', 'SECML_CONFIG',
           'SECML_DS_DIR', 'SECML_MODELS_DIR', 'SECML_EXP_DIR',
           'SECML_STORE_LOGS', 'SECML_LOGS_DIR',
           'SECML_LOGS_FILENAME', 'SECML_LOGS_PATH',
           'SECML_PYTORCH_DIR', 'SECML_PYTORCH_USE_CUDA',
           'parse_config']


def parse_config(conf_files, section, parameter, default=None, dtype=None):
    """Parse input `parameter` under `section` from configuration files.

    Parameters file must have the following structure:

    .. code-block::

       [section1]
       param1=xxx
       param2=xxx

       [section2]
       param1=xxx
       param2=xxx

    Parameters
    ----------
    conf_files : list
        List with the paths of the configuration files to parse.
    section : str
        Section under which look for specified parameter.
    parameter : str
        Name of the parameter. This is not case-sensitive.
    default : any, optional
        Set default value of parameter.
        If None (default), parameter is considered required and
        so must be defined in the input configuration file.
        If not None, the value will be used if configuration file
        does not exists, section is not defined, or the parameter
        is not defined under section.
    dtype : type or None, optional
        Expected dtype of the parameter.
        If None (default), parameter will be parsed as a string.
        Other accepted values are: float, int, bool, str.

    """
    # Parsing parameters
    _config = ConfigParser()

    # Parse configuration files (even if not exists)
    # The first item of the list has LOWER priority (so we reverse)
    _config.read(reversed(conf_files))

    # Try to parse the parameter from section
    try:
        # Call the get function appropriate to specified dtype
        if dtype is None or dtype == str:
            param = _config.get(section, parameter)
        elif dtype == int:
            param = _config.getint(section, parameter)
        elif dtype == float:
            param = _config.getfloat(section, parameter)
        elif dtype == bool:
            param = _config.getboolean(section, parameter)
        else:
            raise TypeError(
                "accepted dtypes are int, float, bool, str (or None)")
    except NoSectionError:
        if default is not None:
            # Use default if config file does not exists
            # or does not have the desired section
            return default
        raise RuntimeError(
            "no section `[{:}]` found in configuration files.".format(section))
    except NoOptionError:
        if default is not None:
            # Use default if desired parameter is not specified under section
            return default
        raise RuntimeError(
            "parameter `{:}` not found under section `[{:}]` in "
            "configuration files.".format(parameter, section))

    return param


def _parse_env(name, default=None, dtype=None):
    """Parse input variable from `os.environ`.

    Parameters
    ----------
    name : str
        Name of the variable to parse from env.
    default : any, optional
        Set default value of variable.
        If None (default), parameter is considered required and
        so must be defined in environment.
        Otherwise, RuntimeError will be raised.
    dtype : type or None, optional
        Expected dtype of the variable.
        If None (default), variable will be parsed as a string.
        Other accepted values are: float, int, bool, str.

    """
    try:
        val = os.environ[name]
    except KeyError:
        if default is not None:
            # Let's use the default value if var not in env
            return default
        raise RuntimeError("variable {:} not specified".format(name))

    # Parse var from env using the specified dtype
    if dtype is None or dtype == str:
        return str(val)
    if dtype == int or dtype == float or dtype == bool:
        return dtype(val)
    else:
        raise TypeError(
            "accepted dtypes are int, float, bool, str (or None)")


def _parse_env_config(name, conf_files, section, parameter,
                      default=None, dtype=None):
    """Parse input variable from `os.environ` or configuration files.

    If input variable `name` is not found in `os.environ`,
    the corresponding parameter is parsed from configuration files.
    If not found, default variable value will be returned.
    If no default value has been defined, RuntimeError will be raised.

    Parameters
    ----------
    name : str
        Name of the variable to parse from env.

    For description of other input parameters see `.parse_config`.

    """
    try:  # Firstly let's try to get variable from environment
        # Don't pass default to _parse_env as we want
        # to catch KeyError and RuntimeError
        return _parse_env(name, default=None, dtype=dtype)
    except (KeyError, RuntimeError):
        # Probably the variable is not in env, try read config
        return parse_config(conf_files, section, parameter, default, dtype)


SECML_HOME_DIR = _parse_env(
    'SECML_HOME_DIR',
    default=os.path.join(os.path.expanduser('~'), 'secml-data'))
"""Main directory for storing datasets, experiments, temporary files.

This is set by default to:
    * Unix -> ``${HOME}/secml-data``
    * Windows -> ``(${HOME}, ${USERPROFILE}, ${HOMEPATH}, ${HOMEDRIVE})/secml-data``

"""
if not os.path.isdir(SECML_HOME_DIR):
    os.makedirs(os.path.abspath(SECML_HOME_DIR))
    _logger.info('New `SECML_HOME_DIR` created: {:}'.format(SECML_HOME_DIR))


SECML_CONFIG_FNAME = 'secml.conf'
"""Name of the configuration file (default `secml.conf`)."""
if not os.path.isfile(os.path.join(SECML_HOME_DIR, SECML_CONFIG_FNAME)):
    def_config = os.path.normpath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), SECML_CONFIG_FNAME))
    home_config = os.path.join(SECML_HOME_DIR, SECML_CONFIG_FNAME)
    # Copy the default config file to SECML_HOME_DIR if not already available
    shutil.copy(def_config, home_config)
    _logger.info(
        'Default configuration file copied to: {:}'.format(home_config))


def _config_fpath():
    """Returns the path of the active configuration file(s).

    The list of active configuration files is sorted from the highest
    to the lowest priority, as follows:
     - `${PWD}/secml.conf`
     - `{SECML_CONFIG}` if it is not a directory
     - `{SECML_CONFIG}/secml.conf`
     - `{SECML_HOME_DIR}/secml.conf`
        - On Unix, `${HOME}/secml-data/secml.conf`
        - On Windows, `(${HOME}, ${USERPROFILE}, ${HOMEPATH}, ${HOMEDRIVE})/secml-data/secml.conf`
     - Lastly, it looks in `INSTALL/secml/secml.conf` for a
       system-defined copy.
       INSTALL should be `/usr/lib/python3.5/site-packages` on Linux,
       and `C:\\Python35\\Lib\\site-packages` on Windows.

    Returns
    -------
    list
        The list of active configuration files is sorted from the highest
        to the lowest priority.

    """
    def gen_candidates():
        yield os.path.join(os.getcwd(), SECML_CONFIG_FNAME)
        try:
            secml_config = os.environ['$SECML_CONFIG']
        except KeyError:
            pass
        else:
            yield secml_config
            yield os.path.join(secml_config, 'SECML_CONFIG_FNAME')
        yield os.path.join(SECML_HOME_DIR, SECML_CONFIG_FNAME)
        yield os.path.normpath(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), SECML_CONFIG_FNAME))

    candidates = []
    for fname in gen_candidates():
        if os.path.isfile(fname):
            candidates.append(fname)

    return candidates


SECML_CONFIG = _config_fpath()
"""Active `secml.conf` configuration files."""


# ------- #
# [SECML] #
# ------- #

SECML_DS_DIR = _parse_env_config(
    'SECML_DS_DIR', SECML_CONFIG, 'secml', 'ds_dir',
    dtype=str, default=os.path.join(SECML_HOME_DIR, 'datasets')
)
"""Main directory for storing datasets.

This is set by default to: ``{SECML_HOME_DIR}/datasets``

"""
if not os.path.isdir(SECML_DS_DIR):
    os.makedirs(os.path.abspath(SECML_DS_DIR))
    _logger.info('New `SECML_DS_DIR` created: {:}'.format(SECML_DS_DIR))

SECML_MODELS_DIR = _parse_env_config(
    'SECML_MODELS_DIR', SECML_CONFIG, 'secml', 'models_dir',
    dtype=str, default=os.path.join(SECML_HOME_DIR, 'models')
)
"""Main directory where pre-trained models are stored.

This is set by default to: ``{SECML_HOME_DIR}/models``

"""
if not os.path.isdir(SECML_MODELS_DIR):
    os.makedirs(os.path.abspath(SECML_MODELS_DIR))
    _logger.info('New `SECML_MODELS_DIR` created: {:}'.format(SECML_MODELS_DIR))

SECML_EXP_DIR = _parse_env_config(
    'SECML_EXP_DIR', SECML_CONFIG, 'secml', 'exp_dir',
    dtype=str, default=os.path.join(SECML_HOME_DIR, 'experiments')
)
"""Main directory of experiments data.

This is set by default to: ``{SECML_HOME_DIR}/experiments``

"""
if not os.path.isdir(SECML_EXP_DIR):
    os.makedirs(os.path.abspath(SECML_EXP_DIR))
    _logger.info('New `SECML_EXP_DIR` created: {:}'.format(SECML_EXP_DIR))

# ------------ #
# [SECML:LOGS] #
# ------------ #

SECML_STORE_LOGS = _parse_env_config(
    'SECML_STORE_LOGS', SECML_CONFIG, 'secml:logs', 'store_logs',
    dtype=bool, default=False
)
"""Whether to store logs to file. Default False."""

SECML_LOGS_DIR = _parse_env_config(
    'SECML_LOGS_DIR', SECML_CONFIG, 'secml:logs', 'logs_dir',
    dtype=str, default=os.path.join(SECML_HOME_DIR, 'logs')
)
"""Directory where logs will be stored.

This is set by default to: ``{SECML_HOME_DIR}/logs``

"""
if not os.path.isdir(SECML_LOGS_DIR):
    os.makedirs(os.path.abspath(SECML_LOGS_DIR))
    _logger.info('New `SECML_LOGS_DIR` created: {:}'.format(SECML_LOGS_DIR))

SECML_LOGS_FILENAME = _parse_env_config(
    'SECML_LOGS_FILENAME', SECML_CONFIG, 'secml:logs', 'logs_filename',
    dtype=str, default='logs.log'
)
"""Name of the logs file on disk. Default: `logs.log`."""

SECML_LOGS_PATH = os.path.join(SECML_LOGS_DIR, SECML_LOGS_FILENAME)
"""Full path to the logs file: ``{SECML_LOGS_DIR}/{SECML_LOGS_FILENAME}``."""


# --------------- #
# [SECML:PYTORCH] #
# --------------- #

SECML_PYTORCH_USE_CUDA = _parse_env_config(
    'SECML_PYTORCH_USE_CUDA', SECML_CONFIG, 'secml:pytorch', 'use_cuda',
    dtype=bool, default=True
)
"""Controls if CUDA should be used by the PyTorch wrapper when available."""

SECML_PYTORCH_DIR =_parse_env(
    'SECML_PYTORCH_DIR',
    default=os.path.join(os.path.expanduser('~'), 'secml-data/pytorch-data'))
if not os.path.isdir(SECML_PYTORCH_DIR):
    os.makedirs(os.path.abspath(SECML_PYTORCH_DIR))
    _logger.info('New `SECML_PYTORCH_DIR` created: {:}'.format(SECML_PYTORCH_DIR))
"""Directory for storing PyTorch data.
 
This is set by default to: `{SECML_HOME_DIR}`/pytorch-data`

"""
