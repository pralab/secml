"""
.. module:: LoadModel
   :synopsis: Functions to load pre-trained SecML models

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import json
import re
from datetime import datetime, timedelta

import secml
from secml.settings import SECML_LOGS_PATH, SECML_STORE_LOGS
from secml.utils import fm, CLog
from secml.utils.download_utils import dl_file_gitlab, md5

from secml.settings import SECML_MODELS_DIR

MODEL_ZOO_REPO_URL = 'https://gitlab.com/secml/secml-zoo'
MODELS_DICT_FILE = 'models_dict.json'
MODELS_DICT_PATH = fm.join(SECML_MODELS_DIR, MODELS_DICT_FILE)

_logger = CLog(
    logger_id=__name__,
    file_handler=SECML_LOGS_PATH if SECML_STORE_LOGS is True else None)


def _dl_data_versioned(file_path, output_dir, md5_digest=None):
    """Download the from different branches depending on version.

    This function tries to download a model zoo resource from:
     1. the branch corresponding to current version,
        e.g. branch `v0.12` for `0.12.*` version
     2. the `master` branch

    Parameters
    ----------
    file_path : str
        Path to the file to download, relative to the repository.
    output_dir : str
        Path to the directory where the file should be stored.
        If folder does not exists, will be created.
    md5_digest : str or None, optional
        Expected MD5 digest of the downloaded file.
        If a different digest is computed, the downloaded file will be
        removed and ValueError is raised.

    """
    try:
        # Try downloading from the branch corresponding to current version
        min_version = re.search(r'^\d+.\d+', secml.__version__).group(0)
        dl_file_gitlab(MODEL_ZOO_REPO_URL, file_path, output_dir,
                       branch='v' + min_version, md5_digest=md5_digest)

    except Exception as e:  # Try looking into 'master' branch...
        _logger.debug(e)
        _logger.debug("Looking in the `master` branch...")
        dl_file_gitlab(MODEL_ZOO_REPO_URL, file_path, output_dir,
                       branch='master', md5_digest=md5_digest)


def _get_models_dict():
    """Downloads the ditionary of models definitions.

    File will be re-downloaded every 30 minutes (upon request) to update
    the models definitions from repository.

    Returns
    -------
    models_dict : dict
        Dictionary with models definitions. Each key is an available model.
        Each model entry is defined by:
         - "model", path to the script with model definition
         - "state", path to the archive containing the pre-saved model state
         - "model_md5", md5 checksum of model definition
         - "state_md5", md5 checksum of pre-saved model state

    """
    # The `.last_update` contains the last time MODELS_DICT_FILE
    # has been download. Read the last update time if this file is available.
    # Otherwise the file will be created later
    last_update_path = fm.join(SECML_MODELS_DIR, '.last_update')
    last_update_format = "%d %m %Y %H:%M"  # Specific format to avoid locale
    current_datetime = datetime.utcnow()  # UTC datetime to avoid locale

    update_models_dict = None  # Trigger flag for model definitions update
    if fm.file_exist(MODELS_DICT_PATH):
        update_models_dict = True  # By default, trigger update
        if fm.file_exist(last_update_path):
            try:
                with open(last_update_path) as fp:
                    last_update = \
                        datetime.strptime(fp.read(), last_update_format)
                    # Compute the threshold for triggering an update
                    last_update_th = last_update + timedelta(minutes=30)
            except ValueError as e:
                # Error occurred while parsing the last update date from file
                # Clean it and re-create later. Definitions update stays True
                _logger.debug(e)  # Log the error for debug purposes
                _logger.debug("Removing `{:}`".format(last_update_path))
                fm.remove_file(last_update_path)
            else:
                # Do not trigger update if last update threshold is not passed
                if current_datetime < last_update_th:
                    update_models_dict = False

    if update_models_dict is not False:
        # if update_models_dict is None means that models dict is not available
        # if it is True means that an update has been triggered
        # Either cases, we need to download the data and extract it

        try:  # Catch download errors

            # Download definitions from current version's branch first,
            # then from master branch
            _dl_data_versioned(MODELS_DICT_FILE, SECML_MODELS_DIR)

        except Exception as e:
            if update_models_dict is None:
                # If update_models_dict is still None, means that models dict
                # is not available, so we propagate the error. Otherwise pass
                raise e
            _logger.debug(e)  # Log the error for debug purposes
            _logger.debug("Error when updating the models definitions. "
                          "Using the last available ones...")

        else:  # No error raised during download process

            # Check if file has been correctly downloaded
            if not fm.file_exist(MODELS_DICT_PATH):
                raise RuntimeError(
                    'Something wrong happened while downloading the '
                    'models definitions. Please try again.')

            # Update or create the "last update" file
            with open(last_update_path, "w") as fp:
                fp.write(current_datetime.strftime(last_update_format))

    with open(MODELS_DICT_PATH) as fp:
        return json.loads(fp.read())


def load_model(model_id):
    """Load a pre-trained classifier.

    Returns a pre-trained SecML classifier given the id of the model.

    Check https://gitlab.com/secml/secml-zoo for the list of available models.

    Parameters
    ----------
    model_id : str
        Identifier of the pre-trained model to load.

    Returns
    -------
    CClassifier
        Desired pre-trained model.

    """
    model_info = _get_models_dict()[model_id]

    model_path = fm.join(SECML_MODELS_DIR, model_info['model'] + '.py')
    # Download (if needed) model's script, check md5 and extract it
    if not fm.file_exist(model_path) or \
            model_info['model_md5'] != md5(model_path):
        model_url_parts = ('models', model_info['model'] + '.py')
        model_url = '/'.join(s.strip('/') for s in model_url_parts)
        out_dir = fm.abspath(model_path)
        # Download requested model from current version's branch first,
        # then from master branch
        _dl_data_versioned(model_url, out_dir, model_info['model_md5'])

        # Check if file has been correctly downloaded
        if not fm.file_exist(model_path):
            raise RuntimeError('Something wrong happened while '
                               'downloading the model. Please try again.')

    state_path = fm.join(SECML_MODELS_DIR, model_info['state'] + '.gz')
    # Download (if needed) state, check md5 and extract it
    if not fm.file_exist(state_path) or \
            model_info['state_md5'] != md5(state_path):
        state_url_parts = ('models', model_info['state'] + '.gz')
        state_url = '/'.join(s.strip('/') for s in state_url_parts)
        out_dir = fm.abspath(state_path)
        # Download requested model state from current version's branch first,
        # then from master branch
        _dl_data_versioned(state_url, out_dir, model_info['state_md5'])

        # Check if file has been correctly downloaded
        if not fm.file_exist(state_path):
            raise RuntimeError('Something wrong happened while '
                               'downloading the model. Please try again.')

    def import_module(full_name, path):
        """Import a python module from a path."""
        from importlib import util

        spec = util.spec_from_file_location(full_name, path)
        mod = util.module_from_spec(spec)

        spec.loader.exec_module(mod)

        return mod

    # Name of the function returning the model
    model_name = model_info["model"].split('/')[-1]

    # Import the python module containing the function returning the model
    model_module = import_module(model_name, model_path)

    # Run the function returning the model
    model = getattr(model_module, model_name)()

    # Restore the state of the model from file
    model.load_state(state_path)

    return model
