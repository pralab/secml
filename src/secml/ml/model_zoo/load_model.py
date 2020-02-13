"""
.. module:: LoadModel
   :synopsis: Functions to load pre-trained SecML models

.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import json
import re

import secml
from secml.utils import fm
from secml.utils.download_utils import dl_file_gitlab

from secml.settings import SECML_MODELS_DIR

MODEL_ZOO_REPO_URL = 'https://gitlab.com/secml/secml-zoo'
MODELS_DICT_PATH = fm.join(fm.abspath(__file__), 'models_dict.json')
with open(MODELS_DICT_PATH) as fp:
    MODELS_DICT = json.loads(fp.read())


def load_model(model_id):
    """Load a pre-trained classifier.

    Returns a pre-trained SecML classifier given the id of the model.

    The following models are available:
     - `mnist-svm`,
            multiclass `CClassifierSVM` trained on MNIST
     - `mnist59-svm`,
            multiclass `CClassifierSVM` with RBF Kernel trained on MNIST59
     - `mnist59-svm-rbf`,
            multiclass `CClassifierSVM` with RBF Kernel trained on MNIST59
     - `mnist159-cnn`,
            `CClassifierPyTorch` CNN trained on MNIST159

    Parameters
    ----------
    model_id : str
        Identifier of the pre-trained model to load.

    Returns
    -------
    CClassifier
        Desired pre-trained model.

    """
    model_info = MODELS_DICT[model_id]
    data_path = fm.join(SECML_MODELS_DIR, model_id, model_id + '.gz')
    # Download (if needed) data and extract it
    if not fm.file_exist(data_path):
        model_url = 'models/' + model_info['url'] + '.gz'
        out_dir = fm.join(SECML_MODELS_DIR, model_id)
        try:
            # Try downloading from the branch corresponding to current version
            min_version = re.search(r'^\d+.\d+', secml.__version__).group(0)
            dl_file_gitlab(MODEL_ZOO_REPO_URL, model_url, out_dir,
                           branch='v' + min_version,
                           md5_digest=model_info['md5'])

        except RuntimeError:
            # Raised if file not found. Try looking in 'master' branch
            dl_file_gitlab(MODEL_ZOO_REPO_URL, model_url, out_dir,
                           branch='master',
                           md5_digest=model_info['md5'])

        # Check if file has been correctly downloaded
        if not fm.file_exist(data_path):
            raise RuntimeError('Something wrong happened while '
                               'downloading the model. Please try again')

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
    model_path = fm.join(
        fm.abspath(__file__), 'models', model_info["model"] + '.py')
    model_module = import_module(model_name, model_path)

    # Run the function returning the model
    model = getattr(model_module, model_name)()

    # Restore the state of the model from file
    model.load_state(data_path)

    return model
