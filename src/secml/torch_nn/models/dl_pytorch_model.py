"""
.. module:: DownloadPyTorchModel
   :synopsis: Functions to allow download pre-trained PyTorch models

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import torch
from secml.core.settings import PYTORCH_MODELS_DIR
from secml.utils import fm
from secml.utils.download_utils import dl_file

PYTORCH_MODELS_URL = 'https://nue.diee.unica.it/public.php/webdav'
PYTORCH_MODELS_FILENAME = 'model_best.pth.tar'
PYTORCH_MODELS = {
    'densenet-bc-L100-K12': ('e3e7eadd83eedd49227644b1582c8d47',
                             '9642927b28d4dbfdd8841bb578bd2fd7')
}


def dl_pytorch_model(model_id):
    """Download a pre-trained PyTorch model.

    Returns a pre-trained pytorch model given its id.
    The list of available models can be found at:
     `secml.torch_nn.models.PYTORCH_MODELS`

    Parameters
    ----------
    model_id : str
        Identifier of the pre-trained model to download

    Returns
    -------
    state_dict : dict
        Dictionary of the state of the model.

    """
    data_path = fm.join(PYTORCH_MODELS_DIR, model_id, PYTORCH_MODELS_FILENAME)
    model_info = PYTORCH_MODELS[model_id]
    # Download (if needed) data and extract it
    if not fm.file_exist(data_path):
        f_dl = dl_file(PYTORCH_MODELS_URL,
                       fm.join(PYTORCH_MODELS_DIR, model_id),
                       user=model_info[0], md5_digest=model_info[1])
        # Copy downloaded file to the expected place and remove temp file
        fm.copy_file(f_dl, data_path)
        fm.remove_file(f_dl)

        # Check if file has been correctly downloaded
        if not fm.file_exist(data_path):
            raise RuntimeError('Something wrong happened while '
                               'downloading the model. Please try again')

    # If CUDA is not available, map the model to cpu
    map_location = 'cpu' if torch.cuda.is_available() is False else None

    return torch.load(data_path, map_location=map_location)


# UNITTESTS
from secml.utils import CUnitTest


class CTestDLPytorchModel(CUnitTest):

    def test_dl_pytorch_model(self):

        model_id = 'densenet-bc-L100-K12'  # Model to test

        model_state = dl_pytorch_model(model_id)

        # Check if the model file has been downloaded in the correct location
        self.assertTrue(fm.file_exist(
            fm.join(PYTORCH_MODELS_DIR, model_id, PYTORCH_MODELS_FILENAME)))
        # Check if the returned state has the required keys
        self.assertIn('state_dict', model_state)


if __name__ == '__main__':
    CUnitTest.main()
