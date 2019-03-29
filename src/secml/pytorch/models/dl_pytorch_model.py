"""
.. module:: DownloadPyTorchModel
   :synopsis: Functions to allow download pre-trained PyTorch models

.. moduleauthor:: Marco Melis <marco.melis@diee.unica.it>

"""
import torch

from secml.pytorch.settings import SECML_PYTORCH_MODELS_DIR
from secml.utils import fm
from secml.utils.download_utils import dl_file

PYTORCH_MODELS_URL = 'https://nue.diee.unica.it/public.php/webdav'
PYTORCH_MODELS_FILENAME = 'model_best.pth.tar'
PYTORCH_MODELS = {  # (fileid, md5)
    'cifar10': {
        'alexnet': ('cd7f76cda281b355ffeba66ddfac7d8f',
                    '47e1279b66a4c06f2368f9f2267a3747'),
        'densenet-bc-L100-K12': ('799286b83f9a530470cc56975c682586',
                                 '9642927b28d4dbfdd8841bb578bd2fd7'),
        'densenet-bc-L190-K40': ('8281038af7203f8d06640815afa2a15c',
                                 '2fa699043b33d5a8b25b6462ec9c29fe'),
        'preresnet-110': ('382d41f7717bfaef848a49c90796c4d5',
                          '4f584ee3cb2569f1f3f223968e8208fe'),
        'resnet-110': ('af1edbe8bb67734c399bf2dc873a6ece',
                       '137c5c3f1cc5906bc690c3862469e0e3'),
        'resnext-8x64': ('01947b212f6fdaefb5f01b292d207736',
                         '545c71016bd338d2fc0abb5509d1396f'),
        'resnext-16x64': ('d00e2983008bb255fbca1cafcf95bb8f',
                          '37797513c38d12f6346e26c8093e9b62'),
        'vgg19-bn': ('2e8e4c04a95f391546bd4b2febfc2f6d',
                     'a619f73e6bb4e6f1eb1676f63354d12d'),
        'wrn-28-10-drop': ('81237a00b2c77e6ed3189b790a75cef2',
                           '2ab8ff8333c2d444e34dd8f21f10e8c9')
    },
    'cifar100': {
        'alexnet': ('ca1c6c0d9246912aa0eebf37f37b9f3a',
                    '9ce0147751c77af6c818b2afc1d6c111'),
        'densenet-bc-L190-k40': ('31509675a4d4d858151a0361c53b2304',
                                 'c2911ec501555402b550967797fcdea9'),
        'preresnet-110': ('90bdc67aceb82dfd25ddbb80e1f829d5',
                          '91277e5063ff9bced55190555aa11a60'),
        'resnet-110': ('7ee53f71cc3746387b67e81dec482866',
                       '558d05f2e3d47c22eecf765c58764bc4'),
        'resnext-8x64': ('7ce060849fc0153b171a323d6043f3a5',
                         '3d45c8114d230e1c85d2f46eba3889ad'),
        'resnext-16x64': ('0c2378f4371f4678ddaa38fce194e687',
                          '87ef8ad82ffa418262ba2847e103712e'),
        'vgg19-bn': ('44ed73a8fe629e2fafa6e1a678fa4a97',
                     'ae52ea757340837a7faa214b87c3f09d'),
        'wrn-28-10-drop': ('200772aca6deed50f2073afb82bdda18',
                           'cea3705079539cb38ab87874569abe3e')
    },
    'imagenet': {
        'resnext50-32x4': ('740ff38f9a71c0208125b84b09a946a6',
                           '621da0e387fe43802f8969b41afe0f1a')
    }
}


def dl_pytorch_model(ds_id, model_id):
    """Download a pre-trained PyTorch model.

    Returns a pre-trained PyTorch model given its id and the dataset id.
    The list of available models can be found at:
     `secml.pytorch.models.PYTORCH_MODELS`

    Parameters
    ----------
    ds_id : str
        Identifier of the dataset on which the model is pre-trained.
    model_id : str
        Identifier of the pre-trained model to download.

    Returns
    -------
    state_dict : dict
        Dictionary of the state of the model.

    """
    data_path = fm.join(
        SECML_PYTORCH_MODELS_DIR, ds_id, model_id, PYTORCH_MODELS_FILENAME)
    model_info = PYTORCH_MODELS[ds_id][model_id]
    # Download (if needed) data and extract it
    if not fm.file_exist(data_path):
        f_dl = dl_file(PYTORCH_MODELS_URL,
                       fm.join(SECML_PYTORCH_MODELS_DIR, ds_id, model_id),
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
from secml.testing import CUnitTest


class CTestDLPytorchModel(CUnitTest):

    def test_dl_pytorch_model(self):

        model_id = 'densenet-bc-L100-K12'  # Model to test
        ds_id = 'cifar10'  # ds on which the model is trained

        model_state = dl_pytorch_model(ds_id, model_id)

        # Check if the model file has been downloaded in the correct location
        self.assertTrue(fm.file_exist(
            fm.join(SECML_PYTORCH_MODELS_DIR, ds_id, model_id,
                    PYTORCH_MODELS_FILENAME)))
        # Check if the returned state has the required keys
        self.assertIn('state_dict', model_state)


if __name__ == '__main__':
    CUnitTest.main()
