from .c_dataloader import CDataLoader
from .c_dataloader_sklearn import *
from .c_dataloader_svmlight import CDataLoaderSvmLight
from .c_dataloader_imgclients import CDataLoaderImgClients
from .c_dataloader_imgfolders import CDataLoaderImgFolders
from .c_dataloader_mnist import CDataLoaderMNIST
from .c_dataloader_lfw import CDataLoaderLFW
from .c_dataloader_cifar import CDataLoaderCIFAR10, CDataLoaderCIFAR100
from .c_dataloader_icubworld import CDataLoaderICubWorld28

try:
    import torch
    import torchvision
except ImportError:
    pass  # pytorch is an extra component
else:
    from .c_dataloader_pytorch import CDataLoaderPyTorch
    from .c_dataloader_torchvision import CDataLoaderTorchDataset
