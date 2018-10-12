'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from secml.core.settings import SECML_PYTORCH_DATA_DIR, SECML_PYTORCH_MODELS_DIR
from secml.utils import fm

from secml.data.loader import CDataLoaderCIFAR10
from secml.pytorch.data import CTorchDataset

from secml.pytorch.models import dl_pytorch_model
import secml.pytorch.models.cifar as models

from secml.pytorch.utils import AverageMeter, accuracy


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
use_cuda = torch.cuda.is_available()
print(use_cuda)

# Random seed
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(manualSeed)

best_acc = 0  # best test accuracy

dataset = 'cifar10'
# model_name = 'densenet-bc-L190-K40'
# depth = 190
# growthRate = 40

model_name = 'densenet-bc-L100-K12'
depth = 100
growthRate = 12

def main():
    global best_acc

    # Data
    print('==> Preparing dataset %s' % dataset)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100

    tr, ts = CDataLoaderCIFAR10().load()

    transform = transforms.Compose(
        [transforms.Lambda(lambda x: x.reshape([3, 32, 32])),
         transforms.Lambda(lambda x: x.transpose([1, 2, 0])),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2023, 0.1994, 0.2010))])

    trainset = CTorchDataset(tr, transform=transform)

    # trainset = datasets.CIFAR10(root=fm.join(SECML_PYTORCH_DATA_DIR, 'data'),
    #                             train=True, download=True,
    #                             transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=25,
                                              shuffle=True, num_workers=1)

    testset = CTorchDataset(ts, transform=transform)

    # testset = datasets.CIFAR10(root=fm.join(SECML_PYTORCH_DATA_DIR, 'data'),
    #                            train=False, download=True,
    #                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=25,
                                             shuffle=False, num_workers=1)

    # Model
    model = models.__dict__['densenet'](
                num_classes=num_classes,
                depth=depth,
                growthRate=growthRate,
                compressionRate=2,
                dropRate=0,
            )
    model = torch.nn.DataParallel(model).cuda()

    state = dl_pytorch_model(model_name)

    epoch = state['epoch']
    best_acc = state['best_acc']

    model.load_state_dict(state['state_dict'])

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()

    print('\nEvaluation only')
    test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
    print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
    return


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(testloader):

        print("Batch {:}".format(batch_idx))

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    return (losses.avg, top1.avg)


if __name__ == '__main__':
    main()
