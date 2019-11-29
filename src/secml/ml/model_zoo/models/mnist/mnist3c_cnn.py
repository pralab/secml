"""
.. module:: MNIST3cCNN
   :synopsis: CNN to be trained on MNIST 3-classes dataset

.. moduleauthor:: Maura Pintor <maura.pintor@unica.it>
.. moduleauthor:: Marco Melis <marco.melis@unica.it>

"""
import torch
from torch import nn, optim

from secml.ml.classifiers import CClassifierPyTorch


class MNIST3cCNN(nn.Module):
    """Model with input size (-1, 28, 28) for MNIST 3-classes dataset."""
    def __init__(self):
        super(MNIST3cCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def mnist3c_cnn():

    # Random seed
    torch.manual_seed(0)
    net = MNIST3cCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=0.01, momentum=0.9)

    return CClassifierPyTorch(model=net,
                              loss=criterion,
                              optimizer=optimizer,
                              epochs=20,
                              batch_size=20,
                              input_shape=(1, 28, 28),
                              random_state=0)
