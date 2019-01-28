'''
The CNN learned on the MNIST dataset by Carlini in the paper:
N. Carlini and D. A. Wagner, "Adversarial examples are not easily
detected: Bypassing ten detection methods"

ref to the Carlini's code:
https://github.com/carlini/nn_breaking_detection/blob/master/setup_mnist.py
'''
import torch.nn as nn

__all__ = ['carlini_mnist_model']


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x

class CheckShape(nn.Module):
    def __init__(self):
        super(CheckShape, self).__init__()

    def forward(self, x):
        print x.shape
        return x

class CarliniMNISTModel(nn.Module):

    def __init__(self, num_classes=10):
        nb_filters = 64

        super(CarliniMNISTModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, nb_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters,
                      kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters,
                      kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(576, out_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5, inplace=False),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    #
    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     return x


def carlini_mnist_model(**kwargs):
    model = CarliniMNISTModel(**kwargs)
    return model
