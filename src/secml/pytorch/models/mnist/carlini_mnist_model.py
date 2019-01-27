'''
The CNN learned on the MNIST dataset by Carlini in the paper:
N. Carlini and D. A. Wagner, "Adversarial examples are not easily
detected: Bypassing ten detection methods"

ref to the Carlini's code:
https://github.com/carlini/nn_breaking_detection/blob/master/setup_mnist.py
'''
import torch.nn as nn

__all__ = ['carlini_mnist_model']


class CarliniMNISTModel(nn.Module):

    def __init__(self, num_classes=10):
        nb_filters = 64

        super(CarliniMNISTModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, nb_filters, kernel_size=(5, 5), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters,
                      kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters,
                      kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            # Flatten(),
            nn.Linear(64, out_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=.5, inplace=False)
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def carlini_mnist_model(**kwargs):
    model = CarliniMNISTModel(**kwargs)
    return model




#model = carlini_mnist_model()
