import torch
from collections import OrderedDict


def mlp(input_dims=1000, hidden_dims=(100, 100), output_dims=10):
    """Multi-layer Perceptron"""
    if len(hidden_dims) < 1:
        raise ValueError("at least one hidden dim should be defined")
    if any(d <= 0 for d in hidden_dims):
        raise ValueError("each hidden layer must have at least one neuron")

    # Input layers
    layers = [
        ('linear1', torch.nn.Linear(input_dims, hidden_dims[0])),
        ('relu1', torch.nn.ReLU()),
    ]
    # Appending additional hidden layers
    for hl_i, hl_dims in enumerate(hidden_dims[1:]):
        prev_hl_dims = hidden_dims[hl_i]  # Dims of the previous hl
        i_str = str(hl_i + 2)
        layers += [
            ('linear' + i_str, torch.nn.Linear(prev_hl_dims, hl_dims)),
            ('relu' + i_str, torch.nn.ReLU())]
    # Output layers
    layers += [
        ('linear' + str(len(hidden_dims) + 1),
         torch.nn.Linear(hidden_dims[-1], output_dims)),
        ('softmax', torch.nn.Softmax(dim=-1))]

    # Creating the model with the list of layers
    return torch.nn.Sequential(OrderedDict(layers))
