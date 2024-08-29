from itertools import permutations
from typing import Optional
from random import choices
import torch


def get_activation(activation_name: str, **kwargs):
    if activation_name.lower() == 'relu':
        return torch.nn.ReLU()
    elif activation_name.lower() in ['leaky_relu', 'leaky']:
        return torch.nn.LeakyReLU(**kwargs)
    elif activation_name.lower() == 'silu':
        return torch.nn.SiLU()
    elif activation_name.lower() == 'elu':
        return torch.nn.ELU(**kwargs)
    elif activation_name.lower() == 'softmax':
        return torch.nn.Softmax(**kwargs)
    elif activation_name.lower() == 'softplus':
        return torch.nn.Softplus(**kwargs)
    elif activation_name.lower() == 'tanh':
        return torch.nn.Tanh(**kwargs)
    elif activation_name.lower() == 'sigmoid':
        return torch.nn.Sigmoid(**kwargs)

    return torch.nn.Identity(**kwargs)


def get_permutations(dim: int, select: Optional[float] = None):
    indices = list(permutations(range(dim)))
    if select is None:
        return indices
    k = max(int(select * len(indices)), 1)
    return choices(indices, k=k)


def fix_activations(activations):
    if len(activations) > 0:
        for i in range(len(activations)):
            if isinstance(activations[i], str):
                activations[i] = {activations[i]: dict()}
    return activations
