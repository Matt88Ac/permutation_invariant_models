import math
from itertools import permutations
from random import sample
from typing import Optional

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
    return sample(indices, k=k,)


def fix_activations(activations):
    if len(activations) > 0:
        for i in range(len(activations)):
            if isinstance(activations[i], str):
                activations[i] = {activations[i]: dict()}
    return activations


def positional_encoding(x: torch.Tensor) -> torch.Tensor:
    b, n, d = x.shape
    position = torch.arange(0, n, dtype=x.dtype, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2, device=x.device, dtype=x.dtype) * -(math.log(10000.0) / d)).unsqueeze(0)

    pos_enc = torch.zeros(1, n, d, device=x.device, dtype=x.dtype)
    enc = position * div_term
    pos_enc[0, :, 0::2] = torch.sin(enc)
    if d % 2:
        pos_enc[0, :, 1::2] = torch.cos(enc)[:, :-1]
    else:
        pos_enc[0, :, 1::2] = torch.cos(enc)

    return x + pos_enc
