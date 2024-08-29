from Networks.layers import *
from symmetry_tests.testing_utils import test_invariance, test_equivariance, gen_random_data
import torch
import numpy as np


def gen_parameters(max_in_dim: int = 8, max_in_channels: int = 5, max_out_channels: int = 5,
                   max_n_hidden: int = 7, max_scale: float = 20):
    in_dim = np.random.randint(2, max_in_dim, size=1)[0]
    in_channels = np.random.randint(2, max_in_channels, size=1)[0]
    out_channels = np.random.randint(2, max_out_channels, size=1)[0]

    n_hidden = np.random.randint(1, max_n_hidden, size=1)[0]
    hidden = np.random.randint(1, max_in_channels, size=n_hidden)

    scale = np.random.uniform(1, max_scale, 1)[0]

    return in_dim, in_channels, out_channels, hidden, scale


def test_model(model_type: MODEL_TYPES, batch_size: int, max_in_dim: int = 8, max_in_channels: int = 5,
               max_out_channels: int = 5,
               max_n_hidden: int = 7, max_scale: float = 20, thr: float = 1e-5,
               device=torch.device('cpu'), dtype=torch.float64, **other_model_parameters):
    in_dim, in_channels, out_channels, hidden, scale = gen_parameters(max_in_dim, max_in_channels, max_out_channels,
                                                                      max_n_hidden, max_scale)

    model = get_models(model_type)(in_dim, 1, in_channels, out_channels, hidden.tolist(),
                                   device=device, dtype=dtype, **other_model_parameters)
    x = gen_random_data(batch_size, (in_dim, in_channels), scale, device, dtype)
    INV = test_invariance(model, x, thr)

    model = get_models(model_type)(in_dim, 1, in_channels, out_channels, hidden.tolist(),
                                   device=device, dtype=dtype, **other_model_parameters)
    x = gen_random_data(batch_size, (in_dim, in_channels), scale, device, dtype)

    EQ = test_equivariance(model, x, thr)
    return INV or EQ

