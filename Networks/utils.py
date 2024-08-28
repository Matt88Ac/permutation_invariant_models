if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch


def get_activation(activation_name: str, *args, **kwargs):
    if activation_name.lower() == 'relu':
        return torch.nn.ReLU()
    elif activation_name.lower() in ['leaky_relu', 'leaky']:
        return torch.nn.LeakyReLU(*args, **kwargs)
    elif activation_name.lower() == 'silu':
        return torch.nn.SiLU()
    elif activation_name.lower() == 'elu':
        return torch.nn.ELU(*args, **kwargs)
    elif activation_name.lower() == 'softmax':
        return torch.nn.Softmax(*args, **kwargs)
    elif activation_name.lower() == 'softplus':
        return torch.nn.Softplus(*args, **kwargs)
    elif activation_name.lower() == 'tanh':
        return torch.nn.Tanh(*args, **kwargs)

    return torch.nn.Identity(*args, **kwargs)