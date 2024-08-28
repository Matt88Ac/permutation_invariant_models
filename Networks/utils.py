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


class MultiChannelMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, feature_dimension: int,
                 include_bias: bool = True,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(MultiChannelMLP, self).__init__()
        self.input_dim = input_dim
        self.feature_dimension = feature_dimension
        self.output_dim = output_dim

        self.weights = torch.nn.Parameter(
            torch.randn(output_dim, input_dim, feature_dimension, dtype=dtype, device=device)
        )

        self.bias = None
        if include_bias:
            self.bias = torch.nn.Parameter(
                torch.randn(output_dim, feature_dimension, dtype=dtype, device=device)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch-tensor of shape [batch_size, input_dim, feature_dimension]
        :return:
        """

