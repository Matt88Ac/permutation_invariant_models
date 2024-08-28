if __name__ == '__main__':
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from typing import Union, List, Dict

try:
    from utils import get_activation
except ImportError:
    from Networks.utils import get_activation


class CanonizationNetMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 layer_dims: list = None, activations: Union[List[Dict], List[str]] = None, inv_dim: int = 1,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(CanonizationNetMLP, self).__init__()

        if layer_dims is None:
            layer_dims = []
        if activations is None:
            activations = ['relu'] * len(layer_dims) if len(layer_dims) > 0 else []

        assert len(layer_dims) == len(activations)
        if len(activations) > 0:
            for i in range(len(activations)):
                if isinstance(activations[i], str):
                    activations[i] = {activations[i]: ([], dict())}

        self.layer_dims = [input_dim] + layer_dims + [output_dim]
        self.layers = torch.nn.ModuleList([
            torch.nn.Conv1d(input_dim, self.layer_dims[1], kernel_size=1)
        ])
        for i, k in enumerate(activations):
            act = tuple(k.items())[0]
            act, (args, kwargs) = act
            self.layers.append(
                get_activation(act, *args, **kwargs)
            )
            self.layers.append(
                torch.nn.Conv1d(self.layer_dims[i + 1], self.layer_dims[i + 2], kernel_size=1)
            )
        self.layers = self.layers.to(device=device, dtype=dtype)
        self.inv_dim = inv_dim

    def canonize(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch-tensor of shape [batch_size, input_dim, feature_dimension]
        :return: a torch-tensor of shape [batch_size, input_dim, feature_dimension],
            canonized by l2 norm feature-wise
        """
        if self.inv_dim == 1:
            x_sort = x.norm(p=2, dim=-1, keepdim=True).argsort(dim=1).expand_as(x)
        else:
            x_sort = x.norm(p=2, dim=1, keepdim=True).argsort(dim=-1).expand_as(x)
        return torch.gather(x, self.inv_dim, x_sort)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        canonized_x = self.canonize(x)
        for layer in self.layers:
            canonized_x = layer(canonized_x)
        return canonized_x


if __name__ == '__main__':
    X = torch.randn(10, 3, 7, dtype=torch.float64)
    model = CanonizationNetMLP(3, 15, layer_dims=[5, 2, 18, 4], # inv_dim=-1,
                               activations=['relu', 'silu', {'softmax': ([1], {})}, 'relu'], dtype=torch.float64)
    perm = torch.randperm(X.shape[1])
    Y = X[:, perm]
    #perm = torch.randperm(X.shape[2])
    #Y = X[:, :, perm]
    print((model(X) - model(Y)).abs().max())
