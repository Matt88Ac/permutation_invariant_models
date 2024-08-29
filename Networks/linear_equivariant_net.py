import torch
from typing import Union, List, Dict

try:
    from utils import get_activation
    from general_layer import GeneralLayer
except ImportError:
    from Networks.utils import get_activation
    from Networks.general_layer import GeneralLayer


class EquivariantLinear(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(EquivariantLinear, self).__init__()
        self.alpha = torch.nn.Linear(input_channels, output_channels, bias=False, device=device, dtype=dtype)
        self.beta = torch.nn.Linear(input_channels, output_channels, bias=True, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha(x) + self.beta(torch.mean(x, dim=1, keepdim=True))


class EquivariantNet(GeneralLayer):
    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int = None,
                 hidden_dims: list = None,
                 activations: Union[List[Dict], List[str]] = None,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(EquivariantNet, self).__init__(input_channels, output_channels,
                                             hidden_dims if hidden_dims is not None else [],
                                             activations)

        if hidden_dims is None:
            hidden_dims = []

        assert output_dim in [input_dim, 1]
        self.layer_dims = [input_channels] + hidden_dims + [self.output_channels]

        self.layers = torch.nn.ModuleList([
            EquivariantLinear(self.layer_dims[0], self.layer_dims[1], device=device, dtype=dtype)
        ])
        for i, k in enumerate(self.activations):
            act = tuple(k.items())[0]
            act, kwargs = act
            self.layers.append(
                get_activation(act, **kwargs).to(device=device, dtype=dtype)
            )
            self.layers.append(
                EquivariantLinear(self.layer_dims[i + 1], self.layer_dims[i + 2], device=device, dtype=dtype)
            )
        if output_dim == 1:
            self.layers.append(torch.nn.AvgPool2d((input_dim, 1)))
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = x.clone()
        for layer in self.layers:
            x_out = layer(x_out)
        return x_out

    @property
    def name(self):
        return 'Linear Equivariant Net'
