import torch
from typing import Union, List, Dict

try:
    from utils import get_activation
    from general_layer import GeneralLayer
except ImportError:
    from Networks.utils import get_activation
    from Networks.general_layer import GeneralLayer


class GeneralInvariantCanonSym(GeneralLayer):
    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int = None,
                 hidden_dims: list = None,
                 activations: Union[List[Dict], List[str]] = None,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(GeneralInvariantCanonSym, self).__init__(input_channels, output_channels,
                                                       hidden_dims if hidden_dims is not None else [],
                                                       activations)

        if hidden_dims is None:
            hidden_dims = []

        self.layer_dims = ([input_dim * input_channels] +
                           [input_dim * d for d in hidden_dims] + [output_dim * self.output_channels])

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(self.layer_dims[0], self.layer_dims[1])
        ])

        for i, k in enumerate(self.activations):
            act = tuple(k.items())[0]
            act, kwargs = act
            self.layers.append(
                get_activation(act, **kwargs)
            )
            self.layers.append(
                torch.nn.Linear(self.layer_dims[i + 1], self.layer_dims[i + 2])
            )
        self.layers = self.layers.to(device=device, dtype=dtype)
        self.output_dim = output_dim

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        pre_processed_x = torch.flatten(self.preprocess(x), 1, -1)
        for layer in self.layers:
            pre_processed_x = layer(pre_processed_x)
        return pre_processed_x.reshape(len(x), self.output_dim, self.output_channels)

