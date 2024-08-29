import torch
from typing import Union, List, Dict

try:
    from utils import get_activation, fix_activations
except ImportError:
    from Networks.utils import get_activation, fix_activations


class GeneralLayer(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int = None,
                 hidden_dims: list = None,
                 activations: Union[List[Dict], List[str]] = None):
        super(GeneralLayer, self).__init__()

        if activations is None:
            activations = ['relu'] * len(hidden_dims) if len(hidden_dims) > 0 else []
        if output_channels is None:
            output_channels = input_channels

        assert len(hidden_dims) == len(activations)
        self.activations = fix_activations(activations)
        self.output_channels = output_channels

    @property
    def name(self):
        return 'General'
