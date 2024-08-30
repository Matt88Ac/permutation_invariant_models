import torch
from typing import Union, List, Dict

try:
    from canon_symm_layer import GeneralInvariantCanonSym, positional_encoding
except ImportError:
    from Networks.canon_symm_layer import GeneralInvariantCanonSym, positional_encoding


class CanonizationNetMLP(GeneralInvariantCanonSym):
    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int = None,
                 hidden_dims: list = None,
                 activations: Union[List[Dict], List[str]] = None,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(CanonizationNetMLP, self).__init__(input_dim, output_dim, input_channels, output_channels,
                                                 hidden_dims, activations, dtype, device)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch-tensor of shape [batch_size, input_dim, feature_dimension]
        :return: a torch-tensor of shape [batch_size, input_dim, feature_dimension],
            canonized by l2 norm feature-wise
        """
        x_sort = x.norm(p=2, dim=-1, keepdim=True).argsort(dim=1).expand_as(x)
        return torch.gather(x, 1, x_sort)

    @property
    def name(self):
        return 'Canonization MLP'


class CanonizationNetPosEncode(GeneralInvariantCanonSym):
    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int = None,
                 hidden_dims: list = None,
                 activations: Union[List[Dict], List[str]] = None,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(CanonizationNetPosEncode, self).__init__(input_dim, output_dim, input_channels, output_channels,
                                                       hidden_dims, activations, dtype, device)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch-tensor of shape [batch_size, input_dim, feature_dimension]
        :return: a torch-tensor of shape [batch_size, input_dim, feature_dimension],
            canonized by l2 norm feature-wise and position-encoded
        """
        x_sort = x.norm(p=2, dim=-1, keepdim=True).argsort(dim=1).expand_as(x)
        return positional_encoding(torch.gather(x, 1, x_sort))

    @property
    def name(self):
        return 'Canonization Transformer'
