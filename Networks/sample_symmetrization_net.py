import torch
from typing import Union, List, Dict

try:
    from utils import get_permutations, positional_encoding
    from canon_symm_layer import GeneralInvariantCanonSym
except ImportError:
    from Networks.utils import get_permutations, positional_encoding
    from Networks.canon_symm_layer import GeneralInvariantCanonSym


class SampleSymmetrizationNetMLP(GeneralInvariantCanonSym):
    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int = None,
                 hidden_dims: list = None,
                 activations: Union[List[Dict], List[str]] = None,
                 sample_rate: float = 0.05,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(SampleSymmetrizationNetMLP, self).__init__(input_dim, output_dim, input_channels, output_channels,
                                                         hidden_dims, activations, dtype, device)
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: a torch-tensor of shape [batch_size, input_dim, feature_dimension]
        :return: a torch-tensor of shape [batch_size, input_dim, feature_dimension],
                namely avg({ g * x: g \in subset_of_S_n }),
        """
        batch_size, input_dim, feature_dimension = x.shape
        permutations = get_permutations(input_dim, select=self.sample_rate)
        permutations = torch.tensor(permutations, device=x.device, dtype=torch.int)
        x_symmetric = super().forward(x[:, permutations[0]].clone())
        for sigma in permutations[1:]:
            # x_symmetric += x[:, sigma].clone()
            x_symmetric = super().forward(x[:, sigma].clone())

        return x_symmetric / len(permutations)

    @property
    def name(self):
        return 'Sampled Symmetrization MLP'


class SampleSymmetrizationNetPosEncode(GeneralInvariantCanonSym):
    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int = None,
                 hidden_dims: list = None,
                 activations: Union[List[Dict], List[str]] = None,
                 sample_rate: float = 0.05,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(SampleSymmetrizationNetPosEncode, self).__init__(input_dim, output_dim, input_channels, output_channels,
                                                               hidden_dims, activations, dtype, device)
        self.sample_rate = sample_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        :param x: a torch-tensor of shape [batch_size, input_dim, feature_dimension]
        :return: a torch-tensor of shape [batch_size, input_dim, feature_dimension],
                namely avg({ g * x: g \in subset_of_S_n }),
        """
        batch_size, input_dim, feature_dimension = x.shape
        permutations = get_permutations(input_dim, select=self.sample_rate)
        permutations = torch.tensor(permutations, device=x.device, dtype=torch.int)
        x = positional_encoding(x)

        # x_symmetric = x[:, permutations[0]].clone()
        x_symmetric = super().forward(x[:, permutations[0]].clone())
        for sigma in permutations[1:]:
            # x_symmetric += x[:, sigma].clone()
            x_symmetric = super().forward(x[:, sigma].clone())

        return x_symmetric / len(permutations)

    @property
    def name(self):
        return 'Sampled Symmetrization Transformer'
