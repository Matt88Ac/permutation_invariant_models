import torch
from typing import Union, List, Dict

try:
    from utils import get_permutations
    from canon_symm_layer import GeneralInvariantCanonSym
except ImportError:
    from Networks.utils import get_permutations
    from Networks.canon_symm_layer import GeneralInvariantCanonSym


class SymmetrizationNetMLP(GeneralInvariantCanonSym):
    def __init__(self, input_dim: int, output_dim: int, input_channels: int, output_channels: int = None,
                 hidden_dims: list = None,
                 activations: Union[List[Dict], List[str]] = None,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(SymmetrizationNetMLP, self).__init__(input_dim, output_dim, input_channels, output_channels,
                                                   hidden_dims, activations, dtype, device)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch-tensor of shape [batch_size, input_dim, feature_dimension]
        :return: a torch-tensor of shape [batch_size, input_dim, feature_dimension], namely avg({ g * x: g \in S_n}),
        """
        batch_size, input_dim, feature_dimension = x.shape

        permutations = get_permutations(input_dim)
        permutations = torch.tensor(permutations, device=x.device, dtype=torch.int)
        x_symmetric = x[:, permutations[0]].clone()
        for sigma in permutations[1:]:
            x_symmetric += x[:, sigma].clone()

        return x_symmetric / len(permutations)

    @property
    def name(self):
        return 'Symmetrization MLP'


if __name__ == '__main__':
    X = torch.randn(10, 3, 7, dtype=torch.float64, device='cuda')
    model = SymmetrizationNetMLP(3, 1, 7, 2, hidden_dims=[5, 2, 18, 4],  # inv_dim=-1,
                                 activations=['relu', 'silu', {'softmax': dict(dim=1)}, 'relu'], dtype=torch.float64,
                                 device='cuda')
    perm = torch.randperm(X.shape[1])
    Y = X[:, perm]
    # perm = torch.randperm(X.shape[2])
    # Y = X[:, :, perm]
    mx = model(X)
    my = model(Y)
    print(((mx - my) / mx).abs().mean())
