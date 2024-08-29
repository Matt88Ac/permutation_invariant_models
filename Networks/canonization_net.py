import torch
from typing import Union, List, Dict

try:
    from canon_symm_layer import GeneralInvariantCanonSym
except ImportError:
    from Networks.canon_symm_layer import GeneralInvariantCanonSym


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


if __name__ == '__main__':
    X = torch.randn(10, 3, 7, dtype=torch.float64, device='cuda')
    model = CanonizationNetMLP(3, 1, 7, hidden_dims=[5, 2, 18, 4],  # inv_dim=-1,
                               activations=['relu', 'silu', {'softmax': {'dim': 1}}, 'relu'], dtype=torch.float64,
                               device='cuda')
    perm = torch.randperm(X.shape[1])
    Y = X[:, perm]
    # perm = torch.randperm(X.shape[2])
    # Y = X[:, :, perm]
    print(model)
    print((model(X) - model(Y)).abs().max())
