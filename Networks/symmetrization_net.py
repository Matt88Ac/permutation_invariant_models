import torch
from typing import Union, List, Dict

try:
    from utils import get_activation, get_permutations
except ImportError:
    from Networks.utils import get_activation, get_permutations


class SymmetrizationNetMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 layer_dims: list = None, activations: Union[List[Dict], List[str]] = None, inv_dim: int = 1,
                 dtype=torch.float64, device=torch.device('cpu')):
        super(SymmetrizationNetMLP, self).__init__()

        if layer_dims is None:
            layer_dims = []
        if activations is None:
            activations = ['relu'] * len(layer_dims) if len(layer_dims) > 0 else []

        assert len(layer_dims) == len(activations)
        if len(activations) > 0:
            for i in range(len(activations)):
                if isinstance(activations[i], str):
                    activations[i] = {activations[i]: dict()}

        self.layer_dims = [input_dim] + layer_dims + [output_dim]
        self.layers = torch.nn.ModuleList([
            torch.nn.Conv1d(input_dim, self.layer_dims[1], kernel_size=1)
        ])
        for i, k in enumerate(activations):
            act = tuple(k.items())[0]
            act, kwargs = act
            self.layers.append(
                get_activation(act, **kwargs)
            )
            self.layers.append(
                torch.nn.Conv1d(self.layer_dims[i + 1], self.layer_dims[i + 2], kernel_size=1)
            )
        self.layers = self.layers.to(device=device, dtype=dtype)
        self.inv_dim = inv_dim

    def symmetrize(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: a torch-tensor of shape [batch_size, input_dim, feature_dimension]
        :return: a torch-tensor of shape [batch_size, input_dim, feature_dimension], namely avg({ g * x: g \in S_n}),
        """
        batch_size, input_dim, feature_dimension = x.shape
        if self.inv_dim == 1:
            permutations = get_permutations(input_dim)
            permutations = torch.tensor(permutations, device=x.device, dtype=torch.int)
            x_symmetric = x[:, permutations[0]].clone()
            for sigma in permutations[1:]:
                x_symmetric += x[:, sigma].clone()
        else:
            permutations = get_permutations(feature_dimension)
            permutations = torch.tensor(permutations, device=x.device, dtype=torch.int)
            x_symmetric = x[:, :, permutations[0]].clone()

            for sigma in permutations[1:]:
                x_symmetric += x[:, :, sigma].clone()

        return x_symmetric / len(permutations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        symmetrize_x = self.symmetrize(x)
        for layer in self.layers:
            symmetrize_x = layer(symmetrize_x)
        return symmetrize_x


if __name__ == '__main__':
    X = torch.randn(10, 3, 7, dtype=torch.float64, device='cuda')
    model = SymmetrizationNetMLP(3, 15, layer_dims=[5, 2, 18, 4],  # inv_dim=-1,
                                 activations=['relu', 'silu', {'softmax': dict(dim=1)}, 'relu'], dtype=torch.float64, device='cuda')
    perm = torch.randperm(X.shape[1])
    Y = X[:, perm]
    # perm = torch.randperm(X.shape[2])
    # Y = X[:, :, perm]
    mx = model(X)
    my = model(Y)
    print(((mx - my)/mx).abs().max())
