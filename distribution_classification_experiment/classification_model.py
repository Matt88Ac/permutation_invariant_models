import torch
from torch import nn

from Networks.canon_symm_layer import GeneralInvariantCanonSym
from Networks.layers import get_models, MODEL_TYPES


class RandomPermutation(nn.Module):
    """randomly permutes elements in a sample.
    Args:
        dim (int): Desired dimension of permutation.
    """

    def __init__(self, dim: int):
        super(RandomPermutation, self).__init__()
        assert isinstance(dim, int)
        self.dim = dim

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        perm = torch.randperm(self.dim)
        return sample.clone()[:, perm]


class DistributionClassifier(nn.Module):
    def __init__(self, model_type: MODEL_TYPES, distribution_shape: tuple,
                 hidden_layers=None, device=torch.device('cpu'), dtype=torch.float64,
                 **other_model_parameters):
        super(DistributionClassifier, self).__init__()
        if hidden_layers is None:
            hidden_layers = [16, 64, 128, 64, 16]
        assert len(distribution_shape) == 2
        self.dist_shape = distribution_shape
        N, D = distribution_shape
        try:
            self.inv_model = get_models(model_type)(N, 1, D, D, hidden_layers, device=device,
                                                    dtype=dtype, **other_model_parameters)
            self.name = self.inv_model.name
        except NotImplementedError:
            # Augmentation based model
            self.inv_model = nn.Sequential(RandomPermutation(N), GeneralInvariantCanonSym(N, 1, D, D, hidden_layers,
                                                                                          device=device, dtype=dtype))
            self.name = "Augmented"

        self.out = nn.Linear(D, 2, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = nn.functional.leaky_relu(self.inv_model(x))
        y = self.out(y.mean(dim=1))
        return nn.functional.softmax(y, dim=-1)
