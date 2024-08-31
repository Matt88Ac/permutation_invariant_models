import math
import torch
from torch.utils.data import Dataset, DataLoader


class GaussianDataset(Dataset):
    def __init__(self, n_samples: int, dims: tuple, mu: float = 0, sigmas=None,
                 device=torch.device('cpu'), dtype=torch.float64):

        if sigmas is None:
            sigmas = [1, 0.8]
        assert len(sigmas) > 1

        self.dims = (n_samples, *dims)

        for_each = n_samples // len(sigmas)

        self.data = sigmas[0] * torch.randn(for_each, *dims, device=device, dtype=dtype) + mu
        self.labels = torch.zeros(n_samples, 1, device=device)

        for i in range(1, len(sigmas) - 1):
            self.labels[for_each * i:] += 1

            self.data = torch.cat([
                self.data,
                sigmas[i] * torch.randn(for_each, *dims, device=device, dtype=dtype) + mu
            ])

        self.labels[len(self.data):] += 1

        self.data = torch.cat([
            self.data,
            math.sqrt(sigmas[-1]) * torch.randn(n_samples - len(self.data), *dims, device=device, dtype=dtype) + mu
        ])

        perm = torch.randperm(n_samples, device=device)
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return self.dims[0]

    def __getitem__(self, idx):
        return self.data[idx].clone(), self.labels[idx].clone()


def get_gaussian_dataloader(batch_size: int, n_samples: int, dims: tuple, mu: float = 0, sigmas=None,
                            device=torch.device('cpu'), dtype=torch.float64, **kwargs) -> DataLoader:
    ds = GaussianDataset(n_samples, dims, mu, sigmas, device, dtype)
    return DataLoader(ds, batch_size=batch_size, **kwargs)
