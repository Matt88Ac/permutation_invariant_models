import torch

from Networks.layers import GeneralLayer


@torch.jit.script
def MeanRE(y_pred: torch.Tensor, y_true: torch.Tensor):
    """ Mean Relative Error """
    error = (y_true - y_pred) / y_true
    return error.abs().mean()


def test_invariance(model: GeneralLayer, x: torch.Tensor, thr: float = 1e-4):
    perm = torch.randperm(x.shape[1], device=x.device)
    y1 = model(x[:, perm].clone())
    y2 = model(x.clone())
    return (MeanRE(y1, y2) < thr).cpu().detach().numpy().item()


def test_equivariance(model: GeneralLayer, x: torch.Tensor, thr: float = 1e-4):
    perm = torch.randperm(x.shape[1], device=x.device)
    y1 = model(x[:, perm].clone())
    y2 = model(x.clone())[:, perm]
    return (MeanRE(y1, y2) < thr).cpu().detach().numpy().item()


def gen_random_data(batch_size: int, shape: tuple, scale: float = 1, device=torch.device('cpu'),
                    dtype=torch.float64) -> torch.Tensor:
    x = torch.randn(batch_size, *shape, device=device, dtype=dtype) * scale
    return x




