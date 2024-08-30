import torch
from symmetry_tests.test_and_log import run_and_log


def test_canonization(device=torch.device('cpu')):
    run_and_log(n_tests=1000, model_kind='canonization_mlp', batch_size=10_000, max_in_dim=7,
                max_in_channels=128, max_scale=10, device=device, log_path='test_results/')

    run_and_log(n_tests=1000, model_kind='canonization_transformer', batch_size=10_000, max_in_dim=7,
                max_in_channels=128, max_scale=10, device=device, log_path='test_results/')


def test_symmetrization(device=torch.device('cpu')):
    run_and_log(n_tests=1000, model_kind='symmetrization_mlp', batch_size=10_000, max_in_dim=7,
                max_in_channels=128, max_scale=10, device=device, log_path='test_results/')

    run_and_log(n_tests=1000, model_kind='symmetrization_transformer', batch_size=10_000, max_in_dim=7,
                max_in_channels=128, max_scale=10, device=device, log_path='test_results/')


def test_equivariant(device=torch.device('cpu')):
    run_and_log(n_tests=1000, model_kind='equivariant', batch_size=10_000, max_in_dim=7,
                max_in_channels=128, max_scale=10, device=device, log_path='test_results/')


def test_sample_symmetrization(device=torch.device('cpu')):
    sample_rates = [0.05, 0.1, 0.2, 0.5, 0.75]

    for sr in sample_rates:
        run_and_log(n_tests=1000, model_kind='sample_symmetrization_mlp', batch_size=10_000, max_in_dim=7,
                    max_in_channels=128, max_scale=10, device=device, sample_rate=sr, log_path=f'test_results/sr_{sr}_')
    for sr in sample_rates:
        run_and_log(n_tests=1000, model_kind='sample_symmetrization_transformer', batch_size=10_000, max_in_dim=7,
                    max_in_channels=128, max_scale=10, device=device, sample_rate=sr, log_path=f'test_results/sr_{sr}_')


def test_all():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_equivariant(device)
    test_canonization(device)
    test_symmetrization(device)
    test_sample_symmetrization(device)


if __name__ == '__main__':
    test_all()
