from functools import partial
from time import time
import numpy as np
import pandas as pd
import torch
from symmetry_tests.testing_funcs import test_model, MODEL_TYPES
from tqdm import tqdm


def run_and_log(n_tests: int, model_kind: MODEL_TYPES, batch_size: int, max_in_dim: int = 8, max_in_channels: int = 5,
                max_out_channels: int = 5, max_n_hidden: int = 7, max_scale: float = 20, thr: float = 1e-5,
                device=torch.device('cpu'), dtype=torch.float64, log_path: str = '', **other_model_parameters):

    tester = partial(test_model, max_in_dim=max_in_dim, max_in_channels=max_in_channels,
                     max_out_channels=max_out_channels, max_n_hidden=max_n_hidden, max_scale=max_scale,
                     thr=thr, device=device, dtype=dtype, **other_model_parameters)

    results = np.zeros((n_tests, 2))  # invariant/equivariant, time of test
    for i in tqdm(range(n_tests)):
        results[i, 1] = time()
        results[i, 0] = tester(model_kind, batch_size)
        results[i, 1] = time() - results[i, 1]

    results = pd.DataFrame(results, columns=['Invariant_Equivariant', 'Test Time'])
    results.to_csv(log_path + f'{model_kind}_{batch_size}.csv', index=False)
