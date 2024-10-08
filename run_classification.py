from time import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from distribution_classification_experiment.classification_model import DistributionClassifier
from distribution_classification_experiment.data_generator import get_gaussian_dataloader


@torch.no_grad()
def accuracy(y_pred, y_true) -> np.ndarray:
    yp: np.ndarray = (y_pred.cpu().detach().numpy().squeeze() > 0.5).astype(int)
    yt: np.ndarray = y_true.cpu().detach().numpy().squeeze()
    return np.mean(yt == yp)


def training_loop(model: DistributionClassifier, train_data, val_data,
                  n_samples: int, n_iter: int, lr: float = 1e-3, log_path: str = '',
                  **kwargs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

    kwargs['device'] = model.device
    kwargs['dtype'] = model.dtype

    log = np.zeros((n_iter, 5))

    criterion = nn.BCELoss()
    with tqdm(range(n_iter), unit=' Epoch') as tepoch:
        for epoch in tepoch:
            log[epoch, -1] = time()
            ovr_loss = 0
            ovr_acc = 0
            i = 0
            for x, y in train_data:
                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat[:, 1:], y.to(dtype=y_hat.dtype))
                loss.backward()
                optimizer.step()
                ovr_loss += loss.cpu().detach().numpy().item()
                ovr_acc += accuracy(y_hat[:, 1:], y)
                i += 1

            train_loss = ovr_loss / i
            log[epoch, 0] = train_loss
            log[epoch, 1] = ovr_acc / i
            model.eval()
            with torch.no_grad():
                ovr_loss = 0
                ovr_acc = 0
                for x, y in val_data:
                    y_hat = model(x)
                    loss = criterion(y_hat[:, 1:], y.to(dtype=y_hat.dtype))
                    ovr_loss += loss.cpu().detach().numpy().item()
                    ovr_acc += accuracy(y_hat[:, 1:], y)
                val_loss = ovr_loss
            log[epoch, 2] = val_loss
            log[epoch, 3] = ovr_acc
            model.train()

            log[epoch, -1] = time() - log[epoch, -1]
            tepoch.set_postfix(train_loss=train_loss, val_loss=val_loss, val_acc=log[epoch, 3], train_acc=log[epoch, 1])

            if log[:, -1].sum() >= 60 * 30:
                log_path += 'OOT_'
                break

    log = pd.DataFrame(log, columns=['train_loss', 'train_acc', 'val_loss', 'val_acc', 'epoch_time'])
    log.to_csv(log_path + str(n_samples) + '_' + '_training_results.csv', index=False)
    return model


if __name__ == '__main__':
    D = 5
    NS = [2, 4]
    hidden = [32, 64, 32, 5]
    LR = 1e-3
    BATCH_SIZE = 256
    activations = ['relu'] * (len(hidden))
    models = ['canonization_mlp', 'symmetrization_mlp', 'sample_symmetrization_mlp', 'canonization_transformer',
              'symmetrization_transformer', 'sample_symmetrization_transformer', 'equivariant', 'augmentation']

    for n_s in [100, 1000, 10000]:
        N_ITER = 5000 if n_s < 10_000 else 1000
        for n in NS:
            train = get_gaussian_dataloader(BATCH_SIZE, n_s, (n, D), device='cuda:0', dtype=torch.float64, shuffle=True, num_workers=0)
            val = get_gaussian_dataloader(n_s, n_s, (n, D), device='cuda:0', dtype=torch.float64)

            for kind in models:
                class_model = DistributionClassifier(kind, (n, D), device='cuda', hidden_layers=hidden,
                                                     activations=activations)

                training_loop(class_model, train, val, n_s, n_iter=N_ITER,
                              lr=LR, log_path=f'classification_results/{kind}_{n}_')
