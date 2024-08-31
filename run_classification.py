from time import time
import torch
from torch import nn
from distribution_classification_experiment.data_generator import get_gaussian_dataloader, GaussianDataset
from distribution_classification_experiment.classification_model import DistributionClassifier
from tqdm import tqdm

import numpy as np
import pandas as pd


@torch.no_grad()
def accuracy(y_pred, y_true) -> np.ndarray:
    yp: np.ndarray = (y_pred.cpu().detach().numpy().squeeze() > 0.5).astype(int)
    yt: np.ndarray = y_true.cpu().detach().numpy().squeeze()
    return np.mean(yt == yp)


def training_loop(model: DistributionClassifier, train_data, val_data,
                  n_samples: int, n_iter: int, lr: float = 1e-3, log_path: str = '',
                  **kwargs):
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)

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
                loss = criterion(y_hat[:, :1], y.to(dtype=y_hat.dtype))
                loss.backward()
                optimizer.step()
                ovr_loss += loss.cpu().detach().numpy().item()
                ovr_acc += accuracy(y_hat[:, :1], y)
                i += 1

            train_loss = ovr_loss / i
            log[epoch, 0] = train_loss
            log[epoch, 1] = ovr_acc / i
            model.eval()
            with torch.no_grad():
                ovr_loss = 0
                ovr_acc = 0
                i = 0
                for x, y in val_data:
                    y_hat = model(x)
                    loss = criterion(y_hat[:, :1], y.to(dtype=y_hat.dtype))
                    ovr_loss += loss.cpu().detach().numpy().item()
                    ovr_acc += accuracy(y_hat[:, :1], y)
                    i += 1
                val_loss = ovr_loss / i
            log[epoch, 2] = val_loss
            log[epoch, 3] = ovr_acc / i
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
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    from sklearn.metrics import accuracy_score, RocCurveDisplay, roc_curve, roc_auc_score
    from matplotlib import pyplot as plt

    plt.style.use('ggplot')
    D = 5
    NS = [2, 4]
    hidden = [16, 256, 16, 5]
    LR = 1e-3
    N_ITER = 5000
    b_size = 256

    activations = ['relu'] * (len(hidden))
    models = ['canonization_mlp', 'symmetrization_mlp', 'sample_symmetrization_mlp', 'canonization_transformer',
              'symmetrization_transformer', 'sample_symmetrization_transformer', 'equivariant', 'augmentation']

    for n_s in [100, 1000, 10000]:

        for n in NS:
            train = get_gaussian_dataloader(b_size, n_s, (n, D), device='cuda:0', dtype=torch.float64, shuffle=True, num_workers=0)
            val = get_gaussian_dataloader(b_size, min(n_s, 500), (n, D), device='cuda:0', dtype=torch.float64)

            plt.figure(figsize=(20, 6), dpi=140)

            for kind in models:
                class_model = DistributionClassifier(kind, (n, D), device='cuda', hidden_layers=hidden,
                                                     activations=activations)

                training_loop(class_model, train, val, n_s, n_iter=N_ITER,
                              lr=LR, log_path=f'classification_results/{kind}_{n}_')

                test = GaussianDataset(10000, (n, D), device='cuda')
                class_model.eval()
                with torch.no_grad():
                    pred = class_model(test.data)[:, :1].cpu().detach().numpy().squeeze()
                    labels = test.labels.cpu().detach().numpy().squeeze()

                fpr, tpr, _ = roc_curve(labels, pred)
                score = roc_auc_score(labels, pred)
                acc = accuracy_score(labels, (pred >= 0.5).astype(int))
                print('\n', kind, acc, '\n')
                roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=f'{kind}: acc = {round(acc, 2)}',
                                              roc_auc=score).plot(ax=plt.gca())
            plt.legend(facecolor='gold', shadow=True, fancybox=True, edgecolor='k')
            plt.savefig(f"classification_results/{n_s}_{n}.png", bbox_inches='tight')
