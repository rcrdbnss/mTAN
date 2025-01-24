import numpy as np
import pickle
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

import utils


def get_X_M_T(X, feat_mask):
    def f(X):
        feat_mask_ = np.array(feat_mask)
        x_arg = feat_mask_ == 0
        m_arg = feat_mask_ == 1
        # T_arg = feat_mask_ == 2
        # x, m, T = X[:, :, x_arg], X[:, :, m_arg], X[:, :, T_arg]
        X, M = X[:, :, x_arg], X[:, :, m_arg]
        X = np.transpose(X, (0, 2, 1))
        M = np.transpose(M, (0, 2, 1))
        # T = np.transpose(T, (0, 2, 1))
        T = []
        return X, M, T

    if len(np.shape(X)) == 3:
        return f(X)
    elif len(np.shape(X)) == 4:
        X_list = X
        X, M, T = [], [], []
        for X_ in X_list:
            x, m, t = f(X_)
            X.append(x)
            M.append(m)
            # T.append(T_)
        X = np.concatenate(X, axis=1)
        M = np.concatenate(M, axis=1)
        # T = np.concatenate(T, axis=1)
        return X, M, T
    else:
        return [], [], []


def adapter(X, X_spt, X_exg, feat_mask, E: bool, S: bool):
    X, M, T = get_X_M_T(X, feat_mask)
    X_spt, M_spt, T_spt = get_X_M_T(X_spt, feat_mask)
    X_exg, M_exg, T_exg = get_X_M_T(X_exg, feat_mask)
    # x = np.concatenate([x, x_spt, x_exg], axis=1)
    # m = np.concatenate([m, m_spt, m_exg], axis=1)
    X, M = [X], [M]
    if S:
        X.append(X_spt)
        M.append(M_spt)
    if E:
        X.append(X_exg)
        M.append(M_exg)
    X, M = np.concatenate(X, axis=1), np.concatenate(M, axis=1)
    M = 1 - M  # null indicator -> mask
    drop_null_windows = (M.sum(axis=-1) > 0).all(axis=-1)  # False if at least one variable in the window is entirely null, True otherwise
    X, M = X[drop_null_windows], M[drop_null_windows]
    T = np.zeros_like(X[:, 0:1])
    T = np.apply_along_axis(lambda x: (np.arange(np.shape(x)[-1]) / np.shape(x)[-1]), -1, T)
    return np.transpose(X, (0, 2, 1)), np.transpose(M, (0, 2, 1)), np.transpose(T, (0, 2, 1)), drop_null_windows


def load_adapt_data(base_path, dataset, subset, nan_pct, num_past, num_fut, abl_code) -> (Dict[str, np.ndarray], Dict):
    base_path = "../ists/output/pickle"
    conf_name = f"{dataset}_{subset}_nan{int(nan_pct * 10)}_np{num_past}_nf{num_fut}"
    print("Loading from", f'{base_path}/{conf_name}.pickle', "...")
    with open(f'{base_path}/{conf_name}.pickle', 'rb') as f:
        train_test_dict = pickle.load(f)
    print("Done!")

    D = train_test_dict
    feat_mask = D['x_feat_mask']
    E, S = 'E' in abl_code, 'S' in abl_code
    x_train, m_train, T_train, N_train = adapter(D['x_train'], D['spt_train'], D['exg_train'], feat_mask, E, S)
    x_valid, m_valid, T_valid, N_valid = adapter(D['x_valid'], D['spt_valid'], D['exg_valid'], feat_mask, E, S)
    x_test, m_test, T_test, N_test = adapter(D['x_test'], D['spt_test'], D['exg_test'], feat_mask, E, S)
    y_train, y_valid, y_test = D['y_train'], D['y_valid'], D['y_test']
    y_train, y_valid, y_test = y_train[N_train], y_valid[N_valid], y_test[N_test]
    input_dim = x_train.shape[-1]

    # masked out values are set to 0
    x_train[m_train == 0] = 0
    x_valid[m_valid == 0] = 0
    x_test[m_test == 0] = 0

    x_train = np.concatenate((x_train, m_train, T_train), axis=-1)
    x_valid = np.concatenate((x_valid, m_valid, T_valid), axis=-1)
    x_test = np.concatenate((x_test, m_test, T_test), axis=-1)
    x_train, x_valid, x_test = torch.tensor(x_train).float(), torch.tensor(x_valid).float(), torch.tensor(
        x_test).float()
    y_train, y_valid, y_test = torch.tensor(y_train).float(), torch.tensor(y_valid).float(), torch.tensor(
        y_test).float()

    X_y_dict = {
        "X_train": x_train, "X_valid": x_valid, "X_test": x_test,
        "y_train": y_train, "y_valid": y_valid, "y_test": y_test,
        # "input_dim": input_dim
    }
    params = {
        "input_dim": input_dim,
        "scalers": D['scalers'],
        "id_array_train": D['id_train'],
        "id_array_valid": D['id_valid'],
        "id_array_test": D['id_test']
    }
    return X_y_dict, params


def data_dict_to_loaders_dict(X_y_dict: dict, batch_size: int) -> Dict[str, DataLoader]:
    D = X_y_dict
    train_dataset = TensorDataset(D["X_train"], D["y_train"].float())
    valid_dataset = TensorDataset(D["X_valid"], D["y_valid"].float())
    test_dataset = TensorDataset(D["X_test"], D["y_test"].float())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }


def evaluate_raw(loader, id_array, rec, dec, args, model, dim, scalers):
    y_pred, y_true = utils.predict_regr(rec, loader, dec, args, model, dim, reconst=True, num_sample=1)
    y_pred, y_true = np.reshape(y_pred, (-1, 1)), np.reshape(y_true, (-1, 1))
    y_true = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_true, id_array)])
    y_pred = np.array([np.reshape([scalers[id][f].inverse_transform([[y__]]) for y__, f in zip(y_, scalers[id])], -1)
                       for y_, id in zip(y_pred, id_array)])
    mse, mae = metrics.mean_squared_error(y_true, y_pred), metrics.mean_absolute_error(y_true, y_pred)
    return mse, mae
