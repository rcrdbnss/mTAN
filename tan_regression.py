#pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from random import SystemRandom

from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import ists_utils
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=20)
parser.add_argument('--rec-hidden', type=int, default=256)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--quantization', type=float, default=0.1, help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true', help="Include binary classification loss")
parser.add_argument('--freq', type=float, default=10.)
parser.add_argument('--k-iwae', type=int, default=1)
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--dataset', type=str, default='mimiciii')
parser.add_argument('--alpha', type=int, default=100.)
parser.add_argument('--old-split', type=int, default=1)
parser.add_argument('--nonormalize', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--classify-pertp', action='store_true')
parser.add_argument('--device', type=str, default='')
parser.add_argument('--dev', action='store_true', help='Run on development data')
parser.add_argument('--patience', default=20)
# ists args
parser.add_argument('--subset', default='all')
parser.add_argument('--num-past', type=int, default=None, help='Number of past values to consider')
parser.add_argument('--num-fut', type=int, default=None, help='Number of future values to predict')
parser.add_argument('--nan-pct', type=float, default=None, help='Percentage of NaN values to insert')
parser.add_argument('--abl-code', type=str, default='ES')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dev:
        args.niters = 2
        args.batch_size = 25
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.set_device(device)

    if args.dataset == 'physionet':
        data_obj = utils.get_physionet_data(args, 'cpu', args.quantization)
    elif args.dataset == 'mimiciii':
        data_obj = utils.get_mimiciii_data_reg(args)
    elif args.dataset in ['french', 'ushcn', 'adbpo']:
        dataset = args.dataset
        subset = args.subset
        num_past = args.num_past
        num_fut = args.num_fut
        nan_pct = args.nan_pct
        abl_code = args.abl_code
        if args.dev:
            subset = f'{args.subset}_dev'
        Xy_dict, D = ists_utils.load_adapt_data(None, dataset, subset, nan_pct, num_past, num_fut, abl_code)
        loaders = ists_utils.data_dict_to_loaders_dict(Xy_dict, args.batch_size)
        data_obj = {
            "train_dataloader": loaders["train"],
            "val_dataloader": loaders["valid"],
            "test_dataloader": loaders["test"],
            "input_dim": D["input_dim"]
        }
        conf_name = f"{dataset}_{subset}_nan{int(nan_pct * 10)}_np{num_past}_nf{num_fut}"

    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    val_loader = data_obj["val_dataloader"]
    dim = data_obj["input_dim"]

    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., 128), args.latent_dim, args.rec_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden,
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)

    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., 128), args.latent_dim, args.gen_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden,
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)

    if args.dataset == 'adbpo':
        regressor = models.RegressorADBPO(args.latent_dim, args.rec_hidden, 1).to(device)
    else:
        regressor = models.Regressor(args.latent_dim, args.rec_hidden).to(device)
    params = (list(rec.parameters()) + list(dec.parameters()) + list(regressor.parameters()))
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec), utils.count_parameters(regressor))
    optimizer = optim.Adam(params, lr=args.lr)
    criterion = nn.MSELoss()  # Use MSELoss for regression

    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])

    train_losses, val_losses, test_losses = [], [], []
    best_val_loss = float('inf')
    epoch_times = []
    patience = args.patience
    for itr in range(1, args.niters + 1):
        train_recon_loss, train_mse_loss = 0, 0
        mse_x = 0
        mse_y = 0
        mae_y = 0
        train_r2 = 0
        train_n = 0
        train_mae = 0
        # avg_reconst, avg_kl, mse = 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1
        start_time = time.time()
        for train_batch, label in tqdm(train_loader):
            train_batch, label = train_batch.to(device), label.to(device)
            batch_len = train_batch.shape[0]
            observed_data, observed_mask, observed_tp = train_batch[:, :, :dim], train_batch[:, :, dim:2 * dim], train_batch[:, :, -1]
            out = rec(torch.cat((observed_data, observed_mask), 2), observed_tp)
            qz0_mean, qz0_logvar = out[:, :, :args.latent_dim], out[:, :, args.latent_dim:]
            epsilon = torch.randn(args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_y = regressor(z0)
            pred_x = dec(z0, observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1]))
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])  # nsample, batch, seqlen, dim
            # compute loss
            logpx, analytic_kl = utils.compute_losses(dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            recon_loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            label = label.unsqueeze(0).repeat_interleave(args.k_iwae, 0).view(-1)
            mse_loss = criterion(pred_y, label)
            loss = recon_loss + args.alpha * mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_mse_loss += mse_loss.item() * batch_len
            train_recon_loss += recon_loss.item() * batch_len
            train_mae += nn.L1Loss()(pred_y, label).item() * batch_len
            train_n += batch_len
            mse_x += utils.mean_squared_error(observed_data, pred_x.mean(0), observed_mask) * batch_len

            label, pred_y = label.cpu().detach().numpy(), pred_y.cpu().detach().numpy()
            label, pred_y = label.reshape(-1, 1), pred_y.reshape(-1, 1)
            mse_y += mean_squared_error(label, pred_y) * batch_len
            mae_y += mean_absolute_error(label, pred_y) * batch_len
            train_r2 += r2_score(label, pred_y) * batch_len

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        val_loss, val_acc, val_mse, val_mae, val_r2 = utils.evaluate_regressor(
            rec, val_loader, args=args, regressor=regressor, reconst=True, num_sample=1, dim=dim,
            data_min=(data_obj['data_min'] if 'data_min' in data_obj else None),
            data_max=(data_obj['data_max'] if 'data_max' in data_obj else None))
        val_losses.append(val_loss)
        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'regressor_state_dict': regressor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': -val_loss,
            }, f'{args.dataset}_{args.enc}_{args.dec}_{experiment_id}.h5')
            best_iter = itr
            patience = args.patience
        else:
            patience -= 1
            if patience == 0:
                break
        test_loss, test_acc, test_mse, test_mae, test_r2 = utils.evaluate_regressor(
            rec, test_loader, args=args, regressor=regressor, reconst=True, num_sample=1, dim=dim,
            data_min=(data_obj['data_min'] if 'data_min' in data_obj else None),
            data_max=(data_obj['data_max'] if 'data_max' in data_obj else None))
        test_losses.append(test_loss)
        print(f'Iter: {itr}, recon_loss: {train_recon_loss / train_n:.4f}, '
            f'mse_loss: {train_mse_loss / train_n:.4f}, acc: {train_mae / train_n:.4f}, '
            f'train_mse: {mse_y / train_n:.4f}, train_mae: {mae_y / train_n:.4f}, '
            f'train_r2: {train_r2 / train_n:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, '
            f'val_mse: {val_mse:.4f}, val_mae: {val_mae:.4f}, val_r2: {val_r2:.4f}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, '
            f'test_mse: {test_mse:.4f}, test_mae: {test_mae:.4f}, test_r2: {test_r2:.4f}')

    print(f'Best iteration: {best_iter}, {best_val_loss}')
    checkpoint = torch.load(f'{args.dataset}_{args.enc}_{args.dec}_{experiment_id}.h5')
    rec.load_state_dict(checkpoint['rec_state_dict'])
    dec.load_state_dict(checkpoint['dec_state_dict'])
    regressor.load_state_dict(checkpoint['regressor_state_dict'])
    os.remove(f'{args.dataset}_{args.enc}_{args.dec}_{experiment_id}.h5')
    
    print('Testing...')
    if args.dataset in ['french', 'ushcn', 'adbpo']:
        scalers = D['scalers']
        for id in scalers:
            for f in scalers[id]:
                if isinstance(scalers[id][f], dict):
                    scaler = {
                        "standard": StandardScaler,
                        # "minmax": MinMaxScaler,
                    }["standard"]()
                    for k, v in scalers[id][f].items():
                        setattr(scaler, k, v)
                    scalers[id][f] = scaler
        mse_test, mae_test = ists_utils.evaluate_raw(test_loader, D['id_array_test'], rec, dec, args, regressor, dim, scalers)
        mse_train, mae_train = ists_utils.evaluate_raw(train_loader, D['id_array_train'], rec, dec, args, regressor, dim, scalers)

        print('mse:', mse_test, 'mae:', mae_test)

        results_path = f'results/{conf_name}.csv'
        results = dict()
        if os.path.exists(results_path):
            results = pd.read_csv(results_path, index_col=0).to_dict(orient='index')
        results[f'{abl_code}'] = {
            'test_mae': mae_test, 'test_mse': mse_test,
            'train_mae': mae_train, 'train_mse': mse_train,
            'val_loss': val_losses, 'test_loss': test_losses,
            "epoch_times": epoch_times
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.index.name = conf_name
        results.to_csv(results_path)
    else:
        test_loss, test_acc, test_mse, test_mae, test_r2 = utils.evaluate_regressor(
            rec, test_loader, args=args, regressor=regressor, reconst=True, num_sample=1, dim=dim)
        print(f'test_loss: {test_loss:.4f}, test_mse: {test_mse:.4f}, test_mae: {test_mae:.4f}, test_r2: {test_r2:.4f}')
