import os
import sys
import torch
import copy

import matplotlib.pyplot as plt
import seaborn
import wandb
from lifelines import *
import pandas as pd
import argparse
from sklearn.model_selection import ParameterGrid
from sklearn.utils import resample

from modules.train import train
from modules.Encoder import UniSurv
from modules.utils import setup_seed
from modules.EncoderLayer import EncoderLayer
from modules.MultiHeadedAttention import MultiHeadedAttention
from modules.PositionwiseFeedForward import PositionwiseFeedForward

seaborn.set_context(context="talk")
sys.path.append("..")


def main(features, image_ids, labels, num_features):
    c = copy.deepcopy
    attn = MultiHeadedAttention(opt.num_heads, opt.d_model, opt.drop_prob)
    ff = PositionwiseFeedForward(opt.d_model, opt.d_ff, opt.drop_prob)
    encoder_layer = EncoderLayer(opt.d_model, c(attn), c(ff), opt.drop_prob)
    encoder = UniSurv(opt.max_time, opt.time_period, opt.in_channel, encoder_layer, opt.N, opt.d_model, opt.drop_prob, num_features, 5*(opt.time_period-2)).cuda(opt.gpu)
    if opt.data_parallel:
        encoder = torch.nn.DataParallel(encoder).cuda(opt.gpu)
    train(opt, features, image_ids, labels, encoder)


def parse_args_gs(param, case):
    parser = argparse.ArgumentParser(description='Survival analysis')
    parser.add_argument('--max_time', type=int, default=96, help='max number of months')
    parser.add_argument('--time_period', type=int, default=param['time_period'], help='time period in months')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--in_channel', type=int, default=3, help='number of input channels of CNN')
    parser.add_argument('--N', type=int, default=4, help='number of modules')
    parser.add_argument('--num_heads', type=int,
                        default=8)  # number of heads, to MultiHeadedAttention   {1,2,4,8}
    parser.add_argument('--d_model', type=int, default=512)  # d_model, to MultiHeadedAttention / PositionwiseFeedForward / EncoderLayer / Encoder  {256,512}
    parser.add_argument('--d_ff', type=int, default=2048)  # hidden layer of PositionwiseFeedForward
    parser.add_argument('--train_batch_size', type=int, default=16)  # {4,8,16,32}
    parser.add_argument('--drop_prob', type=float, default=0.1)  # {0.0, 0.1, 0.3, 0.5}
    parser.add_argument('--lr', type=float, default=1e-4)  # {1e-4, 5e-4, 1e-3}
    parser.add_argument('--lr_decay_epoch', type=int, default=200)
    parser.add_argument('--lr_decay_factor', type=float, default=1)

    parser.add_argument('--alpha_m', type=float, default=param['alpha_m'])  # mean loss
    parser.add_argument('--alpha_v', type=float, default=param['alpha_v'])  # var loss
    parser.add_argument('--alpha_s', type=float, default=param['alpha_s'])  # softmax loss
    parser.add_argument('--alpha_d', type=float, default=param['alpha_d'])  # discordent loss

    parser.add_argument('--maemargin_how', type=str, default='toTmax')
    parser.add_argument('--unbalance_solve', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=3)

    parser.add_argument('--data_dir', type=str, default='data/cro_val_')
    parser.add_argument('--image_dir', type=str, default='data/img/')
    parser.add_argument('--data_fold', type=str, default=param['fold'])
    parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints/cro_val_')
    parser.add_argument('--save_ckpt_dir_case', type=str, default=str(case))
    parser.add_argument('--report_interval', type=int, default=1)
    parser.add_argument('--data_parallel', action='store_true', help='use data parallel?')
    parser.add_argument('--pred_method', type=str, choices=['mean', 'median'], default='mean')
    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':
    setup_seed(seed=42)

    param_grid = {
        'alpha_m': [0.01],
        'alpha_v': [1],
        'alpha_s': [1],
        'alpha_d': [0],

        'fold': ['0', '1', '2', '3', '4'],
        'time_period': [8]
    }
    params = ParameterGrid(param_grid)
    for case, param in enumerate(params):
        opt = parse_args_gs(param, case)
        os.mkdir('{}{}/{}'.format(opt.save_ckpt_dir, opt.data_fold, opt.save_ckpt_dir_case))
        os.mkdir('{}{}/{}/train_softmax_{}{}'.format(opt.save_ckpt_dir, opt.data_fold, opt.save_ckpt_dir_case,
                                                     opt.data_dir.replace('/', '.'), opt.data_fold))

        # change this to your own wandb account
        # this_run = wandb.init(config=opt, project='UniSurv-gs',
        #                       name='upsampled_loss_fold' + opt.data_fold + '_case' + str(case))

        if opt.unbalance_solve:
            train_all = pd.concat([pd.read_csv(
                opt.data_dir + opt.data_fold + '/train_features_' + opt.data_fold + '.csv', header=0, index_col=False),
                                   pd.read_csv(opt.data_dir + opt.data_fold + '/train_labels_' + opt.data_fold + '.csv',
                                               header=0, index_col=False)], axis=1)
            train_majority, train_minority = train_all[train_all['event_accu'] == 0], train_all[
                train_all['event_accu'] == 1]
            train_minority_upsampled = resample(train_minority, replace=True, n_samples=len(train_majority),
                                                random_state=42)
            print('Using upsampled training dataset!')
            print('After minority upsampled, the numbers fo 0 and 1 are', len(train_majority), len(train_minority_upsampled), 'from', len(train_minority))
            train_upsampled = pd.concat([train_majority, train_minority_upsampled])
            train_upsampled = train_upsampled.sample(frac=1).reset_index(drop=True)
            train_data = train_upsampled.iloc[:, 0:-2].to_numpy()
            train_labels = train_upsampled.iloc[:, -2:].to_numpy()
        else:
            print('Using original training dataset, not upsampled!')
            train_data = pd.read_csv(opt.data_dir + opt.data_fold + '/train_features_' + opt.data_fold + '.csv',
                                     header=0, index_col=False).to_numpy()
            train_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/train_labels_' + opt.data_fold + '.csv',
                                       header=0, index_col=False).to_numpy()

        val_data = pd.read_csv(opt.data_dir + opt.data_fold + '/val_features_' + opt.data_fold + '.csv', header=0,
                               index_col=False).to_numpy()
        test_data = pd.read_csv(opt.data_dir + opt.data_fold + '/test_features_' + opt.data_fold + '.csv', header=0,
                                index_col=False).to_numpy()
        train_features, val_features, test_features = train_data[:, 0:-1], val_data[:, 0:-1], test_data[:, 0:-1]
        train_img, val_img, test_img = train_data[:, -1], val_data[:, -1], test_data[:, -1]
        features = [train_features, val_features, test_features]
        image_ids = [train_img, val_img, test_img]

        val_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/val_labels_' + opt.data_fold + '.csv', header=0,
                                 index_col=False).to_numpy()
        test_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/test_labels_' + opt.data_fold + '.csv', header=0,
                                  index_col=False).to_numpy()
        labels = [train_labels, val_labels, test_labels]

        num_features = train_features.shape[1]
        print('train features shape', train_features.shape)
        print('train labels shape', train_labels.shape)
        print('val features shape', val_features.shape)
        print('val labels shape', val_labels.shape)
        print('test features shape', test_features.shape)
        print('test labels shape', test_labels.shape)
        print()

        print('num features', num_features)
        total_data = train_features.shape[0] + val_features.shape[0] + test_features.shape[0]
        print('total data', total_data)
        print()

        print('train max label', train_labels[:, 0].max())
        print('train min label', train_labels[:, 0].min())
        print('val max label', val_labels[:, 0].max())
        print('val min label', val_labels[:, 0].min())
        print('test max label', test_labels[:, 0].max())
        print('test min label', test_labels[:, 0].min())

        print('coeffs', opt.alpha_m, opt.alpha_v, opt.alpha_s, opt.alpha_d)
        print('time period', opt.time_period)
        main(features, image_ids, labels, num_features)
        print('coeffs', opt.alpha_m, opt.alpha_v, opt.alpha_s, opt.alpha_d)
        print('time period', opt.time_period)
        # this_run.finish()
