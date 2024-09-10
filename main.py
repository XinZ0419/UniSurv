import sys
import torch
import copy
import matplotlib.pyplot as plt
import seaborn
import wandb
from lifelines import *
import pandas as pd
import argparse
from sklearn.utils import resample

from modules.train import train
from modules.Encoder import UniSurv
from modules.utils import setup_seed
from modules.options import parse_args
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


if __name__ == '__main__':
    opt = parse_args()
    setup_seed(seed=42)

    # change this to your own wandb account
    # wandb.init(config=opt, project='UniSurv')

    if opt.unbalance_solve:
        train_all = pd.concat([pd.read_csv(opt.data_dir + opt.data_fold + '/train_features_' + opt.data_fold + '.csv', header=0, index_col=False),
                               pd.read_csv(opt.data_dir + opt.data_fold + '/train_labels_' + opt.data_fold + '.csv', header=0, index_col=False)], axis=1)
        train_majority, train_minority = train_all[train_all['event_accu'] == 0], train_all[train_all['event_accu'] == 1]
        train_minority_upsampled = resample(train_minority, replace=True, n_samples=len(train_majority), random_state=42)
        train_upsampled = pd.concat([train_majority, train_minority_upsampled])
        train_upsampled = train_upsampled.sample(frac=1).reset_index(drop=True)
        train_data = train_upsampled.iloc[:, 0:-2].to_numpy()
        train_labels = train_upsampled.iloc[:, -2:].to_numpy()
    else:
        train_data = pd.read_csv(opt.data_dir + opt.data_fold + '/train_features_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
        train_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/train_labels_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()

    val_data = pd.read_csv(opt.data_dir + opt.data_fold + '/val_features_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    test_data = pd.read_csv(opt.data_dir + opt.data_fold + '/test_features_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    train_features, val_features, test_features = train_data[:, 0:-1], val_data[:, 0:-1], test_data[:, 0:-1]
    train_img, val_img, test_img = train_data[:, -1], val_data[:, -1], test_data[:, -1]
    features = [train_features, val_features, test_features]
    image_ids = [train_img, val_img, test_img]

    val_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/val_labels_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
    test_labels = pd.read_csv(opt.data_dir + opt.data_fold + '/test_labels_' + opt.data_fold + '.csv', header=0, index_col=False).to_numpy()
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

    print('coeffs of losses', opt.alpha_m, opt.alpha_v, opt.alpha_s, opt.alpha_d)

    main(features, image_ids, labels, num_features)

    print('coeffs of losses', opt.alpha_m, opt.alpha_v, opt.alpha_s, opt.alpha_d)


