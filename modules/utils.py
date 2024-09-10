import os
import numpy as np
import torch
import torch.nn as nn
import copy
import random


def train_softmax_save(opt, epoch, total_train_sftmax_probs):
    datadir = opt.data_dir.replace('/', '.')
    pdf_probs_out_path = "{}{}/train_softmax_{}{}/in_epoch_{}.pth".format(opt.save_ckpt_dir, opt.data_fold, datadir,
                                                                          opt.data_fold, epoch)
    np.save(pdf_probs_out_path, total_train_sftmax_probs.detach().cpu().numpy())


def checkpoint_mae(opt, model, total_surv_probs, total_sftmax_probs, is_observed_list, duration_list):
    datadir = opt.data_dir.replace('/', '.')
    model_out_path = "{}{}/underMae_best_model_trainset_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold, datadir,
                                                                         opt.data_fold)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    # save npy array
    surv_probs_out_path = "{}{}/underMae_best_surv_probs_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                               datadir, opt.data_fold)
    np.save(surv_probs_out_path, total_surv_probs.cpu().numpy())

    # save npy array
    pdf_probs_out_path = "{}{}/underMae_best_pdf_probs_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold, datadir,
                                                                             opt.data_fold)
    np.save(pdf_probs_out_path, total_sftmax_probs.cpu().numpy())

    # save npy array
    is_observed_list_path = "{}{}/underMae_is_observed_list_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                               datadir, opt.data_fold)
    np.save(is_observed_list_path, is_observed_list.cpu().numpy())

    # save npy array
    duration_list_path = "{}{}/underMae_duration_list_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                               datadir, opt.data_fold)
    np.save(duration_list_path, duration_list.cpu().numpy())


def checkpoint_cidx(opt, model, total_surv_probs, total_sftmax_probs, is_observed_list, duration_list):
    datadir = opt.data_dir.replace('/', '.')
    model_out_path = "{}{}/underCidx_best_model_trainset_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold, datadir,
                                                                          opt.data_fold)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    # save npy array
    surv_probs_out_path = "{}{}/underCidx_best_surv_probs_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                                datadir, opt.data_fold)
    np.save(surv_probs_out_path, total_surv_probs.cpu().numpy())

    # save npy array
    surv_probs_out_path = "{}{}/underCidx_best_pdf_probs_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                               datadir, opt.data_fold)
    np.save(surv_probs_out_path, total_sftmax_probs.cpu().numpy())

    # save npy array
    is_observed_list_path = "{}{}/underCidx_is_observed_list_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                               datadir, opt.data_fold)
    np.save(is_observed_list_path, is_observed_list.cpu().numpy())

    # save npy array
    duration_list_path = "{}{}/underCidx_duration_list_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                               datadir, opt.data_fold)
    np.save(duration_list_path, duration_list.cpu().numpy())


def checkpoint_final(opt, model, total_surv_probs, total_sftmax_probs, is_observed_list, duration_list):
    datadir = opt.data_dir.replace('/', '.')
    model_out_path = "{}{}/final_epoch_model_trainset_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold, datadir,
                                                                       opt.data_fold)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

    # save npy array
    surv_probs_out_path = "{}{}/final_epoch_surv_probs_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                             datadir, opt.data_fold)
    np.save(surv_probs_out_path, total_surv_probs.cpu().numpy())

    # save npy array
    surv_probs_out_path = "{}{}/final_epoch_pdf_probs_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold, datadir,
                                                                            opt.data_fold)
    np.save(surv_probs_out_path, total_sftmax_probs.cpu().numpy())

    # save npy array
    is_observed_list_path = "{}{}/final_epoch_is_observed_list_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                                   datadir, opt.data_fold)
    np.save(is_observed_list_path, is_observed_list.cpu().numpy())

    # save npy array
    duration_list_path = "{}{}/final_epoch_duration_list_test_{}{}.pth".format(opt.save_ckpt_dir, opt.data_fold,
                                                                             datadir, opt.data_fold)
    np.save(duration_list_path, duration_list.cpu().numpy())


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
