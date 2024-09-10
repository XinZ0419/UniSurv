import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from modules.loss import TotalLoss
from modules.dataset import TranDataset
from modules.evaluate import evaluate
from modules.utils import checkpoint_mae, checkpoint_cidx, checkpoint_final, train_softmax_save
# from modules.utils_gs import checkpoint_mae, checkpoint_cidx, checkpoint_final, train_softmax_save


def train(opt, features, image_ids, labels, encoder):
    train_loader = DataLoader(TranDataset(opt, features[0], image_ids[0], labels[0], is_train=True), batch_size=opt.train_batch_size,
                              shuffle=True)
    # NOTE VAL batch size is 1
    val_loader = DataLoader(TranDataset(opt, features[1], image_ids[1], labels[1], is_train=False), batch_size=1, shuffle=False)
    # NOTE TEST batch size is 1
    test_loader = DataLoader(TranDataset(opt, features[2], image_ids[2], labels[2], is_train=False), batch_size=1, shuffle=False)
    # NOTE using mae / C-index as early stopping criterion
    best_val_cindex_1, best_val_mae_1, best_val_mae_hinge_1, best_val_mae_margin_1, best_test_cindex_1, best_test_mae_1, best_test_mae_hinge_1, best_test_mae_margin_1, best_epoch_1 = -1, 1e10, 1e10, 1e10, -1, 1e10, 1e10, 1e10, 0
    best_val_cindex_2, best_val_mae_2, best_val_mae_hinge_2, best_val_mae_margin_2, best_test_cindex_2, best_test_mae_2, best_test_mae_hinge_2, best_test_mae_margin_2, best_epoch_2 = -1, 1e10, 1e10, 1e10, -1, 1e10, 1e10, 1e10, 0

    optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
    scheduler = StepLR(optimizer, step_size=opt.lr_decay_epoch, gamma=opt.lr_decay_factor)

    for t in range(opt.num_epochs):
        print('epoch', t)
        encoder.train()

        tot_loss = 0.
        tot_CE_loss, tot_Disc_loss, tot_CS_loss, tot_mean_loss, tot_var_loss = 0, 0, 0, 0, 0
        total_train_sftmax_probs = []
        for batch_step, (features, images, true_durations, mask, label, is_observed, bg_margin, bg_alpha) in enumerate(train_loader):
            optimizer.zero_grad()
            is_observed_a = is_observed[0]
            bg_margin_a = bg_margin[0]
            bg_alpha_a = bg_alpha[0]
            mask_a = mask[0]
            mask_b = mask[1]
            label_a = label[0]
            label_b = label[1]
            # true_durations_a = true_durations[0]
            # true_durations_b = true_durations[1]
            output_a = encoder.forward(features[0], images[0])
            output_b = encoder.forward(features[1], images[1])
            softmax_a = F.softmax(output_a.squeeze(-1), dim=-1)

            total_train_sftmax_probs.append(softmax_a)

            loss, meanvar, mean, var, disc, ce = TotalLoss(opt).forward(is_observed_a, bg_margin_a, bg_alpha_a, output_a, label_a, output_b, label_b)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            tot_mean_loss += mean.item()
            tot_var_loss += var.item()
            tot_CE_loss += ce.item()
            tot_Disc_loss += disc.item()

        total_train_sftmax_probs = torch.cat(total_train_sftmax_probs)
        train_softmax_save(opt, t, total_train_sftmax_probs)

        print('train total loss', tot_loss/(batch_step+1))
        wandb.log({'training_loss': tot_loss/(batch_step+1), 'epoch': t})
        wandb.log({'training_mean_loss': tot_mean_loss/(batch_step+1), 'training_var_loss': tot_var_loss/(batch_step+1),
                    'training_CE_loss': tot_CE_loss/(batch_step+1),
                   'training_Disc_loss': tot_Disc_loss/(batch_step+1), 'epoch': t})

        # evaluate
        if t > 0 and t % opt.report_interval == 0:
            print('VAL')
            val_cindex, val_mae, val_mae_hinge, val_mae_margin, val_total_surv_probs, _, _, _, _ = evaluate(opt, encoder, val_loader)
            print('TEST')
            test_cindex, test_mae, test_mae_hinge, test_mae_margin, test_total_surv_probs, test_total_sftmax_probs, is_observed_list, duration_list, bg_margin_list = evaluate(opt, encoder, test_loader)
            print('current val cindex', val_cindex, 'val mae', val_mae)
            # using mae as early stopping criterion
            if val_mae < best_val_mae_1:
                best_val_cindex_1 = val_cindex
                best_val_mae_1, best_val_mae_hinge_1, best_val_mae_margin_1 = val_mae, val_mae_hinge, val_mae_margin
                best_test_cindex_1 = test_cindex
                best_test_mae_1, best_test_mae_hinge_1, best_test_mae_margin_1 = test_mae, test_mae_hinge, test_mae_margin
                best_epoch_1 = t
                checkpoint_mae(opt, encoder, test_total_surv_probs, test_total_sftmax_probs, is_observed_list, duration_list, bg_margin_list)
                print('using mae as early stopping criterion')
                print('BEST val cindex', best_val_cindex_1, 'mae', best_val_mae_1, 'mae_hinge', best_val_mae_hinge_1, 'mae_margin', best_val_mae_margin_1, 'at epoch', best_epoch_1)
                print('its test cindex', best_test_cindex_1, 'mae', best_test_mae_1, 'mae_hinge', best_test_mae_hinge_1, 'mae_margin', best_test_mae_margin_1)
            # using cindex as early stopping criterion
            if val_cindex > best_val_cindex_2:
                best_val_cindex_2 = val_cindex
                best_val_mae_2, best_val_mae_hinge_2, best_val_mae_margin_2 = val_mae, val_mae_hinge, val_mae_margin
                best_test_cindex_2 = test_cindex
                best_test_mae_2, best_test_mae_hinge_2, best_test_mae_margin_2 = test_mae, test_mae_hinge, test_mae_margin
                best_epoch_2 = t
                checkpoint_cidx(opt, encoder, test_total_surv_probs, test_total_sftmax_probs, is_observed_list, duration_list, bg_margin_list)
                print('using c-index as early stopping criterion')
                print('BEST val cindex', best_val_cindex_2, 'mae', best_val_mae_2, 'mae_hinge', best_val_mae_hinge_2, 'mae_margin', best_val_mae_margin_2, 'at epoch', best_epoch_2)
                print('its test cindex', best_test_cindex_2, 'mae', best_test_mae_2, 'mae_hinge', best_test_mae_hinge_2, 'mae_margin', best_test_mae_margin_2)

            if t == opt.num_epochs-1:
                checkpoint_final(opt, encoder, test_total_surv_probs, test_total_sftmax_probs, is_observed_list, duration_list, bg_margin_list)
                print('using mae as early stopping criterion')
                print('BEST val cindex', best_val_cindex_1, 'mae', best_val_mae_1, 'mae_hinge', best_val_mae_hinge_1, 'mae_margin', best_val_mae_margin_1, 'at epoch', best_epoch_1)
                print('its test cindex', best_test_cindex_1, 'mae', best_test_mae_1, 'mae_hinge', best_test_mae_hinge_1, 'mae_margin', best_test_mae_margin_1)

                print('using c-index as early stopping criterion')
                print('BEST val cindex', best_val_cindex_2, 'mae', best_val_mae_2, 'mae_hinge', best_val_mae_hinge_2, 'mae_margin', best_val_mae_margin_2, 'at epoch', best_epoch_2)
                print('its test cindex', best_test_cindex_2, 'mae', best_test_mae_2, 'mae_hinge', best_test_mae_hinge_2, 'mae_margin', best_test_mae_margin_2)

            wandb.log({'C-index in val': val_cindex, 'C-index in test': test_cindex,
                       'MAE in val': val_mae, 'MAE-hinge in val': val_mae_hinge, 'MAE-margin in val': val_mae_margin,
                       'MAE in test': test_mae, 'MAE-hinge in test': test_mae_hinge, 'MAE-margin in test': test_mae_margin, 'epoch_eval': t})

        scheduler.step()
