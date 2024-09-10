import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Survival analysis')
    parser.add_argument('--max_time', type=int, default=96, help='max number of months')
    parser.add_argument('--time_period', type=int, default=6, help='time period in months')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--in_channel', type=int, default=3, help='number of input channels of CNN')
    parser.add_argument('--N', type=int, default=4, help='number of modules')
    parser.add_argument('--num_heads', type=int, default=8)  # number of heads, to MultiHeadedAttention   {1,2,4,8}
    parser.add_argument('--d_model', type=int, default=512)  # d_model, to MultiHeadedAttention / PositionwiseFeedForward / EncoderLayer / Encoder  {256,512}
    parser.add_argument('--d_ff', type=int, default=2048)  # hidden layer of PositionwiseFeedForward
    parser.add_argument('--train_batch_size', type=int, default=16)  # {4,8,16,32}
    parser.add_argument('--drop_prob', type=float, default=0.1)  # {0.0, 0.1, 0.3, 0.5}
    parser.add_argument('--lr', type=float, default=1e-4)  # {1e-4, 5e-4, 1e-3}
    parser.add_argument('--lr_decay_epoch', type=int, default=200)
    parser.add_argument('--lr_decay_factor', type=float, default=1)

    parser.add_argument('--alpha_m', type=float, default=0.01)  #mean loss
    parser.add_argument('--alpha_v', type=float, default=1)  # var loss
    parser.add_argument('--alpha_s', type=float, default=1)  # softmax loss
    parser.add_argument('--alpha_d', type=float, default=1)  # discordent loss

    parser.add_argument('--maemargin_how', type=str, default='toTmax')
    parser.add_argument('--unbalance_solve', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=2)

    parser.add_argument('--data_dir', type=str, default='data/cro_val_')
    parser.add_argument('--image_dir', type=str, default='data/img/')
    parser.add_argument('--data_fold', type=str, default='1')
    parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints/cro_val_')
    parser.add_argument('--report_interval', type=int, default=1)
    parser.add_argument('--data_parallel', action='store_true', help='use data parallel?')
    parser.add_argument('--pred_method', type=str, choices=['mean', 'median'], default='mean')
    opt = parser.parse_args()
    print(opt)
    return opt
