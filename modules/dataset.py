import torch
import numpy as np
import pandas as pd
import skimage
from operator import itemgetter
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from modules.mae_margin import mae_margin_info


class TranDataset(Dataset):
    def __init__(self, opt, features, image_ids, labels, is_train=True):
        self.is_train = is_train
        self.data = []
        # self.transform = transforms.Compose([
        #                     transforms.ToTensor(),
        #                     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #                      ])

        # calculate mae_margin for train/val/test separately
        bgs_margin, bgs_alpha, _ = mae_margin_info(labels[:, 0], labels[:, 1], how=opt.maemargin_how)

        # assign associated features, images and labels
        temp = []
        for feature, image_id, label, bg_margin, bg_alpha in zip(features, image_ids, labels, bgs_margin, bgs_alpha):
            feature = torch.from_numpy(feature.astype(float))
            # duration is the true observed survival/censoring time. Meanwhile, true observed survival time for uncensoring and
            # best_guess for censoring are in bgs_margin, and uncensoring in bgs_alpha is 1.
            duration, is_observed = label[0], label[1]

            # read the image
            img = skimage.io.imread(opt.image_dir+image_id+'.png')
            img_swapped = np.transpose(img, (2, 0, 1))          # H*W*C --> C*H*W
            img_parts = np.array(np.array_split(img_swapped, opt.max_time // opt.time_period, axis=1))  # split image based on T_period, in numpy
            img_tensor = torch.from_numpy(img_parts)      # change numpy to tensor

            temp.append([duration, img_tensor, is_observed, feature, bg_margin, bg_alpha])
        sorted_temp = sorted(temp, key=itemgetter(0))

        if self.is_train:
            new_temp = sorted_temp
        else:
            new_temp = temp

        for duration, img_tensor, is_observed, feature, bg_margin, bg_alpha in new_temp:
            if is_observed:
                mask = opt.max_time * [1.]
                # label = duration * [1.] + (opt.max_time - duration) * [0.]
                label = duration
                # feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [feature.cuda(opt.gpu), img_tensor.cuda(opt.gpu), torch.tensor(duration).float().cuda(opt.gpu),
                     torch.tensor(mask).float().cuda(opt.gpu), torch.tensor(label).cuda(opt.gpu),
                     torch.tensor(is_observed).byte().cuda(opt.gpu), torch.tensor(bg_margin).float().cuda(opt.gpu),
                     torch.tensor(bg_alpha).float().cuda(opt.gpu)])
            else:
                # NOTE plus 1 to include day 0
                mask = (duration + 1) * [1.] + (opt.max_time - (duration + 1)) * [0.]
                # label = opt.max_time * [1.]
                label = duration
                # feature = torch.stack(opt.max_time * [feature])
                self.data.append(
                    [feature.cuda(opt.gpu), img_tensor.cuda(opt.gpu), torch.tensor(duration).float().cuda(opt.gpu),
                     torch.tensor(mask).float().cuda(opt.gpu), torch.tensor(label).cuda(opt.gpu),
                     torch.tensor(is_observed).byte().cuda(opt.gpu), torch.tensor(bg_margin).float().cuda(opt.gpu),
                     torch.tensor(bg_alpha).float().cuda(opt.gpu)])

    def __getitem__(self, index_a):
        if self.is_train:
            if index_a == len(self.data) - 1:
                index_b = np.random.randint(len(self.data))
            else:
                # NOTE self.data is sorted
                index_b = np.random.randint(index_a + 1, len(self.data))
            return [[self.data[index_a][i], self.data[index_b][i]] for i in range(len(self.data[index_a]))]
        else:
            return self.data[index_a]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    pass
