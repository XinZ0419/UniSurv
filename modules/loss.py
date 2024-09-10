import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TotalLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.num_classes = opt.max_time
        self.alpha_m = opt.alpha_m
        self.alpha_v = opt.alpha_v
        self.alpha_s = opt.alpha_s
        self.alpha_d = opt.alpha_d

    def forward(self, is_observed, bg_margin_a, bg_alpha_a, output, target, output_b, target_b):
        ce = self.CELoss(bg_margin_a, bg_alpha_a, output, self.alpha_s)

        output = F.softmax(output.squeeze(-1), dim=-1)
        output_b = F.softmax(output_b.squeeze(-1), dim=-1)

        meanvar, mean, var = self.MeanVarianceLoss(bg_margin_a, bg_alpha_a, output, self.alpha_m, self.alpha_v)

        disc = self.DiscLoss(is_observed, output, target, output_b, target_b, self.alpha_d)
        TotalLoss = meanvar + disc + ce
        return TotalLoss, meanvar, mean, var, disc, ce

    def MeanVarianceLoss(self, bg_margin_a, bg_alpha_a, output, alpha_m, alpha_v):
        rank = torch.arange(0, self.num_classes, dtype=output.dtype, device=output.device)
        mean = torch.squeeze((output * rank).sum(1, keepdim=True), dim=1)

        mse = ((mean - bg_margin_a) ** 2) * bg_alpha_a

        mean_loss = (alpha_m * mse).mean() / 2.0
        b = (rank[None, :] - mean[:, None]) ** 2
        variance_loss = alpha_v * (output * b).sum(1, keepdim=True).mean()

        MVLoss = mean_loss + variance_loss
        return MVLoss, mean_loss, variance_loss

    def CELoss(self, bg_margin_a, bg_alpha_a, output, alpha_s):
        criterion1 = nn.CrossEntropyLoss(reduction='mean').to(device=bg_margin_a.device)
        CELoss = alpha_s * criterion1(output, bg_margin_a.long().unsqueeze(1))
        return CELoss

    def DiscLoss(self, is_observed, output, target, output_b, target_b, alpha_d):
        cond_a = is_observed & (target < target_b)
        if torch.sum(cond_a) > 0:
            surv_probs_a = 1 - torch.cumsum(output, dim=1)
            mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)

            surv_probs_b = 1 - torch.cumsum(output_b, dim=1)
            mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)

            diff = mean_lifetimes_a[cond_a.bool()] - mean_lifetimes_b[cond_a.bool()]
            true_diff = target_b[cond_a.bool()] - target[cond_a.bool()]
            discLoss = alpha_d * torch.mean(nn.ReLU()(true_diff + diff))
        else:
            discLoss = alpha_d * torch.tensor(0.0, device=output.device, requires_grad=True)
        return discLoss


if __name__ == '__main__':
    x = torch.rand(3, 4)
    print(x)
    indices = torch.tensor([[0, 0], [2, 0]])
    print(torch.index_select(x, 0, indices))

