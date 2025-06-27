import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, bias, lamwei = 1):
        B, C, H, W = bias.size()
        tv_h = torch.pow((bias[:, :, 1:, :] - bias[:, :, :-1, :]),2).sum()
        tv_w = torch.pow((bias[:, :, :, 1:] - bias[:, :, :, :-1]),2).sum()
        loss_tv = lamwei * (tv_h + tv_w) / (B * C * H * W)

        return loss_tv