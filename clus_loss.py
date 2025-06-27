import torch
import torch.nn as nn
from classCenter import classCenter
from PartialConv import maskCreate, partialFilter, convParFilt

class ClusterLoss(nn.Module):
    def __init__(self):
        super(ClusterLoss, self).__init__()

    def forward(self, I, u, b, p, sigma):

        Kbsize = 4 * sigma + 1
        B, C, H, W = u.shape
        loss_sum = torch.FloatTensor(1).cuda().zero_()
        for i in range(B):

            I_slice = I[i, :, :, :].unsqueeze(0)
            b_slice = b[i, :, :, :].unsqueeze(0)
            u_slice = u[i, :, :, :].unsqueeze(0)
            D = torch.zeros_like(u_slice) #1*C*H*W
            new_u = torch.zeros_like(u_slice) #1*C*H*W
            u_detach = u_slice.detach()
            up = torch.pow(u_detach, p)
            b2 = torch.pow(b_slice, 2)
            q = 1 / (p - 1)
            mask = maskCreate(I_slice) #1*1*H*W
            Kb = partialFilter(mask, Kbsize) #B*C*H*W*size*size
            b_cov = convParFilt(b_slice, Kb)
            b_cov = b_cov * mask + (1 - mask)  # B*C*H*W

            v = classCenter(I_slice, b_slice, b2, up, C, Kb)
            for j in range(C):
                KbIb =  (I_slice - v[j] * b_cov) * (I_slice - v[j] * b_cov)
                D[:, j, :, :] = torch.pow(KbIb, q) + 1e-9

            f = 1 / D
            f_sum = torch.sum(f, dim = 1, keepdim = True)
            for j in range(C):
                new_u[:, j, :, :] = 1 / (D[:, j, :, :] * f_sum + 1e-9)

            # new_u = mediFilter(new_u, size=3)
            loss = torch.mean((u[i, :, :, :].unsqueeze(0) - new_u) * (u[i, :, :, :].unsqueeze(0) - new_u))
            loss_sum = loss_sum + loss
        return loss_sum / B