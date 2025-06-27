import torch
import torch.nn as nn
from classCenter import classCenter
from PartialConv import maskCreate, partialFilter, convParFilt

class BiasPredictLoss(nn.Module):
    def __init__(self):
        super(BiasPredictLoss, self).__init__()

    def forward(self, I, u, b, p, sigma): #I：B*1*H*W; u:B*C*H*W; b:B*1*H*W

        Kbsize = 4 * sigma + 1
        B, C, H, W = u.shape
        loss_sum = torch.FloatTensor(1).cuda().zero_()
        for i in range(B):

            I_slice = I[i, :, :, :].unsqueeze(0)
            b_slice = b[i, :, :, :].unsqueeze(0)
            u_slice = u[i, :, :, :].unsqueeze(0)
            KIcu = torch.zeros_like(b_slice) #1*1*H*W
            cKbu = torch.zeros_like(b_slice) #1*1*H*W

            b_detach = b_slice.detach()
            up = torch.pow(u_slice, p) #1*C*H*W
            b2 = torch.pow(b_detach, 2) #1*1*H*W
            mask = maskCreate(I_slice) #1*1*H*W
            Kb = partialFilter(mask, Kbsize) #1*1*H*W*size*size

            v = classCenter(I_slice, b_detach, b2, up, C, Kb) #C
            for j in range(C):
                KIcu = KIcu + convParFilt(up[:, j, :, :].unsqueeze(dim=1) * v[j] * I_slice, Kb)
                cKbu = cKbu + v[j] * v[j] * convParFilt(up[:, j, :, :].unsqueeze(dim=1), Kb)

            bd = KIcu
            db = cKbu
            bd = bd * mask + (1 - mask)
            db = db * mask + (1 - mask)
            b_new = (bd + 1e-9) / (db + 1e-9)
            loss = torch.mean((b[i, :, :, :].unsqueeze(0) - b_new) * (b[i, :, :, :].unsqueeze(0) - b_new))
            loss_sum = loss_sum + loss

        return loss_sum / B