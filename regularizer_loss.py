import torch
import torch.nn as nn

def classCenter(I,b,b2,up,C):

    center = torch.FloatTensor(C).cuda().zero_()
    for i in range(C):
        bd = I * b * up[:,i,:,:]
        db = b2 * up[:,i,:,:]
        bd_sum = torch.sum(bd)
        db_sum = torch.sum(db)
        center[i] = bd_sum / (db_sum + 1e-9)

    return center

class RegularizerLoss(nn.Module):

    def __init__(self):
        super(RegularizerLoss,self).__init__()

    def forward(self, I, u, b, p):

        loss = torch.FloatTensor(1).cuda().zero_()
        B, C, H, W = u.shape
        for k in range(B):
            u_slice = u[k, :, :, :].unsqueeze(0)
            up = torch.pow(u_slice,p)
            b2 = torch.pow(b,2)
            v = classCenter(I, b, b2, up, C)
            for i in range(C):
                for j in range((i + 1),C):
                    if(v[j] < v[i]):
                        loss = loss + v[i] - v[j]

        return loss