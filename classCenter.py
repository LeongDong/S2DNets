import torch
from PartialConv import convParFilt

def classCenter(I, b, b2, up, C, Kb):#, Kn, weight = 1):
    center = torch.FloatTensor(C).cuda().zero_()

    for i in range(C - 1):
        bKIu = convParFilt(I * up[:, i + 1, :, :].unsqueeze(dim=1), Kb) * b
        b2Ku = convParFilt(up[:, i + 1, :, :].unsqueeze(dim=1), Kb) * b2
        bd_sum = torch.sum(bKIu)
        db_sum = torch.sum(b2Ku)
        center[i + 1] = bd_sum / (db_sum + 1e-9)

    bd0 = I * up[:, 0, :, :] * torch.ones_like(b)
    db0 = up[:, 0, :, :] * torch.ones_like(b)
    center[0] = torch.sum(bd0) / (torch.sum(db0) + 1e-9)

    return center