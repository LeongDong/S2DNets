import torch
import numpy as np
import torch.nn.functional as F

def maskCreate(image): #1*1*H*W

    image = image * 255
    mask = torch.zeros_like(image)
    mask[image > 20] = 1
    # mask = torch.ones_like(image)
    return mask #1*1*H*W

def maxValue(x, thresh = 0.5):

    if(x < thresh):
        x = 0
    else:
        x = 1
    return x

def GaussianKernel(size):

    sigma = (size - 1) / 4#0.3 * ((size - 1) * 0.5 - 1) + 0.8
    center = size // 2
    ele = (np.arange(size, dtype=np.float64) - center)

    kernel1d = np.exp(- (ele ** 2) / (2 * sigma ** 2)) #size * 1
    kernel = kernel1d[..., None] @ kernel1d[None, ...] #size * size
    kernel = torch.from_numpy(kernel)

    return kernel.unsqueeze(0).unsqueeze(0) #1*1*size*size, Unnormalized gaussian kernel

def partialFilter(mask, size): #mask: B*C*H*W

    kernel = GaussianKernel(size)
    kernel = kernel.cuda()
    pad = (size - 1) // 2
    maskPad = F.pad(mask, pad=[pad, pad, pad, pad], mode='reflect') # B*C*(H+2pad)*(W+2pad)
    patches = maskPad.unfold(2, size, 1).unfold(3, size, 1) #B*C*H*W*size*size

    parKernels = patches * kernel.unsqueeze(0).unsqueeze(0) #(B*C*H*W*size*size)*(1*1*1*1*size*size)->B*C*H*W*size*size
    kernelNorm = parKernels / (parKernels.sum(dim=(-1,-2), keepdim=True) + 1e-9) #B*C*H*W*size*size
    kernelNorm = kernelNorm * mask.unsqueeze(-1).unsqueeze(-1) #(B*C*H*W*size*size) * (B*C*H*W*1*1)->B*C*H*W*size*size

    return kernelNorm

def convParFilt(image, kernel): #B*C*H*W

    size = kernel.shape[-1]
    pad = (size - 1) // 2
    imgPad = F.pad(image, pad=[pad, pad, pad, pad], mode='reflect') #B*C*(H+2pad)*(W+2pad)
    imgPatches = imgPad.unfold(2, size, 1).unfold(3, size, 1) #B*C*H*W*size*size
    imgConv = kernel * imgPatches #B*C*H*W*size*size
    return imgConv.sum(dim=(-1, -2), keepdim=False) #B*C*H*W

def mediFilter(image, size):

    N, B, H, W = image.shape
    pad = (size - 1) // 2
    imgPad = F.pad(image, pad = [pad, pad, pad, pad], mode = 'reflect')
    patches = imgPad.unfold(2, size, 1).unfold(3, size, 1) #N*B*H*W*size*size

    patchFlat = patches.reshape(N,B,H,W,size * size) #N*B*H*W*(size*size)
    patchMedia = torch.median(patchFlat, dim=-1, keepdim=False) #N*B*H*W

    return patchMedia[0] #dimension 0 : value; dimension 1: indice