import torch
import torch.nn as nn
import torch.nn.functional as F

class bias_conv(nn.Module):
    def __init__(self,in_ch,out_ch,ker_size,padding = 0):
        super(bias_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,ker_size, padding = padding),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=False)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(double_conv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1),
            nn.InstanceNorm2d(out_ch,eps=1e-05,affine=True),
            nn.LeakyReLU(negative_slope=0.01,inplace=False),

            nn.Conv2d(out_ch,out_ch,3,padding=1),
            nn.InstanceNorm2d(out_ch,eps=1e-05,affine=True),
            nn.LeakyReLU(negative_slope=0.01,inplace=False)
        )
    def forward(self,x):
        x = self.conv(x)
        return x
class inconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(inconv, self).__init__()
        self.conv=double_conv(in_ch,out_ch)
    def forward(self,x):
        x=self.conv(x)
        return x

class down(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(down,self).__init__()
        self.conv=nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch,out_ch)
        )

    def forward(self,x):
        x=self.conv(x)
        return x

class up(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(up,self).__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=double_conv(in_ch,out_ch)
    def forward(self,xl,xr):
        xr=self.up(xr)
        diffH=xr.size()[2] - xl.size()[2]
        diffW=xr.size()[3] - xl.size()[3]
        xl=F.pad(xl,(diffW//2,diffW-diffW//2,diffH//2,diffH-diffH//2))
        x=torch.cat([xr,xl],dim=1)
        x=self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(outconv, self).__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=1)

    def forward(self,x):
        x=self.conv(x)
        return x

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class ResidualConv(nn.Module):
    def __init__(self, in_ch, out_ch, dial, stride = 1):
        super(ResidualConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, dilation = dial, padding=dial),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, dilation = dial, padding=dial),
            nn.BatchNorm2d(out_ch),
        )
        self.re = nn.ReLU(inplace=True)
        self.shorcut = nn.Sequential()
        if(stride != 1 or in_ch != out_ch):
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        out = self.conv(x)
        out = out + self.shorcut(x)
        out = self.relu(out)
        return out