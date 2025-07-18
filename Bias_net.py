from net_parts import *
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self,in_chan):
        super(Generator,self).__init__()

        self.inc = inconv(in_chan,64)
        self.down1 = down(64,128)
        self.down2 = down(128,256)
        self.down3 = down(256,512)
        self.down4 = down(512,512)
        self.up1 = up(1024,256)
        self.up2 = up(512,128)
        self.up3 = up(256,64)
        self.up4 = up(128,64)
        self.outc = outconv(64,1)

    def forward(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.outc(x)

        return F.sigmoid(x)