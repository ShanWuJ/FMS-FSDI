from models.backbones import ResNet
from models.backbones import Conv_4
from models.module.noisefilter import NoiseFilter
from models.module.ddff import DDFF
import torch.nn as nn
from models.module.blocks import BasicConv

# spatial and multi-scale frequency domain feature fusion
class SMS3F(nn.Module):
    def __init__(self,resnet,ffc=True,enable_lfu=True):
        super().__init__()
        self.resnet = resnet
        self.ffc = ffc
        if self.resnet:
            l2_channel = 160
            l3_channel = 320
            self.feature_extractor = ResNet.resnet12()
            self.fc_l2 = BasicConv(in_channel=160, out_channel=160, kernel_size=3, stride=1, padding=1)
            self.fc_l3 = BasicConv(in_channel=320, out_channel=320, kernel_size=3, stride=1, padding=1)
            self.fc_l4 = BasicConv(in_channel=640, out_channel=640, kernel_size=3, stride=1, padding=1)

        else:
            num_channel = 64
            l2_channel = 64
            l3_channel = 64
            self.feature_extractor = Conv_4.BackBone(num_channel)
            self.fc_l2 = BasicConv(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1)
            self.fc_l3 = BasicConv(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1)
            self.fc_l4 = BasicConv(in_channel=64, out_channel=64, kernel_size=3, stride=1, padding=1)
        self.l2c = l2_channel
        self.l3c = l3_channel
        self.nf = NoiseFilter(resnet)
        if ffc:
            self.l2ffc = DDFF(self.l2c, kernel_size=3, nb1=3, nb2=7, stride=1, padding=1, bias=False, enable_lfu=enable_lfu)
            self.l3ffc = DDFF(self.l3c, kernel_size=3, nb1=2, nb2=5, stride=1, padding=1, bias=False, enable_lfu=enable_lfu)

    def forward(self,x):

        l2,l3,l4 = self.feature_extractor(x)
        f2,f3 = self.nf(l2,l3,l4)
        if self.ffc:
            l2_xl_xg = self.l2ffc(f2)
            l3_xl_xg = self.l3ffc(f3)
        else:
            l2_xl_xg = f2
            l3_xl_xg = f3
        l2_xl_xg = self.fc_l2(l2_xl_xg)
        l3_xl_xg = self.fc_l3(l3_xl_xg)
        l4 = self.fc_l4(l4)

        return l2_xl_xg,l3_xl_xg,l4








