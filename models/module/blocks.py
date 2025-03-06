import torch
import torch.nn as nn

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, elu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_channel
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.elu = nn.ELU() if elu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.elu is not None:
            x = self.elu(x)
        return x

# 1x1卷积模块
class Conv1x1Block(nn.Module):
    def __init__(self,in_c,out_c):
        super(Conv1x1Block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False)
        self.elu = nn.ELU(inplace=True)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self,x):
        x = self.conv(x)
        x= self.elu(self.bn(x))
        return x