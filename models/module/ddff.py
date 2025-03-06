import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.module.attention import SpatialAttention
from models.module.blocks import Conv1x1Block, BasicConv


# adaptive spectral transformation unit
class ASTU(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(ASTU, self).__init__()
        self.conv_block1 = BasicConv(in_channel=in_channels*2, out_channel=out_channels*2,kernel_size=3,stride=1,padding=1)  # MK
        self.conv_block2 = BasicConv(in_channel=in_channels*2, out_channel=out_channels*2,kernel_size=3,stride=1,padding=1)
        self.conv_block3 = BasicConv(in_channel=in_channels, out_channel=in_channels, kernel_size=3, stride=1,padding=1)
        self.sp_att = SpatialAttention()

    def forward(self, x):

        batch, c, h, w = x.size()
        r_size = x.size()
        ffted = torch.fft.rfft2(x, dim=(-2,-1), norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag),-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])
        ffted = self.conv_block1(ffted)  #
        att_map = self.sp_att(ffted)
        ffted = ffted * att_map
        ffted = self.conv_block2(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()
        real_part = ffted[...,0]
        imag_part = ffted[...,1]
        complex_tensor = real_part + 1j*imag_part
        output = torch.fft.irfft2(complex_tensor, dim=(-2,-1), s=r_size[2:], norm='ortho')
        output = self.conv_block3(output)

        return output


# multi-scale spectral transformer module
class MSST(nn.Module):

    def __init__(self, in_channels, nb1, nb2, enable_lfu=True):

        super(MSST, self).__init__()
        self.enable_lfu = enable_lfu
        self.mid_channel = in_channels // 2
        self.nb1 = nb1
        self.nb2 = nb2

        self.conv1 = BasicConv(in_channel=in_channels,out_channel=self.mid_channel,kernel_size=1,stride=1,padding=0)
        self.fu = ASTU(self.mid_channel, self.mid_channel)
        self.semiFusionblock1 = BasicConv(in_channel=self.mid_channel, out_channel=self.mid_channel, kernel_size=3, stride=1, padding=1)
        self.semiFusionblock2 = BasicConv(in_channel=self.mid_channel, out_channel=self.mid_channel, kernel_size=3, stride=1, padding=1)
        self.semiFusionblockcat = BasicConv(in_channel=self.mid_channel * 4, out_channel=self.mid_channel, kernel_size=3, stride=1, padding=1)
        self.outputfusion = BasicConv(in_channel=self.mid_channel * 2, out_channel=in_channels, kernel_size=3, stride=1, padding=1)

        self.sp_att = SpatialAttention(kernel_size=3)
        self.sdconv = BasicConv(in_channel=in_channels,out_channel=self.mid_channel,kernel_size=3,stride=1,padding=1)

        if self.enable_lfu:
            self.semi_fu1 = []
            self.semi_fu2 = []
            for _ in range(self.nb1 ** 2):
                semi_fu = ASTU(self.mid_channel, self.mid_channel)
                self.semi_fu1.append(semi_fu)
            for _ in range(self.nb2 ** 2):
                semi_fu = ASTU(self.mid_channel, self.mid_channel)
                self.semi_fu2.append(semi_fu)

    # split features
    def split_feature (self,input):

        b, c, h, w = input.shape
        block_size1 = h // self.nb1
        block_size2 = h // self.nb2
        patches1 = []
        patches2 = []
        for i in range(0, h, block_size1):
            for j in range(0, h, block_size1):
                patch = input[:, :, i:i + block_size1, j:j + block_size1]
                patches1.append(patch)
        for i in range(0, h, block_size2):
            for j in range(0, h, block_size2):
                patch = input[:, :, i:i + block_size2, j:j + block_size2]
                patches2.append(patch)
        return patches1,patches2

    def forward(self, x):

        xsd = x
        x = self.conv1(x)
        output_fu = self.fu(x)
        if self.enable_lfu:
            feature1 = []
            feature2 = []
            patches1,patches2 = self.split_feature(x)
            for i in range(self.nb1 ** 2):
                semi_fu1 = self.semi_fu1[i]
                patch1 = patches1[i]
                semi_fu_patch1 = semi_fu1.to(patch1.device)(patch1)
                feature1.append(semi_fu_patch1)

            for j in range(self.nb2 ** 2):
                semi_fu2 = self.semi_fu2[j]
                patch2 = patches2[j]
                semi_fu_patch2 = semi_fu2.to(patch2.device)(patch2)
                feature2.append(semi_fu_patch2)
            tem_feature1 = []
            tem_feature2 = []
            for i in range (0, self.nb1 ** 2, self.nb1):
                feature = torch.cat(feature1[i:i+self.nb1],dim=3)
                tem_feature1.append(feature)
            semi_out1 = torch.cat(tem_feature1,dim=2)

            for i in range (0, self.nb2 ** 2, self.nb2):
                feature = torch.cat(feature2[i:i+self.nb2],dim=3)
                tem_feature2.append(feature)
            semi_out2 = torch.cat(tem_feature2,dim=2)
            semi_out1 = self.semiFusionblock1(semi_out1)
            semi_out2 = self.semiFusionblock2(semi_out2)
            xsd = xsd * self.sp_att(xsd)
            sd_output = self.sdconv(xsd)
            combine_fusion = sd_output+output_fu+semi_out1+semi_out2
            ffc_cat = torch.cat((sd_output,output_fu,semi_out1,semi_out2),dim=1)
            cat_fusion = self.semiFusionblockcat(ffc_cat)
            output = torch.cat((combine_fusion,cat_fusion),dim=1)
            output = self.outputfusion(output)
        else:
            x_fu = torch.cat((x,output_fu),dim=1)
            output = self.outputfusion(x_fu)

        return output

#  dual-path dual-domain feature fusion module
class DDFF(nn.Module):

    def __init__(self, in_channels, kernel_size, nb1, nb2, stride=1, padding=0, bias=False, enable_lfu=True):
        super(DDFF, self).__init__()

        self.stride = stride
        mid_channels = in_channels // 2
        self.xlconv = BasicConv(in_channel=in_channels, out_channel=mid_channels, kernel_size=1, stride=1, padding=0)  # 160->80 MK
        self.xgconv = BasicConv(in_channel=in_channels, out_channel=mid_channels, kernel_size=1, stride=1, padding=0)   #160->80

        self.convl2l = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.convl2g = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.convg2l = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.convg2g = MSST(mid_channels, nb1=nb1, nb2=nb2, enable_lfu=enable_lfu)

        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.elu = nn.ELU(inplace=True)
        self.spatt = SpatialAttention(kernel_size=3)

    def forward(self, x):

        x_l = self.xlconv(x)
        x_g = self.xgconv(x)
        # local features
        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xl = self.elu(self.bn1(out_xl))
        att = self.spatt(out_xl)
        out_xl = out_xl * att
        # multi-scale global features
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)  #[b,c/2,h,w]
        out_xg = self.elu(self.bn2(out_xg))
        # fused features
        out_xl_xg = torch.cat((out_xl,out_xg),dim=1)  #[b,c,h,w]

        return out_xl_xg




