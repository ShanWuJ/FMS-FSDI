import torch
import torch.nn as nn
from models.module.blocks import BasicConv

# joint prompt for enhancing semantics module
class JPESM(nn.Module):
    def __init__(self, resnet, kernel_size=3):
        super(JPESM, self).__init__()
        if resnet:
            self.num_channel = 640
            self.TG_prompt = nn.Parameter(torch.randn((1, self.num_channel, 5, 5)))   # 这里就是那个TG_prompt  MK
        else:
            self.num_channel = 64
            self.TG_prompt = nn.Parameter(torch.randn((1, self.num_channel, 5, 5)))

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv =BasicConv(self.num_channel, self.num_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, s):
        pt_s = s + self.TG_prompt
        avg_out = torch.mean(pt_s, dim=1, keepdim=True)
        max_out, _ = torch.max(pt_s, dim=1, keepdim=True)
        att = torch.cat([avg_out, max_out], dim=1)
        att = self.conv1(att)
        att_map = self.sigmoid(att)
        out = pt_s * att_map
        out = self.conv(out)
        return out
