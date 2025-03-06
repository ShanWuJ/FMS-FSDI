import torch
import torch.nn as nn

# noise filter in SMS3F
class NoiseFilter(nn.Module):
    def __init__(self, resnet):
        super(NoiseFilter,self).__init__()
        if resnet:
            self.num_channel = 640
            self.fiter_g2 = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fiter_g3 = nn.Sequential(
                nn.Conv2d(self.num_channel, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0)
            )

        else:
            self.num_channel = 64
            self.fiter_g2 = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0)
            )
            self.fiter_g3 = nn.Sequential(
                nn.Conv2d(self.num_channel, 30, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(30),
                nn.ReLU(inplace=True),
                nn.Conv2d(30, 1, kernel_size=1, stride=1, padding=0)
            )


    def forward(self, F_2, F_3, F_4):
        heat_map_2 = nn.functional.interpolate(F_4, size=(F_2.shape[-1], F_2.shape[-1]), mode='bilinear', align_corners=False)
        fiter_2 = nn.Sigmoid()(self.fiter_g2(heat_map_2))
        f_2 = F_2 * fiter_2

        heat_map_3 = nn.functional.interpolate(F_4, size=(F_3.shape[-1], F_3.shape[-1]), mode='bilinear', align_corners=False)
        fiter_3 = nn.Sigmoid()(self.fiter_g3(heat_map_3))
        f_3 = F_3 * fiter_3

        return f_2,f_3


