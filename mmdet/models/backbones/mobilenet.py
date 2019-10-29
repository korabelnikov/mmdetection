from typing import Tuple

from ..registry import BACKBONES
# borrowed from "https://github.com/marvis/pytorch-mobilenet"

import torch.nn as nn
import torch.nn.functional as F


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        #self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        #x = self.fc(x)
        return x


from torch.nn import Conv2d, Sequential, ModuleList, ReLU

@BACKBONES.register_module
class MobileNet(MobileNetV1):
    def __init__(self, out_feature_indices=(11, 13), **kwargs):
        super().__init__(**kwargs)
        self.extras = ModuleList([
            Sequential(
                Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            )
        ])
        self.out_feature_indices = out_feature_indices

    def forward(self, x):
        outs = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in self.out_feature_indices:
                outs.append(x)
        for layer in self.extras:
            x = layer(x)
            outs.append(x)

        return tuple(outs)  # should return a tuple

    def init_weights(self, pretrained=None):
        if pretrained:
            raise NotImplementedError()
            #TODO look to self.load_state_dict()

""" template is as following

import torch.nn as nn

@BACKBONES.register_module
class MobileNet(nn.Module):

    def __init__(self, arg1, arg2):
        pass

    def forward(x):  # should return a tuple
        pass
"""