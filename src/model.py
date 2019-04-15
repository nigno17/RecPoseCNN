# -*- coding: utf-8 -*-
"""
@author: Nino Cauli
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models


class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


class SegCNN(torch.nn.Module):
    def __init__(self, classNum = 10):
        
        super(SegCNN, self).__init__()
        self.D_features = 256 * 6 * 6
        
        vgg16Net = models.vgg16(pretrained=True)
        self.features = vgg16Net.features
        self.deconv22 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )
        self.deconv29 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding = 1),
            nn.ReLU(inplace=True),
        )
        self.deconvFinal = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(64, 64, kernel_size=16, stride=8, padding = 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classNum, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.hk22 = Hook(self.features[22])
        self.hk29 = Hook(self.features[29])

    def forward(self, img1):
        feat = self.features(img1)
        
        dec22 = self.deconv22(self.hk22.output)
        dec29 = self.deconv29(self.hk29.output)
        
        print(self.hk22.output.shape)
        print(self.hk29.output.shape)
        print(dec22.shape)
        print(dec29.shape)

        output = self.deconvFinal(dec22 + dec29)
        return output