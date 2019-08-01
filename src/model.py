# -*- coding: utf-8 -*-
"""
@author: Nino Cauli
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models
from convgru import ConvGRU
from torch.autograd import Variable


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
    def __init__(self, classNum = 22):
        
        super(SegCNN, self).__init__()

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

        output = self.deconvFinal(dec22 + dec29)
        return output

class SegCNNRec(torch.nn.Module):
    def __init__(self, classNum = 22, n_rec_layers = 2, hidden_sizes = [512, 512], kernel_sizes = [3, 3]):
        
        super(SegCNNRec, self).__init__()
        self.n_rec_layers = n_rec_layers

        self.hidden22 = [None]*self.n_rec_layers
        self.hidden29 = [None]*self.n_rec_layers
        
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
        self.recurrent22 = ConvGRU(input_size = 512, hidden_sizes = hidden_sizes,
                                   kernel_sizes = kernel_sizes, n_layers = n_rec_layers)
        self.recurrent29 = ConvGRU(input_size = 512, hidden_sizes = hidden_sizes,
                                   kernel_sizes = kernel_sizes, n_layers = n_rec_layers)
        self.relu = nn.ReLU(inplace=True)
        self.deconvFinal = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.ConvTranspose2d(64, 64, kernel_size=16, stride=8, padding = 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classNum, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

        self.hk22 = Hook(self.features[22])
        self.hk29 = Hook(self.features[29])

    def forward(self, img1, init_h = True):
        feat = self.features(img1)

        if init_h == True:
            rec22 = self.recurrent22(self.hk22.output)
            rec29 = self.recurrent29(self.hk29.output)
        else:
            rec22 = self.recurrent22(self.hk22.output, self.hidden22)
            rec29 = self.recurrent29(self.hk29.output, self.hidden29)

        #self.hidden22 = [x.detach() for x in rec22]
        #self.hidden29 = [x.detach() for x in rec29]
	self.hidden22 = rec22
        self.hidden29 = rec29

        rec22out = self.relu(rec22[-1])
        rec29out = self.relu(rec29[-1])

        dec22 = self.deconv22(rec22out)
        dec29 = self.deconv29(rec29out)

        output = self.deconvFinal(dec22 + dec29)
        return output

class SegCNNRecSimple(torch.nn.Module):
    def __init__(self, classNum = 22, n_rec_layers = 2, hidden_sizes = [128, 128], kernel_sizes = [3, 3]):
        
        super(SegCNNRecSimple, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.recurrent = ConvGRU(input_size = 128, hidden_sizes = hidden_sizes,
                                   kernel_sizes = kernel_sizes, n_layers = n_rec_layers)
        self.relu = nn.ReLU(inplace=True)
        self.deconvFinal = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=16, stride=8, padding = 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classNum, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, img1, init_h = True):
        feat = self.features(img1)

        if init_h == True:
            rec = self.recurrent(feat)
        else:
            rec = self.recurrent(feat, self.hidden)

        #self.hidden = [x.detach() for x in rec]
	self.hidden = rec

        recOut = self.relu(rec[-1])

        output = self.deconvFinal(recOut)
        return output

class SegCNNRecSimpleSum(torch.nn.Module):
    def __init__(self, classNum = 22, n_rec_layers = 2, hidden_sizes = [128, 128], kernel_sizes = [3, 3]):
        
        super(SegCNNRecSimpleSum, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.recurrent = ConvGRU(input_size = 128, hidden_sizes = hidden_sizes,
                                   kernel_sizes = kernel_sizes, n_layers = n_rec_layers)
        self.relu = nn.ReLU(inplace=True)
        self.deconvFinal = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=16, stride=8, padding = 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classNum, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, img1, init_h = True):
        feat = self.features(img1)

        if init_h == True:
            rec = self.recurrent(feat)
        else:
            rec = self.recurrent(feat, self.hidden)

        self.hidden = [x.detach() for x in rec]

        recOut = self.relu(rec[-1])

        output = self.deconvFinal(feat + recOut)
        return output

class SegCNNRecSimpleFlat(torch.nn.Module):
    def __init__(self, classNum = 22, n_rec_layers = 1, image_size=(160, 160)):
        
        super(SegCNNRecSimpleFlat, self).__init__()

        self.image_size = image_size
        self.input_size = 128 * (self.image_size[0] / 8) * (self.image_size[1] / 8)
        self.hidden_size = self.input_size // 100
        self.num_layers = n_rec_layers

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.recurrent = RNNMultyGRU(int(self.input_size), int(self.hidden_size), int(self.input_size), n_rec_layers)
        self.deconvFinal = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=16, stride=8, padding = 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classNum, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, img1, init_h = True):
        featOr = self.features(img1)

        shape = featOr.shape
        
        feat = torch.reshape(featOr, (1, -1, int(self.input_size)))
        if init_h == True:
            hidden = self.recurrent.initHidden()
            self.hidden = [x.detach() for x in hidden]
       
        out, hidden = self.recurrent(feat, self.hidden)

        rec = torch.reshape(out, shape)

        self.hidden = [x.detach() for x in hidden]
        #self.hidden = hidden

        output = self.deconvFinal(featOr + rec)
        return output

class RNNMultyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNNMultyGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.i2h = nn.GRU(input_size, hidden_size, num_layers)
        self.i2o = nn.Linear(hidden_size, output_size)
        #self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        hOut, hidden = self.i2h(input, hidden)
        output = self.i2o(hOut)
        #output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        if torch.cuda.is_available():
           hidden = hidden.cuda()
        return hidden

class SegCNNSimple(torch.nn.Module):
    def __init__(self, classNum = 22):
        
        super(SegCNNSimple, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.deconvFinal = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 64, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=16, stride=8, padding = 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, classNum, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, img1):
        feat = self.features(img1)

        output = self.deconvFinal(feat)
        return output
