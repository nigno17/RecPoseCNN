# -*- coding: utf-8 -*-
"""
@author: Nino Cauli
"""

from __future__ import print_function, division

import torch
#import torch.nn as nn
#import torch.optim as optim
import numpy as np
import cv2
#import time

from torchvision import transforms
#from torchvision import models
from model import *
from dataLoading import *
from torch.autograd import Variable


newx = 640 / 4 #480 / 4
newy = 640 / 4

# check if the checkpoints dir exist otherwise create it
checkpoint_dir = '../checkpoints_small_' + str(newx) +  '_' + str(newy) + '/'

data_transforms = transforms.Compose([Rescale((newx, newy)),
                                      ToTensor()])

#checkpoint_dir = '../checkpoints/'

#data_transforms = transforms.Compose([ToTensor()])                                      
dataset = YCBSegmentation(root_dir = '../data/YCB/', 
                          transform = data_transforms,
                          dset_type = 'train')

print(len(dataset))
dataset_size = len(dataset)   

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=1)


myNet = SegCNNSimple()
myNet.train(False)
cpName = checkpoint_dir + 'checkpointAllEpochs.tar'
if os.path.isfile(cpName):
    print("=> loading checkpoint '{}'".format(cpName))
    checkpoint = torch.load(cpName)
    myNet.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))

if torch.cuda.is_available():
    myNet = myNet.cuda()
print (myNet)

samples_count = 0
for data in dataloader:
    samples_count += 1
    # get the inputs
    samples = data
    
    # wrap them in Variable
    if torch.cuda.is_available():
        inputs = Variable(samples['image'].cuda())
        label = Variable(samples['label'].cuda())
    else:
        inputs, label = Variable(samples['image']), Variable(samples['label'])
    
    inputNet = inputs / 255
    outputs = myNet(inputNet)

    label_pred =  torch.argmax(outputs, dim=1)
    if torch.cuda.is_available():
        npinputs = inputs[0].data.cpu().numpy()
        nplabel = label[0].data.cpu().numpy()
        nplabel_pred = label_pred[0].data.cpu().numpy()
        npsingleclass = outputs[0].data.cpu().numpy()
    else:
        npinputs = inputs[0].data.numpy()
        nplabel = label[0].data.numpy()
        nplabel_pred = label_pred[0].data.numpy()
        npsingleclass = outputs[0].data.numpy()

    npinputs = npinputs.transpose((1, 2, 0))

    classList = np.unique(nplabel.astype(dtype=np.uint8))

    nplabel = nplabel / 22.0 * 255
    nplabel_pred = nplabel_pred / 22.0 * 255

    nplabel_color = cv2.applyColorMap(nplabel.astype(dtype=np.uint8), cv2.COLORMAP_JET)
    nplabel_pred_color = cv2.applyColorMap(nplabel_pred.astype(dtype=np.uint8), cv2.COLORMAP_JET)

    cv2.imshow('image', npinputs.astype(dtype=np.uint8))
    cv2.imshow('label', nplabel_color)
    cv2.imshow('label_pred', nplabel_pred_color)

    for classNum in classList:
        nptempclass = (npsingleclass[classNum] - np.min(npsingleclass[classNum])) / np.max(npsingleclass[classNum]) * 255
        #nptempclass = npsingleclass[classNum] / np.max(npsingleclass[classNum]) * 255
        nptempclass = nptempclass.astype(dtype=np.uint8)
        cv2.imshow('label_pred ' + str(classNum), nptempclass)

    c = cv2.waitKey(0)

    if c == 27:
        print('quitting')
        break

