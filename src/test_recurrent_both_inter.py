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
import matplotlib.pyplot as plt


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()



newx = 640 / 4 #480 / 4
newy = 640 / 4

# check if the checkpoints dir exist otherwise create it
checkpoint_rec_dir = '../checkpoints_rec_1_lay_small' + str(newx) +  '_' + str(newy) + '/'
checkpoint_dir = '../checkpoints_small_' + str(newx) +  '_' + str(newy) + '/'

data_transforms = transforms.Compose([Rescale((newx, newy)),
                                      ToTensor()])

#checkpoint_dir = '../checkpoints_rec/'

#data_transforms = transforms.Compose([ToTensor()])
dataset = YCBSegmentation(root_dir = '../data/YCB/', 
                          transform = data_transforms,
                          dset_type = 'train')

print(len(dataset))
dataset_size = len(dataset)   

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=1)


myNetRec = SegCNNRecSimple(n_rec_layers = 1, hidden_sizes = [128], kernel_sizes = [3])
myNetRec.train(False)
cpName = checkpoint_rec_dir + 'checkpointAllEpochs.tar'
if os.path.isfile(cpName):
    print("=> loading checkpoint '{}'".format(cpName))
    checkpoint = torch.load(cpName)
    myNetRec.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))

if torch.cuda.is_available():
    myNetRec = myNetRec.cuda()
print (myNetRec)

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
IoU_total1 = 0
IoU_total2 = 0
IoU_list1 = []
IoU_list2 = []
for data in dataloader:
    samples_count += 1
    printProgressBar(samples_count, len(dataloader), prefix = 'testing', suffix = 'Complete', length = 50)
    # get the inputs
    samples = data

     # wrap them in Variable
    if torch.cuda.is_available():
        inputs = Variable(samples['image'].cuda())
        label = Variable(samples['label'].cuda())
        init_h_temp = Variable(samples['startSeq'].cuda())
        init_h = bool(sum(init_h_temp.data.cpu().numpy()))
    else:
        inputs = Variable(samples['image'])
        label = Variable(samples['label']),
        init_h_temp = Variable(samples['startSeq'].cuda())
        init_h = bool(sum(init_h_temp.data.numpy()))

    inputNet = inputs / 255
    
    outputs = myNetRec(inputNet, init_h)
    outputs2 = myNet(inputNet)

    label_pred =  torch.argmax(outputs, dim=1)
    label_pred2 =  torch.argmax(outputs2, dim=1)
    if torch.cuda.is_available():
        npinputs = inputs[0].data.cpu().numpy()
        nplabel = label[0].data.cpu().numpy()
        nplabel_pred = label_pred[0].data.cpu().numpy()
        nplabel_pred2 = label_pred2[0].data.cpu().numpy()
        npsingleclass = outputs[0].data.cpu().numpy()
    else:
        npinputs = inputs[0].data.numpy()
        nplabel = label[0].data.numpy()
        nplabel_pred = label_pred[0].data.numpy()
        nplabel_pred2 = label_pred2[0].data.numpy()
        npsingleclass = outputs[0].data.numpy()

    npinputs = npinputs.transpose((1, 2, 0))

    IoU1 = 0
    IoU2 = 0
    for i in range(22):
        tempLab = (nplabel == i)
        tempLab_pred = (nplabel_pred == i)
        tempLab_pred2 = (nplabel_pred2 == i)

        intersection1 = np.sum(np.logical_and(tempLab, tempLab_pred))
        intersection2 = np.sum(np.logical_and(tempLab, tempLab_pred2))

        union1 = np.sum(np.logical_or(tempLab, tempLab_pred))
        union2 = np.sum(np.logical_or(tempLab, tempLab_pred2))

        if union1 > 0:
            IoU1 += intersection1 / union1
        else:
            IoU1 += 1

        if union2 > 0:
            IoU2 += intersection2 / union2
        else:
            IoU2 += 1
    
    IoU_list1.append(IoU1 / 22)
    IoU_list2.append(IoU2 / 22)

    IoU_total1 += IoU1 / 22
    IoU_total2 += IoU2 / 22
    
    # print(IoU_total1)
    # print(IoU_total2)
    # print('--------------')
    # input('Press enter to continue: ')
IoU_total1 /= dataset_size
IoU_total2 /= dataset_size

plt.plot(IoU_list1)
plt.plot(IoU_list2)
plt.show()
print ('Recurrent IoU: ' + str(IoU_total1))
print ('Normal IoU: ' + str(IoU_total2))
input('Press enter to continue: ')
