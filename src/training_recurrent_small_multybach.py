# -*- coding: utf-8 -*-
"""
@author: Nino Cauli
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import time

from torchvision import transforms
from torchvision import models
from model import *
from dataLoading import *
from torch.autograd import Variable

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


restore = True

newx = 640 / 4 #480 / 4
newy = 640 / 4

seq_size = 50

# check if the checkpoints dir exist otherwise create it
checkpoint_dir = '../checkpoints_rec_1_lay_small_seq_scratch_' + str(newx) +  '_' + str(newy) + '/'
old_checkpoint_dir = '../checkpoints_small_' + str(newx) +  '_' + str(newy) + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

data_transforms = transforms.Compose([Rescale((newx, newy)),
                                      ToTensor()])
train_dataset = YCBSegmentationSeq(root_dir = '/media/nigno/data/YCB/',
                                transform = data_transforms,
                                dset_type = 'train',
                                seq_size = seq_size)

print(len(train_dataset))
train_size = len(train_dataset)

dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=3,
                                               shuffle=True, num_workers=6)




def train_model(model, criterion, optimizer, scheduler, num_epochs=25, start_epoch=0, 
                loss_list=[]):
    since = time.time()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + start_epoch, num_epochs - 1 + start_epoch))
        print('-' * 10)

        if (exp_lr_scheduler != None):
            scheduler.step()
            print ('scheduler on')
        model.train(True)  # Set model to training mode
        dataloader = dataloader_train
        dataset_size = train_size

        running_loss = 0.0

        # Iterate over data.
        samples_count = 0
        print('reading data')

        for data in dataloader_train:
            samples_count += 1
            printProgressBar(samples_count, len(dataloader), prefix = 'train', suffix = 'Complete', length = 50)
            # get the inputs
            samples = data
            
            # wrap them in Variable
            if torch.cuda.is_available():
                inputs = Variable(samples['image'].cuda())
                label = Variable(samples['label'].cuda())
            else:
                inputs = Variable(samples['image'])
                label = Variable(samples['label'])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            inputs /= 255
            
            # Sequence change
            inputs = inputs.permute(1, 0, 2, 3, 4)
            label = label.permute(1, 0, 2, 3)
            
            loss = 0
            for i in range(inputs.shape[0]):
                if i == 0:
                    init_h = True
                else:
                    init_h = False
                outputs = myNet(inputs[i], init_h) 
                loss += criterion(outputs, label[i])  

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()  

            # statistics
            running_loss += loss.data

        epoch_loss = running_loss / (dataset_size * seq_size)

        print('{} Loss: {:.7f}'.format('train', epoch_loss)) 

        loss_list += [epoch_loss]

        # deep copy the model
        torch.save({
                    'epoch': epoch + start_epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss_list': loss_list,
                    'optimizer': optimizer.state_dict(),
                    }, checkpoint_dir + 'checkpointAllEpochs.tar' ) 

        if ((epoch + start_epoch + 1) % 1) == 0:
            torch.save({
                        'epoch': epoch + start_epoch + 1,
                        'state_dict': model.state_dict(),
                        'loss_list': loss_list,
                        'optimizer': optimizer.state_dict(),
                        }, checkpoint_dir + 'checkpoint' + str(epoch + start_epoch + 1) + '.tar')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

oldNet = SegCNNSimple()
cpName = old_checkpoint_dir + 'checkpointAllEpochs.tar'
if os.path.isfile(cpName):
    print("=> loading checkpoint '{}'".format(cpName))
    checkpoint = torch.load(cpName)
    oldNet.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))

myNet = SegCNNRecSimple()
myNet.features = oldNet.features
myNet.deconvFinal = oldNet.deconvFinal

#for param in myNet.features.parameters():
#    param.require_grad = False

if torch.cuda.is_available():
    myNet = myNet.cuda()
print (myNet)

criterion = nn.CrossEntropyLoss()
params = list(myNet.recurrent.parameters())
params_feat = list(myNet.features.parameters())
params_out = list(myNet.deconvFinal.parameters())
#optimizer_ft = optim.Adam(params + params_out)
optimizer_ft = optim.Adam(myNet.parameters())
exp_lr_scheduler = None

if restore == False:
    myNet = train_model(myNet, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=50)
else:
    cpName = checkpoint_dir + 'checkpointAllEpochs.tar'
    if os.path.isfile(cpName):
        print("=> loading checkpoint '{}'".format(cpName))
        checkpoint = torch.load(cpName)
        start_epoch = checkpoint['epoch']
        loss_list = checkpoint['loss_list']
        myNet.load_state_dict(checkpoint['state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))

    myNet = train_model(myNet, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=50, start_epoch=start_epoch,
                        loss_list=loss_list)




input('Press enter to continue: ')
