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


restore = False

newx = 640 / 4 #480 / 4
newy = 640 / 4

# check if the checkpoints dir exist otherwise create it
checkpoint_dir = '../checkpoints_rec_onlyrec_1_lay' + str(newx) +  '_' + str(newy) + '/'
old_checkpoint_dir = '../checkpoints_' + str(newx) +  '_' + str(newy) + '/'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

data_transforms = transforms.Compose([Rescale((newx, newy)),
                                      ToTensor()])
train_dataset = YCBSegmentationH5(root_dir = '/media/nigno/data/YCB_h5/',
                                transform = data_transforms,
                                dset_type = 'train')

print(len(train_dataset))
train_size = len(train_dataset)

dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               shuffle=False, num_workers=1)




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
                init_h_temp = Variable(samples['startSeq'].cuda())
                init_h = bool(sum(init_h_temp.data.cpu().numpy()))
            else:
                inputs = Variable(samples['image'])
                label = Variable(samples['label']),
                init_h_temp = Variable(samples['startSeq'].cuda())
                init_h = bool(sum(init_h_temp.data.numpy()))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = myNet(inputs, init_h) 
            loss = criterion(outputs, label)  

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()  

            # statistics
            running_loss += loss.data

        epoch_loss = running_loss / dataset_size 

        print('{} Loss: {:.7f}'.format('train', epoch_loss)) 

        loss_list += [epoch_loss]

        # deep copy the model
        torch.save({
                    'epoch': epoch + start_epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss_list': loss_list,
                    'optimizer': optimizer.state_dict(),
                    }, checkpoint_dir + 'checkpointAllEpochs.tar' ) 

        if ((epoch + start_epoch + 1) % 10) == 0:
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

oldNet = SegCNN()
cpName = old_checkpoint_dir + 'checkpointAllEpochs.tar'
if os.path.isfile(cpName):
    print("=> loading checkpoint '{}'".format(cpName))
    checkpoint = torch.load(cpName)
    oldNet.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))

myNet = SegCNNRec(n_rec_layers = 1, hidden_sizes = [512], kernel_sizes = [3])
#myNet.features = oldNet.features
myNet.deconv22 = oldNet.deconv22
myNet.deconv29 = oldNet.deconv29
myNet.deconvFinal = oldNet.deconvFinal

for param in myNet.features.parameters():
    param.requires_grad = False

if torch.cuda.is_available():
    myNet = myNet.cuda()
print (myNet)

criterion = nn.CrossEntropyLoss()
params = list(myNet.recurrent22.parameters()) + list(myNet.recurrent29.parameters())
params_out = list(myNet.deconv22.parameters()) + list(myNet.deconv29.parameters()) + list(myNet.deconvFinal.parameters())
optimizer_ft = optim.Adam(params)
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
