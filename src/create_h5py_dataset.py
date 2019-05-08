from __future__ import print_function, division
import os
import torch
from skimage import transform, io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import h5py
import datetime as dt

        
dset_type = 'train'

root_dir = '../data/YCB/'
new_dataset_root_dir = '../data/YCB_h5/'
if not os.path.exists(new_dataset_root_dir):
    os.makedirs(new_dataset_root_dir)
data_dir = root_dir + 'data/'
sets_dir = root_dir + 'image_sets/'



data_list_file_name = sets_dir + dset_type + '.txt'
with open(data_list_file_name) as f:
    data_list = f.read().splitlines() 

start = dt.datetime.now()

with h5py.File(new_dataset_root_dir + dset_type + '.h5', 'w') as hf:
    count = 0
    startSeq = np.ones(len(data_list), dtype=np.uint8)
    for file_name in data_list:

        img_name = data_dir + file_name + '-color.png'
        cv_image = cv2.imread(img_name)
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        imageSet = hf.create_dataset(
                name='image' + str(count),
                data=image,
                shape=image.shape,
                compression="lzf")

        label_name = data_dir + file_name + '-label.png'
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        labelSet = hf.create_dataset(
                name='label' + str(count),
                data=label,
                shape=label.shape,
                compression="lzf")

        seq = list(map(int, file_name.split('/')))
        if seq[1] == 1:
            startSeq[count] = 1
        else:
            startSeq[count] = 0

        startSeqSet = hf.create_dataset(
                name='startSeq' + str(count),
                data=startSeq[count],
                shape=(1,),
                compression="lzf")

        end=dt.datetime.now()
        print(str(count) + ' : ' + str((end-start).seconds) + 'seconds')

        count += 1