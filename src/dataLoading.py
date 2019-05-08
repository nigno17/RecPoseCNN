from __future__ import print_function, division
import os
import torch
from skimage import transform, io
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import h5py
from PIL import Image


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image,label,startSeq = sample['image'], sample['label'], sample['startSeq']

        new_x, new_y = self.output_size

        image = cv2.resize(image, (int(new_y), int(new_x)), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (int(new_y), int(new_x)), interpolation=cv2.INTER_NEAREST)
        #image = transform.resize(image, (int(new_x), int(new_y)))
        #label = transform.resize(label, (int(new_x), int(new_y)))

        return {'image': image,
                'label': label,
                'startSeq': startSeq}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image,label,startSeq = sample['image'], sample['label'], sample['startSeq']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).long(),
                'startSeq': torch.tensor(startSeq).byte()}
    
class YCBSegmentation(Dataset):
    """Segmentation dataset."""

    def __init__(self, root_dir, transform=None, dset_type='train'):
        """
        Args:
            root_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            dset_type (string): String to decide if use the training set, the validation set or the test set (possible values: train, trainval, val or keyframe).
        """
        
        self.data_dir = root_dir + 'data/'
        self.sets_dir = root_dir + 'image_sets/'

        data_list_file_name = self.sets_dir + dset_type + '.txt'
        with open(data_list_file_name) as f:
            self.data_list = f.read().splitlines() 
        
        count = 0
        self.startSeq = np.ones(len(self.data_list), dtype=np.uint8)
        for file_name in self.data_list:
            seq = list(map(int, file_name.split('/')))
            if seq[1] == 1:
                self.startSeq[count] = 1
            else:
                self.startSeq[count] = 0
            count += 1

        self.transform = transform

    def __len__(self):
        dset_size = len(self.data_list)
        return dset_size

    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_list[idx] + '-color.png'
        cv_image = cv2.imread(img_name)
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        label_name = self.data_dir + self.data_list[idx] + '-label.png'
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        
        sample = {'image': image, 'label': label, 'startSeq': self.startSeq[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class YCBSegmentationPIL(Dataset):
    """Segmentation dataset."""

    def __init__(self, root_dir, transform=None, dset_type='train'):
        """
        Args:
            root_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            dset_type (string): String to decide if use the training set, the validation set or the test set (possible values: train, trainval, val or keyframe).
        """
        
        self.data_dir = root_dir + 'data/'
        self.sets_dir = root_dir + 'image_sets/'

        data_list_file_name = self.sets_dir + dset_type + '.txt'
        with open(data_list_file_name) as f:
            self.data_list = f.read().splitlines() 
        
        count = 0
        self.startSeq = np.ones(len(self.data_list), dtype=np.uint8)
        for file_name in self.data_list:
            seq = list(map(int, file_name.split('/')))
            if seq[1] == 1:
                self.startSeq[count] = 1
            else:
                self.startSeq[count] = 0
            count += 1

        self.transform = transform

    def __len__(self):
        dset_size = len(self.data_list)
        return dset_size

    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_list[idx] + '-color.png'
        #cv_image = cv2.imread(img_name)
        #image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.open(img_name)

        label_name = self.data_dir + self.data_list[idx] + '-label.png'
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        
        sample = {'image': image, 'label': label, 'startSeq': self.startSeq[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class YCBSegmentationH5(Dataset):
    """Segmentation dataset."""

    def __init__(self, root_dir, transform=None, dset_type='train'):
        """
        Args:
            root_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            dset_type (string): String to decide if use the training set, the validation set or the test set (possible values: train, trainval, val or keyframe).
        """
        super(YCBSegmentationH5, self).__init__()

        self.hf = h5py.File(root_dir + dset_type + '.h5', 'r')

        self.dset_size = int(len(self.hf) / 3)

        print(self.dset_size)

        self.transform = transform

    def __len__(self):
        return self.dset_size

    def __getitem__(self, idx):
        img_name = 'image' + str(idx)
        image = self.hf[img_name][:,:,:]

        label_name = 'label' + str(idx)
        label = self.hf[label_name][:,:]

        startSeq_name = 'startSeq' + str(idx)
        startSeq = self.hf[startSeq_name][:]
        
        sample = {'image': image, 'label': label, 'startSeq': startSeq}

        if self.transform:
            sample = self.transform(sample)

        return sample

class YCBSegmentationSeq(Dataset):
    """Segmentation dataset."""

    def __init__(self, root_dir, transform=None, dset_type='train', seq_size=50):
        """
        Args:
            root_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
            dset_type (string): String to decide if use the training set, the validation set or the test set (possible values: train, trainval, val or keyframe).
        """
        
        self.data_dir = root_dir + 'data/'
        self.sets_dir = root_dir + 'image_sets/'

        self.seq_size = seq_size

        data_list_file_name = self.sets_dir + dset_type + '.txt'
        with open(data_list_file_name) as f:
            self.data_list = f.read().splitlines() 
        
        self.chunks = [self.data_list[x:x+seq_size] for x in range(0, len(self.data_list), self.seq_size)]
        
        count = 0
        self.startSeq = np.ones(len(self.data_list), dtype=np.uint8)
        for file_name in self.data_list:
            seq = list(map(int, file_name.split('/')))
            if seq[1] == 1:
                self.startSeq[count] = 1
            else:
                self.startSeq[count] = 0
            count += 1

        self.transform = transform

    def __len__(self):
        dset_size = len(self.chunks)
        return dset_size

    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_list[idx] + '-color.png'
        cv_image = cv2.imread(img_name)
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        label_name = self.data_dir + self.data_list[idx] + '-label.png'
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        
        sample = {'image': image, 'label': label, 'startSeq': self.startSeq[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample