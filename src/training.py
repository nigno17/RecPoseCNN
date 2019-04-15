# -*- coding: utf-8 -*-
"""
@author: Nino Cauli
"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models
from model import SegCNN


cv_image = cv2.imread('../data/sardegna.jpg')
image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))
input = torch.from_numpy(image).float()
input = input.unsqueeze(0)

myNet = SegCNN()
if torch.cuda.is_available():
 myNet = myNet.cuda()
print (myNet)

output = myNet(input.cuda())

print(output.shape)
