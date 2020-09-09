#-*-coding: utf-8-*-

###用　Dataset 类进行简单的数据加载

from __future__ import print_function, division

import time
import os

import torch
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image


class simpleDataset(data.Dataset):
    
    # initialise function of class
    def __init__(self, root, filenames, labels ):
        # the data directory 
        self.root = root
        # the list of filename
        self.filenames = filenames
        # the list of label
        self.labels = labels

    # obtain the sample with the given index
    def __getitem__(self, index):
        # obtain filenames from list
        image_filename = self.filenames[index]
        # Load data and label
        image = Image.open(os.path.join(self.root, image_filename))
        label = self.labels[index]
        
        # output of Dataset must be tensor
        image = transforms.ToTensor()(image)
        label = torch.as_tensor(label, dtype=torch.int64)
        return image, label
    
    # the total number of samples (optional)
    def __len__(self):
        return len(self.filenames)

# data directory
#root = "my_data"
root="/home/deakin/ml/data/hymenoptera_data/train/ants"

# assume we have 3 jpg images
filenames = ['img1.jpg', 'img2.jpg', 'img3.jpg']

# the class of image might be ['black cat', 'tabby cat', 'tabby cat']
labels = [0, 1, 1]

# create own Dataset
my_dataset = simpleDataset(root=root,
                           filenames=filenames,
                           labels=labels
                           )

# data loader
batch_size = 1
num_workers = 4

data_loader = torch.utils.data.DataLoader(my_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers
                                         )

for images, labels in data_loader:
    # image shape is [batch_size, 3 (due to RGB), height, width]
    img = transforms.ToPILImage()(images[0])
    plt.imshow(img)
    plt.show()
    print(labels)



