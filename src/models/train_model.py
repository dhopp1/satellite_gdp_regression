import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from skimage import io, transform
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler



class SatelliteDataset(Dataset):
    def __init__(self, csv_path, image_path, label="gdp", transformations=None):
        """
        parameters:
        ----------
            csv_path : str
                path to csv key file
            image_path : str
                path to image directory
            label : str
                which key column to use as the labels (e.g. "gdp", "population")
        """
        
        tmp = pd.read_csv(csv_path)
        tmp = tmp.loc[~pd.isna(tmp.capture_date),:].reset_index(drop=True)

        self.to_tensor = transforms.ToTensor()
        self.transformations = transformations
        self.data_info = tmp
        self.image_arr = image_path + self.data_info.loc[:,'id'].astype(str) + '.png'
        self.label_arr = self.data_info.loc[:,label]
        self.data_len = len(self.data_info)
        
    def __getitem__(self, index, transformed_show=False, orig_show=False):
        "transformed_show=True to show the transformed image, orig_show=True to see original image"
        
        single_image_name = self.image_arr[index]

        img = Image.open(single_image_name).convert("RGB")
        
        if self.transformations:
            img_tensor = self.transformations(img)
        else:
            img_tensor = self.to_tensor(img)

        if orig_show:
            plt.imshow(img)       
        if transformed_show:
            plt.imshow(img_tensor.permute(1,2,0))
        
        label = self.label_arr[index]
        return (img_tensor, label)
        
    def __len__(self):
        return self.data_len
    

# calculating mean and std for the three channels for normalization
if True:
    # load with no transformations
    images = SatelliteDataset("../../data/key.csv", "../../data/images/", label="population")
    # initialize arrays to store means and stdevs
    channel_1_mean = [0] * images.__len__()
    channel_2_mean = [0] * images.__len__()
    channel_3_mean = [0] * images.__len__()
    channel_1_std = [0] * images.__len__()
    channel_2_std = [0] * images.__len__()
    channel_3_std = [0] * images.__len__()
    for i in range(images.__len__()):
        channel_1_mean[i] = images.__getitem__(i)[0][0,:,:].mean()
        channel_2_mean[i] = images.__getitem__(i)[0][1,:,:].mean()
        channel_3_mean[i] = images.__getitem__(i)[0][2,:,:].mean()
        channel_1_std[i] = images.__getitem__(i)[0][0,:,:].std()
        channel_2_std[i] = images.__getitem__(i)[0][1,:,:].std()
        channel_3_std[i] = images.__getitem__(i)[0][2,:,:].std()
        print(i)
    print(f'Means :{np.mean(channel_1_mean)}, {np.mean(channel_2_mean)}, {np.mean(channel_3_mean)}')
    print(f'Std :{np.mean(channel_1_std)}, {np.mean(channel_2_std)}, {np.mean(channel_3_std)}')

# generating final dataset with transformations, below for reference
# Means :0.22847716510295868, 0.3062216639518738, 0.2528402805328369
# Std :0.13356611132621765, 0.10527726262807846, 0.11306707561016083
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[np.mean(channel_1_mean), np.mean(channel_2_mean), np.mean(channel_3_mean)],
                     std=[np.mean(channel_1_std), np.mean(channel_2_std), np.mean(channel_2_std)])
    ])
images = SatelliteDataset("../../data/key.csv", "../../data/images/", label="population", transformations=transformations)

# train test split
test_ratio = 0.2
batch_size = 10
indices = list(range(images.__len__()))
np.random.shuffle(indices)
split = int(np.floor(test_ratio * images.__len__()))
train_index, test_index = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_index)
test_sampler = SubsetRandomSampler(test_index)

test_loader = torch.utils.data.DataLoader(dataset=images, batch_size=batch_size, shuffle=False, sampler=train_sampler)
train_loader = torch.utils.data.DataLoader(dataset=images, batch_size=batch_size, shuffle=False, sampler=test_sampler)


# defining the model from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html, https://nextjournal.com/gkoehler/pytorch-mnist
channels = images.__getitem__(0)[0].shape[0]
x_dim = images.__getitem__(0)[0].shape[1]
y_dim = images.__getitem__(0)[0].shape[2]
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(channels * x_dim * y_dim, 500)
        self.fc2 = nn.Linear(500, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)