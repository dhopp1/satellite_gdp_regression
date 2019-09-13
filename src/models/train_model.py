import io
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from skimage import io, transform
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

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
    

transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])


test = SatelliteDataset("../../data/key.csv", "../../data/images/", label="population", transformations=transformations)
test.__getitem__(2, False, True)
