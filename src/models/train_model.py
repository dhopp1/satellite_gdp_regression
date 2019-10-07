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

# parameters
label = "gdp"
test_ratio = 0.2
batch_size = 10
lr = 1e-3
n_epochs = 20
notes=''

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
        self.city_arr = self.data_info.loc[:,'city']
        self.state_arr = self.data_info.loc[:,'state']
        self.country_arr = self.data_info.loc[:,'country']
        self.region_arr = self.data_info.loc[:,'region']
        
        self.data_len = len(self.data_info)
        
    def __getitem__(self, index, transformed_show=False, orig_show=False):
        "transformed_show=True to show the transformed image, orig_show=True to see original image"
        
        single_image_name = self.image_arr[index]

        img = Image.open(single_image_name).convert("RGB")
        
        if self.transformations:
            img_tensor = self.transformations(img)
        else:
            img_tensor = self.to_tensor(img)

        city = self.city_arr[index]
        state = self.state_arr[index]
        country = self.country_arr[index]
        region = self.region_arr[index]
        if orig_show:
            plt.imshow(img)
            plt.title(f'{city}, {state}, {country}, {region}')
        if transformed_show:
            plt.imshow(img_tensor.permute(1,2,0))
            plt.title(f'{city}, {state}, {country}, {region}')
        
        label = self.label_arr[index]
        
        if orig_show == False and transformed_show == False:
            return (img_tensor, label)
        
    def __len__(self):
        return self.data_len
    

# calculating mean and std for the three channels for normalization
if False:
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
    means = [np.mean(channel_1_mean), np.mean(channel_2_mean), np.mean(channel_3_mean)]
    std = [np.mean(channel_1_std), np.mean(channel_2_std), np.mean(channel_2_std)]
else:
    means = [0.23268045485019684, 0.308925062417984, 0.25266459584236145]
    std = [0.13584113121032715, 0.10670432448387146, 0.11148916929960251]


transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=means,
                     std=std)
    ])
    
images = SatelliteDataset("../../data/key.csv", "../../data/images/", label=label, transformations=transformations)
if type(images.__getitem__(0)[1]) == str:
    classification = True
else:
    classification = False
if classification:
    classes = pd.read_csv("../../data/key.csv")
    n_classes = len(classes.loc[~pd.isna(classes.capture_date),label].unique())
    classes = classes.loc[~pd.isna(classes.capture_date),label].unique()
    class_key = {}
    for i in range(len(classes)):
        class_key[classes[i]] = i
else:
    n_classes = 1
    

# train test split
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
#        self.fc1 = nn.Linear(channels * x_dim * y_dim, 500)
#        self.fc2 = nn.Linear(500, 200)
#        self.fc3 = nn.Linear(200, n_classes)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.linear = nn.Sequential(
            nn.Linear(128 * 128 * 32, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, n_classes),
        )
    
    def forward(self, x):
 #       x = F.relu(self.fc1(x))
 #       x = F.relu(self.fc2(x))
 #       return self.fc3(x)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x


net = Net()
if classification:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(net.parameters(), lr=lr)


# training the model
total_loss = []
for epoch in range(n_epochs):
    epoch_loss = []
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs).squeeze()
        
        if classification:
            labels = [class_key[i] for i in labels]
            labels = torch.tensor(labels)
            loss = criterion(outputs, labels)
            epoch_loss.append(np.mean(loss.data.numpy()))
        else:
            loss = criterion(outputs, labels)
            epoch_loss.append(np.mean(np.sqrt(loss.data.numpy())))
            
        loss.backward()
        optimizer.step()
    
    total_loss.append(np.mean(epoch_loss))
    if classification:
        torch.save(net.state_dict(), 'states/classification_model.pth')
        torch.save(optimizer.state_dict(), 'states/classification_optimizer.pth')
    else:
        torch.save(net.state_dict(), 'states/regression_model.pth')
        torch.save(optimizer.state_dict(), 'states/regression_optimizer.pth')
    # net = Net()
    # net.load_state_dict(torch.load('states/regression_model.pth'))
    print(f'epoch {epoch}: loss {np.mean(epoch_loss)}')

plt.plot(total_loss)


# evaluation
predictions = []
actual = []
net.eval()
for i, data in enumerate(test_loader, 0):
    inputs, labels = data
    actual.append(labels)
    # inputs = inputs.view(-1, channels * x_dim * y_dim)
    outputs = net(inputs)
    if classification:
        outputs = torch.max(outputs, 1)[1].numpy()
        reverse_key = {v: k for k, v in class_key.items()}
        outputs = [reverse_key[i] for i in outputs]
    predictions.append(outputs)
    
if classification:
    accuracy = 0
    for i in range(len(predictions)):
        accuracy += sum([x==y for (x,y) in zip(list(actual[i]), predictions[i])]) / len(predictions[i])
    accuracy = accuracy/len(predictions)
    print(f'{accuracy * 100}% accuracy')
else:
    mse = 0
    me = 0
    for i in range(len(predictions)):
        pred = pd.Series([i[0] for i in predictions[0].detach().numpy()])
        act = pd.Series([i for i in actual[0].numpy()])
        me += abs((pred - act)).mean()
        mse += ((pred - act)**2).mean()
    me /= len(predictions)
    mse /= len(predictions)
    print(f'{round(mse, 0)} MSE')
    print(f'{round(me, 0)} ME')

# recording evaluation
evaluation = pd.read_csv('evaluation.csv')
if classification:
    metric = 'Cross Entropy Loss'
    score = accuracy
else:
    metric = 'RMSE'
    score = np.sqrt(mse)
tmp = pd.DataFrame(
    {'net_schema': str(net),
     'lr': lr,
     'target': label,
     'n_train_images': len(train_index),
     'n_test_images': len(test_index),
     'epochs': n_epochs,
     'metric': metric,
     'score': score,
     'notes': notes
     },
     index=[0])
evaluation.append(tmp).to_csv('evaluation.csv', index=False)

# individual check, have to modify to be input of 4 dimensional convolutional network
index = 67
if classification:
    predicted = torch.max(net(test_loader.dataset.__getitem__(index)[0].unsqueeze(0)), 1).indices.numpy()[0]
    predicted = reverse_key[predicted]
else:
    predicted = round(net(test_loader.dataset.__getitem__(index)[0].unsqueeze(0)).detach().numpy()[0][0], 0)
actual = test_loader.dataset.__getitem__(index)[1]
print(f'Predicted: {predicted}')
print(f'Actual: {actual}')
if not(classification):
    print(f'Off by: {actual - predicted}')
test_loader.dataset.__getitem__(index, transformed_show=True)
