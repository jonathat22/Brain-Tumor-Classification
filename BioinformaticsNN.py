#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 18:30:21 2022

@author: Jonathan
"""

from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import os
from torchvision import datasets, transforms


def data_transform(image_size, means, sds):
    transform = T.Compose([
            T.Grayscale(3),
            T.Resize(image_size),
            T.ToTensor(),
            T.RandomRotation(50),
            T.RandomHorizontalFlip(0.4),
            T.RandomVerticalFlip(0.35),
            T.Normalize(means, sds)
            ])
    return transform


def getDataset(data_dir, transform):
    #print(os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    print("Classes:", classes)
    
    glioma_brains = os.listdir(data_dir + "/train/glioma")
    pituitary_brains = os.listdir(data_dir + "/train/pituitary")
    meningioma_brains = os.listdir(data_dir + "/train/meningioma")
    healthy_brains = os.listdir(data_dir + "/train/notumor")
    
    print('No. of training examples for glioma tumor samples:', len(glioma_brains))
    print('No. of training examples for meningioma tumor samples:', len(meningioma_brains))
    print('No. of training examples for pituitary tumor samples:', len(pituitary_brains))
    print('No. of training examples for healthy samples:', len(healthy_brains))
    
    dataset = datasets.ImageFolder(data_dir + '/train', transform = transform)
    return dataset
    
    
def buildDataLoaders(dataset, kwargs):
    random_seed = 42
    torch.manual_seed(random_seed)
    val_size = 307
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])  
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=128, **kwargs) 
    return train_dl, val_dl


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class modelFunctions(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images, labels = images.to(torch.device("mps")), labels.to(torch.device("mps"))
        output = self(images)
        loss = F.cross_entropy(output, labels)
        return loss
    
    def validate(self, batch):
        img, labels = batch
        img, labels = img.to(torch.device("mps")), labels.to(torch.device("mps"))
        output = self(img)
        loss = F.cross_entropy(output, labels)
        acc = accuracy(output, labels)
        return {'validation_loss': loss.detach(), 'validation_acc': acc}
    
    def validation_results(self, outputs):
        batch_losses = [x['validation_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['validation_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'validation_loss': epoch_loss.item(), 
                'validation_acc': epoch_acc.item()}


    def result_per_epoch(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, validation_loss: {:.4f}, validation_acc: {:.4f}".format(
            epoch, result['train_loss'], result['validation_loss'], result['validation_acc']))
        

class TumorClassification(modelFunctions):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 4))
        
    def forward(self, xb):
        return self.network(xb)
    

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validate(batch) for batch in val_loader]
    return model.validation_results(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    epoch_lst = []
    for epoch in range(epochs):
        epoch_lst.append(epoch)
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.result_per_epoch(epoch, result)
        history.append(result)
    return history, epoch_lst


def plot_results(history, epoch_lst):
    accuracies = [x['validation_acc'] for x in history]
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['validation_loss'] for x in history]
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.plot(epoch_lst, accuracies)
    ax1.set_title('Validation Accuracy vs. No. of epochs')
    ax1.set(ylabel='Accuracy')
    ax2.plot(epoch_lst, train_losses)
    ax2.plot(epoch_lst, val_losses)
    ax2.set_title('Losses vs. No. of epochs')
    ax2.set(xlabel='Epochs', ylabel='Loss')
    ax2.legend(['Training', 'Validation'])
    plt.show()

    

if __name__ == '__main__':
    #device = torch.device("mps")
    kwargs = {'num_workers': 4, 'pin_memory': True}
    data_dir = '/Users/Jonathan/Desktop/Errythang/MSDAS/Bioinformatics/project/Brain-Tumor-Classification'
    image_size = (32,32)
    means = torch.tensor((0.2031, 0.2031, 0.2031))
    sds = torch.tensor((0.1547, 0.1547, 0.1547))
    transform = data_transform(image_size, means, sds)
    dataset = getDataset(data_dir, transform)
    train_dl, val_dl = buildDataLoaders(dataset, kwargs)
    model = TumorClassification().to(torch.device("mps"))
    print("Evaluation Before Training:")
    print(evaluate(model, val_dl))
    num_epochs = 30
    opt_func = torch.optim.Adam
    lr = 0.001
    results, epoch_lst = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    plot_results(results, epoch_lst)
    torch.save(model.state_dict(), '/Users/Jonathan/Desktop/Errythang/MSDAS/Bioinformatics/project/Brain-Tumor-Classification/model.pth')