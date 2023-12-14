#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:22:42 2023

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

from BioinformaticsNN import TumorClassification, modelFunctions, data_transform



def predict_image(img, model, classes):
    # Convert to a batch of 1
    model.eval()
    xb = img.unsqueeze(0)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return classes[preds[0].item()]


def idx_to_class(dataset, idx):
    """
    This function takes the prediction (which is an index)
    and finds the corresponding class name from the 
    dataset.class_to_idx dictionary
    """

    class_dict = dataset.class_to_idx
    key_lst = list(class_dict.keys())
    val_lst = list(class_dict.values())
    class_name = val_lst.index(idx)
    return key_lst[class_name]


def resize_image(image):
    #convert back to PIL
    image = T.ToPILImage()(image)
    image = T.Resize((512,512))(image)
    image = T.ToTensor()(image)
    return image


if __name__ == '__main__':
    data_dir = '/Users/Jonathan/Desktop/Errythang/MSDAS/Bioinformatics/project/Brain-Tumor-Classification'
    device = torch.device("mps")
    image_size = (32,32)
    means = torch.tensor((0.2031, 0.2031, 0.2031))
    sds = torch.tensor((0.1547, 0.1547, 0.1547))
    transform = data_transform(image_size, means, sds)
    dataset = datasets.ImageFolder(data_dir + '/Testing', transform = transform)
    model = TumorClassification()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    image, label = dataset[356]
    label = idx_to_class(dataset, label)
    print('Predicted:', predict_image(image, model, dataset.classes))
    print('True label', label)
    #resize = T.Resize((512,512))
    image = resize_image(image)
    #img_display = T.ToPILImage(image.permute(1, 2, 0))
    plt.imshow(image.permute(1, 2, 0))
    plt.show()