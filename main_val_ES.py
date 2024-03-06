#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 14:25:22 2022

@author: moreau
"""

import os
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
import sys
from monai.transforms import LoadImageD, EnsureChannelFirstD, ToTensorD, Compose
from monai.data import CacheDataset

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#################################################################################################################################load data

outdir = sys.argv[1]
dossiers = sys.argv[2]
dossier_val = sys.argv[3]
dossier_test = sys.argv[4]

dossiers = dossiers.split(", ")

if not os.path.exists(outdir):
    os.makedirs(outdir)   

KEYS = ("cerveau", "GT")

train_dirs=[]
train_dirs_label = []
for i in range(len(dossiers)):
    train_dirs.append(dossiers[i]+'/CT/')
    train_dirs_label.append(dossiers[i]+'/GT/')

val_dir = dossier_val + '/CT/'
val_dir_label = dossier_val + '/GT/'
test_dir = dossier_test + '/CT/'
test_dir_label = dossier_test + '/GT/'


train_images = []
train_labels = []
for CT in train_dirs:
    train_images = train_images + sorted(glob.glob(CT + "*.jpg"))
for GT in train_dirs_label:
    train_labels = train_labels + sorted(glob.glob(GT + "*.png"))

print(val_dir)
val_images = sorted(glob.glob(val_dir + "*.jpg"))
val_labels = sorted(glob.glob(val_dir_label + "*.png"))

test_images = sorted(glob.glob(test_dir + "*.jpg"))
test_labels = sorted(glob.glob(test_dir_label + "*.png"))

train_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(train_images, train_labels)
]
val_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(val_images, val_labels)
]
test_files = [
    {"cerveau": image_name, "GT": label_name} for image_name, label_name in zip(test_images, test_labels)
]
print("Number Train files: "+str(len(train_files)))
print("Number val files: "+str(len(val_files)))
print("Number test files: "+str(len(test_files)))

# Create dataloaders
xform = Compose([LoadImageD(KEYS),
    EnsureChannelFirstD(KEYS),
    ToTensorD(KEYS)])

bs = 12
train_ds = CacheDataset(data=train_files, transform=xform, num_workers=10)
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_ds = CacheDataset(data=val_files, transform=xform, num_workers=10)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=True)
test_ds = CacheDataset(data=test_files, transform=xform, num_workers=10)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)


#bidouille pour avoir un channel d'un image NB qui est reconnue comme RGB
for i, batch in enumerate(train_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]
    batch['GT']=batch['GT'][0:1,:,:]
    I = batch['GT']>1
    batch['GT'] = I.long()
for i, batch in enumerate(val_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]
    batch['GT']=batch['GT'][0:1,:,:]
    I = batch['GT']>1
    batch['GT'] = I.long()
for i, batch in enumerate(test_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]
    batch['GT']=batch['GT'][0:1,:,:]
    I = batch['GT']>1
    batch['GT'] = I.long()


################################################################################################################paramètres d'apprentissage
# Parameters for Adam optimizer
lr = 0.002
beta1 = 0.5
beta2 = 0.999


# Number of epochs
num_epoch = 200
patience = 5


# ################################################################################################################generator Unet
from generator_UNet import UNet
# Summary of the generator
summary(UNet().cuda(), (1, 192, 192))

from train_generator_val_ES import train_net
generator = train_net(train_loader, train_ds, val_loader, outdir, patience=patience, num_epoch=num_epoch, lr=lr, beta1=beta1, beta2=beta2)


from evaluation import evaluate_generator

print(' ')
print(evaluate_generator(generator, train_loader, test_loader, outdir))
