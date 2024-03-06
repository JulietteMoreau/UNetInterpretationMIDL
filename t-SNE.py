#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:31:11 2023

@author: moreau
"""

import matplotlib.pyplot as plt
import os
import glob
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from monai.transforms import LoadImageD, EnsureChannelFirstD, ToTensorD, Compose
from monai.data import CacheDataset
import time
import numpy as np
import pandas as pd
import random as rd
from sklearn.manifold import TSNE
import seaborn as sns

from encoder_UNet import UNet

xform = Compose([LoadImageD(('cerveau')),
    EnsureChannelFirstD(('cerveau')),
    ToTensorD(('cerveau'))])

bs = 1

data = []
label = []
coupe=[]

data_dir = 'path/to/data'

test_images = sorted(glob.glob(data_dir + "*.jpg")) 

test_files=[]
for im in range(len(test_images)):
    test_files.append({"cerveau": test_images[im]})

test_ds = CacheDataset(data=test_files, transform=xform, num_workers=10)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

for i, batch in enumerate(test_ds):
    batch['cerveau']=batch['cerveau'][0:1,:,:]

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

encoder = UNet()

state_dict = torch.load("path/to/checkpoint", map_location=lambda storage, loc: storage)

with torch.no_grad():
    encoder.inc.double_conv[0].weight.copy_(state_dict['inc.double_conv.0.weight'])
    encoder.inc.double_conv[1].weight.copy_(state_dict['inc.double_conv.1.weight'])
    encoder.inc.double_conv[3].weight.copy_(state_dict['inc.double_conv.3.weight'])
    encoder.inc.double_conv[4].weight.copy_(state_dict['inc.double_conv.4.weight'])
    encoder.down1.maxpool_conv[1].double_conv[0].weight.copy_(state_dict['down1.maxpool_conv.1.double_conv.0.weight'])
    encoder.down1.maxpool_conv[1].double_conv[1].weight.copy_(state_dict['down1.maxpool_conv.1.double_conv.1.weight'])
    encoder.down1.maxpool_conv[1].double_conv[3].weight.copy_(state_dict['down1.maxpool_conv.1.double_conv.3.weight'])
    encoder.down1.maxpool_conv[1].double_conv[4].weight.copy_(state_dict['down1.maxpool_conv.1.double_conv.4.weight'])
    encoder.down2.maxpool_conv[1].double_conv[0].weight.copy_(state_dict['down2.maxpool_conv.1.double_conv.0.weight'])
    encoder.down2.maxpool_conv[1].double_conv[1].weight.copy_(state_dict['down2.maxpool_conv.1.double_conv.1.weight'])
    encoder.down2.maxpool_conv[1].double_conv[3].weight.copy_(state_dict['down2.maxpool_conv.1.double_conv.3.weight'])
    encoder.down2.maxpool_conv[1].double_conv[4].weight.copy_(state_dict['down2.maxpool_conv.1.double_conv.4.weight'])
    encoder.down3.maxpool_conv[1].double_conv[0].weight.copy_(state_dict['down3.maxpool_conv.1.double_conv.0.weight'])
    encoder.down3.maxpool_conv[1].double_conv[1].weight.copy_(state_dict['down3.maxpool_conv.1.double_conv.1.weight'])
    encoder.down3.maxpool_conv[1].double_conv[3].weight.copy_(state_dict['down3.maxpool_conv.1.double_conv.3.weight'])
    encoder.down3.maxpool_conv[1].double_conv[4].weight.copy_(state_dict['down3.maxpool_conv.1.double_conv.4.weight'])
    encoder.down4.maxpool_conv[1].double_conv[0].weight.copy_(state_dict['down4.maxpool_conv.1.double_conv.0.weight'])
    encoder.down4.maxpool_conv[1].double_conv[1].weight.copy_(state_dict['down4.maxpool_conv.1.double_conv.1.weight'])
    encoder.down4.maxpool_conv[1].double_conv[3].weight.copy_(state_dict['down4.maxpool_conv.1.double_conv.3.weight'])
    encoder.down4.maxpool_conv[1].double_conv[4].weight.copy_(state_dict['down4.maxpool_conv.1.double_conv.4.weight'])

encoder.cuda()

for i, batch in enumerate(test_loader):
    torch.cuda.empty_cache()
    real_CT = batch["cerveau"].type(Tensor)
    image_latent = encoder(real_CT)

    image = Tensor.cpu(image_latent).detach().numpy()
    
    descripteurs = []
    for c in range(512):
        descripteurs.append(np.mean(image[0,c,:,:]))
        
    down4_BN2 = state_dict['down4.maxpool_conv.1.double_conv.4.weight']
    down4_BN2 = down4_BN2.cpu()
    down4_BN2 = down4_BN2.detach().numpy()
    down4_BN2 = list(down4_BN2)
    
    sorted_indices = sorted(range(len(down4_BN2)), key=lambda k: down4_BN2[k])   
    sorted_indices = sorted_indices[:(len(sorted_indices))] #adapt when keeping only a percentage of the input dimensions 
    
    kept_descripteurs = []
    for d in sorted_indices:
        kept_descripteurs.append(descripteurs[d])
    
    data.append(kept_descripteurs)
    if test_files[i]['cerveau'].split('/')[-1][0]=='p':
        label.append(1)
    else:
        label.append(2)
    coupe.append((test_files[i]['cerveau'].split('/')[-1]))
    
feat_cols = ['pixel'+str(i) for i in range(1,len(data[0])+1)]

df = pd.DataFrame(data, columns=feat_cols)

df['y'] = label
df['label'] = df['y'].apply(lambda i: str(i))
df['coupe'] = coupe

data_tsne = df[feat_cols].values

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_results = tsne.fit_transform(data_tsne)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

x=[]
y=[]
for i in range(len(df)):
    if df.loc[i, 'coupe'][0]=='p':
        x.append(df.loc[i, 'tsne-2d-one'])
        y.append(df.loc[i, 'tsne-2d-two'])
        
X = np.sum(x)/len(x)
Y = np.sum(y)/len(y)

print(X,Y)


plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 1),
    data=df,
    legend="full",
    alpha=0.3
)

import pickle

res = {}
for i in range(len(coupe)):
    res[coupe[i]] = [tsne_results[i,0], tsne_results[i,1]]
    
with open('path/to/save/new/dimensions', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)