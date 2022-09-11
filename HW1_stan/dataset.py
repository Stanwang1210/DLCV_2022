import pandas as pd
from os.path import join
from torch.utils.data.dataset import Dataset

import json
import os
import logging
import sys
import json
import glob
from torchvision.io import read_image
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
# im = Image.open('1.jpg')
# im2arr = np.array(im) # im2arr.shape: height x width x channel
# arr2im = Image.fromarray(im2arr)
class P1_Dataset(Dataset):

    def __init__(self, data_dir='hw1_data/p1_data', task='train', transform=None):
        self.data = glob.glob(join(data_dir, f'{task}_50', "*.png"))
        self.process_data = []
        self.labels = []
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        for d in tqdm(self.data):
            self.labels.append(int(d.split('/')[-1].split('_')[0]))
            im = Image.open(d)
            self.process_data.append(self.transform(im))
            del im

        self.class_num = len(np.unique(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):

        return self.process_data[key], self.labels[key], self.data[key].split('/')[-1]
class P2_Dataset(Dataset):

    def __init__(self, data_dir='hw1_data/p2_data', task='train', transform=None):
        self.mask = sorted(glob.glob(join(data_dir, f'{task}', "*mask.png")))
        self.data = sorted(glob.glob(join(data_dir, f'{task}', "*sat.jpg")))
        self.process_data = []
        self.labels = []
        self.pixel_table = {
            (0, 255, 255): 0,
            (255, 255, 0): 1,
            (255, 0, 255): 2,
            (0, 255, 0): 3,
            (0, 0, 255): 4,
            (255, 255, 255): 5,
            (0, 0, 0): 6,
        }
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        for d, m in tqdm(list(zip(self.data, self.mask))):
            im = Image.open(d)
            im2arr = np.array(Image.open(m))
            self.labels.append(self.read_masks(im2arr))
            self.process_data.append(self.transform(im))
            
            del im, im2arr

        self.class_num = len(self.pixel_table)

    def read_masks(self, seg):
        masks = np.zeros((seg.shape[0], seg.shape[1]))
        mask = (seg >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[mask == 2] = 3  # (Green: 010) Forest land 
        masks[mask == 1] = 4  # (Blue: 001) Water 
        masks[mask == 7] = 5  # (White: 111) Barren land 
        masks[mask == 0] = 6  # (Black: 000) Unknown
        # deal with invalid masks
        masks[masks > 6] = 0
        masks[masks < 0] = 0
        masks = torch.LongTensor(masks)
        return masks
    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):

        return self.process_data[key], self.labels[key], self.data[key].split('/')[-1]

if __name__ == '__main__':

    dataset = P2_Dataset()
    print(dataset[0])