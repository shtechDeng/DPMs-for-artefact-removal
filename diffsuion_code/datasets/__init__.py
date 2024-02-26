import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Subset
import numpy as np

from skimage.transform import resize
from skimage.exposure import rescale_intensity
import nibabel as nib
from torch.utils.data import Dataset

import shutil
import logging
import random
import time

def proc_nib_data(nib_data):
    p10 = np.percentile(nib_data, 10)
    p99 = np.percentile(nib_data, 99.9)

    nib_data[nib_data<p10] = p10
    nib_data[nib_data>p99] = p99

    m = np.mean(nib_data, axis=(0, 1, 2))
    s = np.std(nib_data, axis=(0, 1, 2))
    nib_data = (nib_data - m) / s

    nib_data = torch.tensor(nib_data, dtype=torch.float32)

    return nib_data

class MRI_Data(Dataset):
    def __init__(
            self,
            data_path,
            transform = None
    ):
        data_list = []
        self.data_len = len(data_path)
        self.transform = transform
        for subj in os.listdir(data_path):
            for age in os.listdir(os.path.join(data_path, subj)):
                for nii in os.listdir(os.path.join(data_path, subj, age)):
                    if('T1skull' in nii): 
                        T1_path = os.path.join(data_path, subj, age, nii)
            data_list.append(T1_path)
        self.data_list = data_list

    def __getitem__(self, index):
        T1_path = self.data_list[index]
        T1_data = nib.load(T1_path).get_fdata()
        
        p10 = np.percentile(T1_data, 10)
        p99 = np.percentile(T1_data, 99)
        T1_data = rescale_intensity(T1_data, in_range=(p10, p99), out_range=(0, 1))
        m = np.mean(T1_data, axis=(0, 1, 2))
        s = np.std(T1_data, axis=(0, 1, 2))
        T1_data = (T1_data - m) / s
        T1_data = torch.tensor(T1_data, dtype=torch.float32)
        return T1_data.unsqueeze(0), T1_path

    def __len__(self):
        return len(self.data_list)

class MRI_Data2(Dataset):
    def __init__(
            self,
            data_path,
            transform = None
    ):

        data_list = []
        self.transform = transform
        with open(data_path) as f:
            for row in f.readlines():
                data_path = row
                data_path = data_path.strip("\n")
                data_list.append(data_path)
        self.data_list = data_list

    def __getitem__(self, index):
        aT1_path = self.data_list[index]
        aT1_path = aT1_path.strip('\n')
        aT1_data = nib.load(aT1_path).get_fdata()
        x, y, z = aT1_data.shape
        p10 = np.percentile(aT1_data, 10)
        p99 = np.percentile(aT1_data, 99)
        aT1_data = rescale_intensity(aT1_data, in_range=(p10, p99), out_range=(0, 1))
        m = np.mean(aT1_data, axis=(0, 1, 2))
        s = np.std(aT1_data, axis=(0, 1, 2))
        aT1_data = (aT1_data - m) / s
        aT1_data = torch.tensor(aT1_data, dtype=torch.float32)

        return aT1_data.unsqueeze(0), aT1_path

    def __len__(self):
        return len(self.data_list)
   
def get_dataset(args, config):

    dataset, test_dataset = MRI_Data(config.data.data_path), None
    return dataset, test_dataset
