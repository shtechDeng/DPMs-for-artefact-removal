import os
import torch
import numbers
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from datasets.celeba import CelebA
from datasets.ffhq import FFHQ
from datasets.lsun import LSUN
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

class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

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
        
# def proc_month(month):
#     if month == '4yr':
#         month = '48mo'
#     elif month == '2wk':
#         month = '1mo'
#     elif month == '09mo':
#         month = '9mo'

#     return int(month[:-2])

def nifti_rotate(src_img, xrot, yrot, zrot, save_img=None):
    home_path = os.getenv('HOME')
    # home_path = "home_path"
    temp_fold = os.path.join(home_path, '.nifti_rotate', str(int(time.time() * 1000000)) + '_' + str(random.randint(0, 1000000)))
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)

    xrot = xrot*np.pi/180
    yrot = yrot*np.pi/180
    zrot = zrot*np.pi/180

    xmat = np.eye(3)
    ymat = np.eye(3)
    zmat = np.eye(3)

    xmat[1, 1] =  np.cos(xrot)
    xmat[1, 2] = -np.sin(xrot)
    xmat[2, 1] =  np.sin(xrot)
    xmat[2, 2] =  np.cos(xrot)

    ymat[0, 0] =  np.cos(yrot)
    ymat[0, 2] =  np.sin(yrot)
    ymat[2, 0] = -np.sin(yrot)
    ymat[2, 2] =  np.cos(yrot)

    zmat[0, 0] =  np.cos(zrot)
    zmat[0, 1] = -np.sin(zrot)
    zmat[1, 0] =  np.sin(zrot)
    zmat[1, 1] =  np.cos(zrot)

    mat = np.dot(np.dot(zmat, ymat), xmat)
    mat_all = np.eye(4)
    mat_all[:3, :3] = mat

    np.savetxt(os.path.join(temp_fold, 'rotate.mat'), mat_all, fmt="%.10f", delimiter="  ")

    if not save_img:
        save_img = os.path.join(temp_fold, 'rotate.nii.gz')

    cmd_applywarp = f"applywarp -i {src_img} -r {src_img} -o {save_img} --premat={os.path.join(temp_fold, 'rotate.mat')} --interp=spline"
    os.system(cmd_applywarp)

    ret_data = nib.load(save_img).get_fdata()
    shutil.rmtree(temp_fold)

    return ret_data

class BCP_data1(Dataset):
    def __init__(
        self, 
        data_path,
        data_file_name,
    ):
        self.data_path = data_path
        self.data_file_name = data_file_name

        self.data_list = []
        self.data_argu_main = {}
        self.data_argu_ref = {}
        self.sub2month = {}
        count = 0
        for sub in os.listdir(data_path):
            month_list = os.listdir(os.path.join(data_path, sub))
            if month_list == []:
                continue
            count += len(month_list)

            for month in month_list:
                for ref_month in month_list:
                    if month != ref_month:
                        self.data_list.append((sub, month, ref_month))

        logging.info("Number of data = {}, number of pear data = {}".format(count, len(self.data_list)))

    def __getitem__(self, index):
        main_sub, main_month ,ref_month = self.data_list[index]
        # ref_month = random.choice(self.sub2month[main_sub])

        main_data_path = os.path.join(self.data_path, main_sub, main_month, self.data_file_name)
        ref_data_path = os.path.join(self.data_path, main_sub, ref_month, self.data_file_name)

        self.data_argu_main[main_data_path] = self.data_argu_main.get(main_data_path, [])
        self.data_argu_ref[ref_data_path] = self.data_argu_ref.get(ref_data_path, [])

        if random.random() < 0.05:
            main_data = nifti_rotate(main_data_path, random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
            self.data_argu_main[main_data_path].append(main_data)
        else:
            if len(self.data_argu_main[main_data_path]) > 10:
                main_data = self.data_argu_main[main_data_path].pop(random.choice(list(range(len(self.data_argu_main[main_data_path])))))
            elif len(self.data_argu_main[main_data_path]) > 0:
                main_data = random.choice(self.data_argu_main[main_data_path])
            else:
                main_data = nib.load(main_data_path).get_fdata()
        main_data = proc_nib_data(main_data)

        if random.random() < 0.1:
            ref_data = nifti_rotate(ref_data_path, random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3))
            self.data_argu_ref[ref_data_path].append(ref_data)
        else:
            if len(self.data_argu_ref[ref_data_path]) > 10:
                ref_data = self.data_argu_ref[ref_data_path].pop(random.choice(list(range(len(self.data_argu_ref[ref_data_path])))))
            elif len(self.data_argu_ref[ref_data_path]) > 0:
                ref_data = random.choice(self.data_argu_ref[ref_data_path])
            else:
                ref_data = nib.load(ref_data_path).get_fdata()
        ref_data = proc_nib_data(ref_data)
        
        return main_data.unsqueeze(0), ref_data.unsqueeze(0), proc_month(main_month)

    def __len__(self):
        return len(self.data_list)
    
class MRI_Data(Dataset):
    def __init__(
            self,
            data_path, #被试文件夹路径
            # transform = transforms.Compose([transforms.Resize(208, 300, 320)])
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
        # aT1=(238,288,180)  T1_regred=(238,288,180) (224, 240, 208)
        # aT1_data = resize(aT1_data, (224, 288, 176))
        # T1_data = resize(T1_data, (224, 288, 176))
        # x, y, z = T1_data.shape
        #需不需要同时对aT1和T1做标准化？
        
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

class MRI_Data_valid(Dataset):
    def __init__(
            self,
            data_path, #被试文件夹路径
            # transform = transforms.Compose([transforms.Resize(208, 300, 320)])
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

        #aT1=(238,288,180)  T1_regred=(238,288,180)
        #aT1_data = resize(aT1_data, (224, 288, 176))

        x, y, z = aT1_data.shape
        #需不需要同时对aT1和T1做标准化？

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
   
class MRI_Data_inference(Dataset):
    def __init__(
            self,
            data_path, #被试文件夹路径
            # transform = transforms.Compose([transforms.Resize(208, 300, 320)])
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
        # aT1=(238,288,180)  T1_regred=(238,288,180) (224, 240, 208)
        # aT1_data = resize(aT1_data, (224, 288, 176))
        # T1_data = resize(T1_data, (224, 288, 176))
        # x, y, z = T1_data.shape
        #需不需要同时对aT1和T1做标准化？
        
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

def get_dataset(args, config):

    dataset, test_dataset = MRI_Data_inference(config.data.data_path), None


    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
