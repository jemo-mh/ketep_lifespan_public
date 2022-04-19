import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os


class Life_Span_Dataset(Dataset):
    def __init__(self, train):
        if train==True:
            txt ='Dataset/Train2.txt'
        else:
            txt = 'Dataset/Test2.txt'
            
        self.f_ = np.loadtxt(txt, 'int', delimiter=',',skiprows=1)

        # self.transform = torch.Tensor
    def __len__(self):
        return self.f_.shape[0]

    def __getitem__(self, index):
        raw = self.f_[index]
        data= torch.Tensor(raw[:-1])
        label = torch.Tensor([raw[-1]])
        return data, label



class HI_Dataset(Dataset):
    def __init__(self, train):
        if train==True:
            txt ='Dataset/HI_Train3.txt'
        else:
            txt = 'Dataset/HI_Test3.txt'
            
        self.f_ = np.loadtxt(txt, 'str', delimiter=',',skiprows=1)

        # self.transform = torch.Tensor
    def __len__(self):
        return self.f_.shape[0]

    def __getitem__(self, index):
        raw = self.f_[index]
        # data = torch.Tensor(np.array_split(raw[:-1],3))
        data= torch.Tensor(raw[:-1].astype("uint8"))
        label = torch.Tensor([raw[-1].astype("float")])
        # label = torch.Tensor(raw[-1])
        return data, label


class LSDataset(Dataset):
    def __init__(self, train):

        if train==True:
            txt ='Dataset/TrainFFF.txt'
        else:
            txt = 'Dataset/TestFFF.txt'
        self.f_ = np.loadtxt(txt, 'str', delimiter=',',skiprows=1)
    
    def __len__(self):
        return self.f_.shape[0]
    
    def __getitem__(self, idx):
        raw = self.f_[idx]
        data = torch.Tensor(raw[:-2].astype("float"))
        hi = torch.Tensor([raw[-2].astype("float")])
        flsp=torch.Tensor([raw[-1].astype("float")])
        return data, hi, flsp 

class LSDataset_val(Dataset):
    def __init__(self):
        # txt = 'Dataset/Integrated_val.txt'
        txt='/mnt/e/Workspace/jm/Projects/baby_eval/utils/Lifespan_validation0414.txt'
        self.f_ = np.loadtxt(txt, 'str', delimiter=',',skiprows=1)
    
    def __len__(self):
        return self.f_.shape[0]
    
    def __getitem__(self, idx):
        raw = self.f_[idx]
        data = torch.Tensor(raw[:-2].astype("float"))
        hi = torch.Tensor([raw[-2].astype("float")])
        flsp=torch.Tensor([raw[-1].astype("float")])
        return data, hi, flsp 