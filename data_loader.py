import torch
import torch.utils.data as data
import numpy as np

class GetLoader(data.Dataset):
    def __init__(self,Data, Label,transform = None):
        self.Data = Data
        self.Label = Label
        self.transform = transform
    def __len__(self):
        return len(self.Data)

    def __getitem__(self, item):
        data = torch.Tensor(self.Data[item])
        label = int(self.Label[item])
        if self.transform is not None:
            data = self.transform(data)
        return data, label

class GetLoader1(data.Dataset):
    def __init__(self,Data, Label,transform = None):
        self.Data = Data
        self.Label = Label
        self.transform = transform
    def __len__(self):
        return len(self.Data)

    def __getitem__(self, item):
        data = torch.Tensor(self.Data[item])
        #label = int(self.Label[item])
        label = torch.Tensor(self.Label[item])
        if self.transform is not None:
            data = self.transform(data)
        return data, label

class GetLoader2(data.Dataset):
    def __init__(self,Data, Eye,Label,transform = None):
        self.Data = Data
        self.Eye = Eye
        self.Label = Label
        self.transform = transform
    def __len__(self):
        return len(self.Data)

    def __getitem__(self, item):
        data = torch.Tensor(self.Data[item])
        eye = torch.Tensor(self.Eye[item])
        label = int(self.Label[item])
        if self.transform is not None:
            data = self.transform(data)
            eye = self.transform(eye)
        return data, eye, label