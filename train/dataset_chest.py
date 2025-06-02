import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def img_transform_test():
    transform = transforms.Compose([
        transforms.Resize([224,224]),  # 3 is bicubic
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform


class data_test(Dataset):
    def __init__(self, dir,transform=None):
        self.dir = dir
        # self.image_names=[]
        self.image_names=glob.glob(dir+'/*/*')
        if transform==None:
            self.transforms = img_transform_test()
        else:
            self.transforms=transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, id):
        filename = self.image_names[id]
        label = filename.split('/')[-2]
        img = Image.open(filename).convert('RGB')
        img = self.transforms(img)
        if label.startswith('ABNORMAL') or label.startswith('abnormal'):
            label = int(1)
        else:
            label = int(0)
        return img, label,filename

#