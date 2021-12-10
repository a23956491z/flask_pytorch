import torchvision
#from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torch.nn.functional as F
from os.path import join
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class NoneTransform(object):
    """ Does nothing to the image, to be used instead of None

    Args:
        image in, image out, nothing is done
    """

    def __call__(self, image):
        return image

def get_default_transform(img_size = 224, norm= True):


    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if norm else NoneTransform(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) if norm else NoneTransform(),
    ])

    return transform_train, transform_test


class Image2FeedDataset(Dataset):

    def __init__(self, root_dir, train=True,transform=None):

        self.transform = transform
        self.parent_folder = 'train' if train else 'val'

        csv = pd.read_csv(join(root_dir, 'folder_label'))
        self.folder_labels = csv.set_index('folder')['label'].to_dict()

        label_list = csv.set_index('folder')['label'].to_list()
        self.label_names = [ label_list[i] for i in sorted(np.unique(label_list, return_index=True)[1])]
        self.classes = len(self.label_names)
        self.imgs = []
        self.labels = []
        for folder in self.folder_labels:
            folder_path = join(root_dir, self.parent_folder, folder)
            label = self.folder_labels[folder]
            for img in os.listdir(folder_path):
                img_path = join(folder_path, img)
                self.imgs.append(img_path)
                self.labels.append(self.label_names.index(label))

        # self.labels = F.one_hot(torch.tensor(self.labels), num_classes=self.classes)

        print(len(self.imgs))


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.imgs[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_feeds_dataset_loader(dir, batch_size):
    tnf_train, tnf_val = get_default_transform()
    train_feed = Image2FeedDataset(root_dir=dir, train=True, transform=tnf_train)
    train_loader = torch.utils.data.DataLoader(train_feed, batch_size=batch_size, shuffle=True, num_workers=8)

    val_feed = Image2FeedDataset(root_dir=dir, train=False, transform=tnf_val)
    val_loader = torch.utils.data.DataLoader(val_feed, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader

if __name__ == '__main__':

    dataset_dir = '/home/divclab/Desktop/Enip/data/feeds'

    transform_train, transform_test = get_default_transform()
    train_dataset = Image2FeedDataset(root_dir=dataset_dir, train=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)

    val_dataset = Image2FeedDataset(root_dir=dataset_dir, train=False, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=8)
    for data, label in train_loader:
        print(data, label)
        break