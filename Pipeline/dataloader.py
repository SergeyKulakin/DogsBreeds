import tarfile
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import config as conf

def dataset_load(path=conf.path, path_unzip=conf.path_unzip):
    '''INPUT
            -> path: need path to dataset archive
            -> path_unzip: need path to unzip dir

        OUTPUT
            -> unzip dataset to unzip dir'''

    with tarfile.open(path, "r") as tar:
        tar.extractall(path_unzip)
    return


def get_dataloader(train_transform=conf.train_transform, val_transform=conf.val_transform, path_work=conf.path_work, batch_size=conf.batch_size):
    '''INPUT
            -> train_transform: params transformation train data
            -> val_transform: params transformation val data
            -> path_work: path to work directory
            -> batch_size: number of images in batch
        OUTPUT
            -> train_dataloader, val_dataloader'''

    train_dataset = ImageFolder(
        f'{path_work}/train',
        transform=train_transform)

    val_dataset = ImageFolder(
        f'{path_work}/val',
        transform=val_transform)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader

